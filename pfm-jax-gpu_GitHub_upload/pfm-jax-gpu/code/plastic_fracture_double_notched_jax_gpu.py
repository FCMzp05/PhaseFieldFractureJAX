#!/usr/bin/env python3
"""Standalone JAX/GPU port of plastic_fracture_double_notched.jl.

The implementation intentionally follows the reference algorithm, including
its P1 triangles, one cellwise plastic state, first degree-two quadrature-point
damage in the return map, displacement-only plastic Picard iteration, repeated
phase-field solve, and reaction evaluation before the new damage is applied.

Dependencies: jax, numpy, gmsh, meshio (matplotlib is optional).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path


def _requested_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    args, _ = parser.parse_known_args()
    return args.platform


_PLATFORM = _requested_platform()
if _PLATFORM != "auto":
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if _PLATFORM == "gpu" else "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "JAX is required (for NVIDIA/WSL, install e.g. 'jax[cuda13]')."
    ) from exc


# Exact reference constants.
W = 10.0
HEIGHT = 10.0
G = 70_300.0
K_BULK = 136_500.0
LAMBDA = K_BULK - 2.0 * G / 3.0
E_MODULUS = 9.0 * K_BULK * G / (3.0 * K_BULK + G)
POISSON = (3.0 * K_BULK - 2.0 * G) / (2.0 * (3.0 * K_BULK + G))
SIGMA_Y0 = 443.0
H_HARD = 300.0
LC = 0.03
GC = 20.9
ETA = 1.0e-4
THICKNESS = 1.0
H_FINE = 0.07
H_COARSE = 0.45
N_STEPS = 400
MAX_STAGGER = 30
STAGGER_TOL = 1.0e-6
STOP_FORCE_RATIO = 0.05
STOP_AFTER_STEPS = 6
SNAPSHOT_DAMAGE_LEVELS = (0.20, 0.50, 0.80, 0.95, 0.99)


@dataclass(frozen=True)
class GeometryCase:
    tag: str
    title: str
    bl_radius: float
    tr_radius: float
    u_total: float
    left_scale: float


CASES = (
    GeometryCase("paper", "Paper baseline", 2.5, 2.0, 0.40, 1.00),
    GeometryCase("load_bias", "Load-biased case", 2.5, 2.0, 0.40, 0.70),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU double-notched ductile phase-field solver"
    )
    parser.add_argument("--case", choices=("all", "paper", "load_bias"), default="paper")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: outputs/results_<device>/plastic_fracture_double_notched/jax)",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="reuse a Gmsh mesh (otherwise generate the exact reference mesh per case)",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument(
        "--u-total", type=float, default=None, help="override case total displacement"
    )
    parser.add_argument("--max-stagger", type=int, default=MAX_STAGGER)
    parser.add_argument("--stagger-tol", type=float, default=STAGGER_TOL)
    parser.add_argument("--plastic-maxiter", type=int, default=10)
    parser.add_argument("--plastic-tol", type=float, default=1.0e-5)
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=5000)
    parser.add_argument("--linear-residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--vtu-every", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run one 0.001 mm substep with two plastic/stagger iterations",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="disable VTK and plotting so timing covers solver/checkpoint/QoI only",
    )
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    return parser


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def create_mesh(filename: Path, geom: GeometryCase) -> None:
    """Generate the same Gmsh GEO model, fields, and physical tags as Julia."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh is required: python -m pip install gmsh") from exc

    filename.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("Ductile_double_notched")
        geo = gmsh.model.geo
        pt = 1

        def point(x, y, h):
            nonlocal pt
            tag = pt
            geo.addPoint(x, y, 0.0, h, tag)
            pt += 1
            return tag

        p_bb = point(geom.bl_radius, 0.0, H_FINE)
        p_br = point(W, 0.0, H_COARSE)
        p_rr = point(W, HEIGHT - geom.tr_radius, H_FINE)
        p_tr = point(W, HEIGHT, H_COARSE)
        p_tt = point(HEIGHT - geom.tr_radius, HEIGHT, H_FINE)
        p_tl = point(0.0, HEIGHT, H_COARSE)
        p_ll = point(0.0, geom.bl_radius, H_FINE)
        p_cbl = point(0.0, 0.0, H_FINE)
        p_ctr = point(W, HEIGHT, H_FINE)
        root2 = np.sqrt(2.0)
        p_ds = point(geom.bl_radius / root2, geom.bl_radius / root2, H_FINE)
        p_de = point(
            W - geom.tr_radius / root2,
            HEIGHT - geom.tr_radius / root2,
            H_FINE,
        )

        line = 1

        def straight(a, b):
            nonlocal line
            tag = line
            geo.addLine(a, b, tag)
            line += 1
            return tag

        bottom = straight(p_bb, p_br)
        right = straight(p_br, p_rr)
        arc_tr = line
        geo.addCircleArc(p_rr, p_ctr, p_tt, arc_tr)
        line += 1
        top = straight(p_tt, p_tl)
        left = straight(p_tl, p_ll)
        arc_bl = line
        geo.addCircleArc(p_ll, p_cbl, p_bb, arc_bl)
        line += 1
        diagonal = straight(p_ds, p_de)
        geo.addCurveLoop([bottom, right, arc_tr, top, left, arc_bl], 1)
        geo.addPlaneSurface([1], 1)
        # Gmsh physical groups operate on model entities; synchronize the GEO
        # kernel first.  This does not alter the reference geometry or tags.
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        for tag, curve, name in (
            (1, bottom, "bottom"),
            (2, right, "right"),
            (3, top, "top"),
            (4, left, "left"),
        ):
            gmsh.model.addPhysicalGroup(1, [curve], tag)
            gmsh.model.setPhysicalName(1, tag, name)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(
            1, "CurvesList", [diagonal, arc_bl, arc_tr]
        )
        gmsh.model.mesh.field.setNumber(1, "Sampling", 160)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", H_FINE)
        gmsh.model.mesh.field.setNumber(2, "LcMax", H_COARSE)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2.00)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 9.00)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(filename))
    finally:
        gmsh.finalize()


def load_mesh(filename: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    mesh = meshio.read(str(filename))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangles = [
        np.asarray(block.data, dtype=np.int64)
        for block in mesh.cells
        if block.type == "triangle"
    ]
    if not triangles:
        raise RuntimeError("mesh contains no linear triangles")
    tri = np.concatenate(triangles)
    boundary_edges: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}
    physical = mesh.cell_data.get("gmsh:physical", [])
    for iblock, block in enumerate(mesh.cells):
        if block.type != "line" or iblock >= len(physical):
            continue
        for edge, tag in zip(block.data, physical[iblock]):
            if int(tag) in boundary_edges:
                boundary_edges[int(tag)].append(np.asarray(edge, dtype=np.int64))
    for tag in boundary_edges:
        if not boundary_edges[tag]:
            raise RuntimeError(f"mesh has no physical boundary edges with tag {tag}")

    xy = nodes[tri]
    edge_1 = xy[:, 1] - xy[:, 0]
    edge_2 = xy[:, 2] - xy[:, 0]
    twice_area = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    flip = twice_area < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    xy = nodes[tri]
    edge_1 = xy[:, 1] - xy[:, 0]
    edge_2 = xy[:, 2] - xy[:, 0]
    areas = 0.5 * (
        edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    )
    if np.any(areas <= 0.0):
        raise RuntimeError("mesh contains zero-area triangles")
    edges = {
        tag: np.asarray(values, dtype=np.int64)
        for tag, values in boundary_edges.items()
    }
    return nodes, tri, areas, edges


def find_boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lookup: dict[tuple[int, int], int] = {}
    for cell, element in enumerate(tri):
        for a, b in ((element[0], element[1]), (element[1], element[2]), (element[2], element[0])):
            lookup[tuple(sorted((int(a), int(b))))] = cell
    return np.asarray(
        [lookup[tuple(sorted((int(edge[0]), int(edge[1]))))] for edge in edges],
        dtype=np.int64,
    )


def preprocess(nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray):
    x, y = nodes[:, 0], nodes[:, 1]
    i0, i1, i2 = tri.T
    gx = np.stack([y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]], axis=1)
    gy = np.stack([x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]], axis=1)
    gx /= 2.0 * areas[:, None]
    gy /= 2.0 * areas[:, None]
    nel = len(tri)
    bmat = np.zeros((nel, 3, 6), dtype=np.float64)
    dofs = np.empty((nel, 6), dtype=np.int64)
    for a in range(3):
        bmat[:, 0, 2 * a] = gx[:, a]
        bmat[:, 1, 2 * a + 1] = gy[:, a]
        bmat[:, 2, 2 * a] = gy[:, a]
        bmat[:, 2, 2 * a + 1] = gx[:, a]
        dofs[:, 2 * a] = 2 * tri[:, a]
        dofs[:, 2 * a + 1] = 2 * tri[:, a] + 1
    cmat = np.asarray(
        [
            [LAMBDA + 2.0 * G, LAMBDA, 0.0],
            [LAMBDA, LAMBDA + 2.0 * G, 0.0],
            [0.0, 0.0, G],
        ]
    )
    ke0 = areas[:, None, None] * np.einsum(
        "eia,ij,ejb->eab", bmat, cmat, bmat, optimize=True
    )
    mass0 = np.asarray([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float) / 12.0
    mass = areas[:, None, None] * mass0
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    return bmat, dofs, ke0, mass, grad


def make_kernels(
    *,
    nodes_np,
    tri_np,
    areas_np,
    edges_np,
    bmat_np,
    dofs_np,
    ke0_np,
    mass_np,
    grad_np,
    free_np,
    cg_tol,
    cg_maxiter,
):
    dtype = jnp.float64
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    tri = jnp.asarray(tri_np)
    areas = jnp.asarray(areas_np, dtype=dtype)
    bmat = jnp.asarray(bmat_np, dtype=dtype)
    dofs = jnp.asarray(dofs_np)
    ke0 = jnp.asarray(ke0_np, dtype=dtype)
    mass = jnp.asarray(mass_np, dtype=dtype)
    grad = jnp.asarray(grad_np, dtype=dtype)
    free = jnp.asarray(free_np, dtype=dtype)
    ndof = len(free_np)
    nnode = len(nodes_np)

    top_edges = jnp.asarray(edges_np[3])
    left_edges = jnp.asarray(edges_np[4])
    top_cells = jnp.asarray(find_boundary_cells(tri_np, edges_np[3]))
    left_cells = jnp.asarray(find_boundary_cells(tri_np, edges_np[4]))
    top_lengths = jnp.linalg.norm(
        nodes[top_edges[:, 1]] - nodes[top_edges[:, 0]], axis=1
    )
    left_lengths = jnp.linalg.norm(
        nodes[left_edges[:, 1]] - nodes[left_edges[:, 0]], axis=1
    )

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(values.reshape(-1))

    def degradation_integral(damage):
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        mean_d = jnp.mean(dv, axis=1)
        mean_d2 = (
            jnp.sum(dv * dv, axis=1)
            + dv[:, 0] * dv[:, 1]
            + dv[:, 1] * dv[:, 2]
            + dv[:, 2] * dv[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * mean_d + mean_d2 + ETA

    def first_qp_damage(damage):
        # Gridap's first point of its degree-two triangle quadrature.
        dv = damage[tri]
        return jnp.clip((2.0 * dv[:, 0] + 0.5 * dv[:, 1] + 0.5 * dv[:, 2]) / 3.0, 0.0, 1.0)

    def kmv(x, weights):
        local = weights[:, None] * jnp.einsum("eij,ej->ei", ke0, x[dofs])
        return scatter(dofs, local, ndof)

    @jax.jit
    def solve_u(damage, epsp, boundary, x0):
        weights = degradation_integral(damage)
        sigma_p = jnp.stack(
            (
                (LAMBDA + 2.0 * G) * epsp[:, 0] + LAMBDA * epsp[:, 1],
                LAMBDA * epsp[:, 0] + (LAMBDA + 2.0 * G) * epsp[:, 1],
                2.0 * G * epsp[:, 2],
            ),
            axis=1,
        )
        rhs_local = weights[:, None] * areas[:, None] * jnp.einsum(
            "eji,ej->ei", bmat, sigma_p
        )
        plastic_rhs = scatter(dofs, rhs_local, ndof)

        def raw(x):
            return kmv(x, weights)

        rhs = free * (plastic_rhs - raw(boundary))

        def operator(x):
            return free * raw(free * x) + (1.0 - free) * x

        diagonal = scatter(
            dofs,
            weights[:, None] * jnp.diagonal(ke0, axis1=1, axis2=2),
            ndof,
        )
        diagonal = free * diagonal + (1.0 - free)
        correction, _ = cg(
            operator,
            rhs,
            x0=free * x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, jnp.finfo(dtype).tiny),
        )
        displacement = boundary + free * correction
        residual = jnp.linalg.norm(operator(correction) - rhs) / (
            jnp.linalg.norm(rhs) + 1.0e-30
        )
        return displacement, residual

    @jax.jit
    def return_map(displacement, damage, epsp, epsp33, epspeq, update):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        ee11 = strain[:, 0] - epsp[:, 0]
        ee22 = strain[:, 1] - epsp[:, 1]
        ee12 = 0.5 * strain[:, 2] - epsp[:, 2]
        ee33 = -epsp33
        trace = ee11 + ee22 + ee33
        s110 = LAMBDA * trace + 2.0 * G * ee11
        s220 = LAMBDA * trace + 2.0 * G * ee22
        s330 = LAMBDA * trace + 2.0 * G * ee33
        s120 = 2.0 * G * ee12
        pressure = (s110 + s220 + s330) / 3.0
        q11, q22, q33, q12 = s110 - pressure, s220 - pressure, s330 - pressure, s120
        seq = jnp.sqrt(jnp.maximum(1.5 * (q11**2 + q22**2 + q33**2 + 2.0 * q12**2), 0.0))
        sy0 = SIGMA_Y0 + H_HARD * epspeq
        plastic = (seq > (1.0 + 1.0e-6) * sy0) & (seq >= 1.0e-10)

        def newton_body(_, state):
            deq, sy, active = state
            rhs = seq - 3.0 * G * deq - sy
            deq_new = deq + rhs / (3.0 * G + H_HARD)
            sy_new = SIGMA_Y0 + H_HARD * (epspeq + deq_new)
            keep = active & (jnp.abs(rhs) >= 1.0e-6 * SIGMA_Y0)
            return jnp.where(active, deq_new, deq), jnp.where(active, sy_new, sy), keep

        deq, sy, _ = jax.lax.fori_loop(
            0,
            20,
            newton_body,
            (jnp.zeros_like(seq), sy0, plastic),
        )
        deq = jnp.where(plastic, deq, 0.0)
        inv_seq = jnp.where(plastic, 1.0 / jnp.maximum(seq, 1.0e-30), 0.0)
        f11, f22, f33, f12 = q11 * inv_seq, q22 * inv_seq, q33 * inv_seq, q12 * inv_seq
        ds11, ds22, ds33, ds12 = 1.5 * f11 * deq, 1.5 * f22 * deq, 1.5 * f33 * deq, 1.5 * f12 * deq
        sn11 = jnp.where(plastic, f11 * sy + pressure, s110)
        sn22 = jnp.where(plastic, f22 * sy + pressure, s220)
        sn33 = jnp.where(plastic, f33 * sy + pressure, s330)
        sn12 = jnp.where(plastic, f12 * sy, s120)
        een11, een22 = ee11 - ds11, ee22 - ds22
        een33, een12 = ee33 - ds33, ee12 - ds12
        sse = 0.5 * (
            sn11 * een11 + sn22 * een22 + sn33 * een33 + 2.0 * sn12 * een12
        )
        gqp = (1.0 - first_qp_damage(damage)) ** 2 + ETA
        stress = gqp[:, None] * jnp.stack((sn11, sn22, sn12), axis=1)
        new_epsp = epsp + jnp.stack((ds11, ds22, ds12), axis=1)
        new_epsp33 = epsp33 + ds33
        new_epspeq = epspeq + deq
        return (
            stress,
            sse,
            jnp.where(update, new_epsp, epsp),
            jnp.where(update, new_epsp33, epsp33),
            jnp.where(update, new_epspeq, epspeq),
        )

    @jax.jit
    def solve_d(history, x0):
        element = GC * LC * grad + (GC / LC + 2.0 * history)[:, None, None] * mass
        rhs = scatter(
            tri,
            (2.0 * history * areas / 3.0)[:, None] * jnp.ones((1, 3), dtype=dtype),
            nnode,
        )

        def operator(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", element, x[tri]), nnode)

        diagonal = scatter(tri, jnp.diagonal(element, axis1=1, axis2=2), nnode)
        result, _ = cg(
            operator,
            rhs,
            x0=x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, jnp.finfo(dtype).tiny),
        )
        residual = jnp.linalg.norm(operator(result) - rhs) / (
            jnp.linalg.norm(rhs) + 1.0e-30
        )
        return result, residual

    @jax.jit
    def reaction(stress):
        # Outward normals are (0,1) on top and (-1,0) on left.
        fy = jnp.sum(stress[top_cells, 1] * top_lengths)
        fy -= jnp.sum(stress[left_cells, 2] * left_lengths)
        return jnp.abs(fy * THICKNESS) / 1000.0

    return solve_u, return_map, solve_d, reaction


CSV_FIELDS = (
    "step",
    "time_s",
    "step_time_s",
    "u_mm",
    "F_kN",
    "d_max",
    "epspeq_max",
    "plastic_iterations",
    "stagger_iterations",
    "stagger_residual",
    "u_linear_residual",
    "d_linear_residual",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, step: int, state: tuple, rows: list[dict]) -> None:
    displacement, damage, history, epsp, epsp33, epspeq = state
    np.savez_compressed(
        path,
        step=np.asarray(step),
        displacement=displacement,
        damage=damage,
        history=history,
        epsp=epsp,
        epsp33=epsp33,
        epspeq=epspeq,
        rows_json=np.asarray(json.dumps(rows)),
    )


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    state = tuple(
        np.asarray(data[name])
        for name in ("displacement", "damage", "history", "epsp", "epsp33", "epspeq")
    )
    return int(data["step"]), state, json.loads(str(data["rows_json"]))


def save_vtu(path, nodes, tri, displacement, damage, history, epspeq):
    import meshio

    points = np.column_stack((nodes, np.zeros(len(nodes))))
    vectors = np.column_stack(
        (displacement[0::2], displacement[1::2], np.zeros(len(nodes)))
    )
    meshio.write_points_cells(
        str(path),
        points,
        [("triangle", tri)],
        point_data={"d": damage, "u": vectors},
        cell_data={"epspeq": [epspeq], "H": [history]},
    )


def save_plot(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["u_mm"] for r in rows], [r["F_kN"] for r in rows], lw=2.2)
    ax.set(xlabel="Displacement u (mm)", ylabel="Reaction force (kN)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def host_state(state):
    return tuple(np.asarray(jax.device_get(value)) for value in state)


def save_scientific_snapshot(
    path: Path, nodes: np.ndarray, tri: np.ndarray, state: tuple, metadata: dict
) -> bool:
    if path.is_file() and metadata.get("milestone") != "final":
        return False
    displacement, damage, history, epsp, epsp33, epspeq = host_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        nodes=nodes,
        triangles=tri,
        displacement=displacement.reshape(-1, 2),
        damage=damage,
        history=history,
        plastic_strain=epsp,
        plastic_strain_33=epsp33,
        equivalent_plastic_strain=epspeq,
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    temporary.replace(path)
    return True


def run_case(args, geom: GeometryCase) -> dict:
    case_dir = args.outdir / geom.tag
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh_file = args.mesh if args.mesh is not None else case_dir / "mesh_double_notched.msh"
    if args.mesh is None and not mesh_file.is_file():
        create_mesh(mesh_file, geom)
    if not mesh_file.is_file():
        raise FileNotFoundError(mesh_file)
    nodes, tri, areas, edges = load_mesh(mesh_file)
    bmat, dofs, ke0, mass, grad = preprocess(nodes, tri, areas)
    nnode, nel = len(nodes), len(tri)
    ndof = 2 * nnode

    boundary_nodes = {
        tag: np.unique(edge.reshape(-1)) for tag, edge in edges.items()
    }
    fixed = np.unique(
        np.concatenate(
            [2 * values for values in boundary_nodes.values()]
            + [2 * values + 1 for values in boundary_nodes.values()]
        )
    )
    free = np.ones(ndof)
    free[fixed] = 0.0
    solve_u, return_map, solve_d, reaction = make_kernels(
        nodes_np=nodes,
        tri_np=tri,
        areas_np=areas,
        edges_np=edges,
        bmat_np=bmat,
        dofs_np=dofs,
        ke0_np=ke0,
        mass_np=mass,
        grad_np=grad,
        free_np=free,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
    )

    checkpoint = case_dir / "checkpoint_latest.npz"
    csv_path = case_dir / "force_displacement.csv"
    if args.resume and checkpoint.is_file():
        start_step, state_np, rows = load_checkpoint(checkpoint)
    else:
        start_step = 0
        state_np = (
            np.zeros(ndof),
            np.zeros(nnode),
            np.zeros(nel),
            np.zeros((nel, 3)),
            np.zeros(nel),
            np.zeros(nel),
        )
        rows = []
    state = tuple(jnp.asarray(value, dtype=jnp.float64) for value in state_np)
    displacement, damage, history, epsp, epsp33, epspeq = state
    u_total = geom.u_total if args.u_total is None else args.u_total
    du = u_total / args.steps
    elapsed_offset = float(rows[-1]["time_s"]) if rows else 0.0
    run_t0 = time.perf_counter()
    peak_force = max((float(row["F_kN"]) for row in rows), default=-np.inf)
    stop_trigger = None
    snapshots_enabled = not args.no_snapshots and geom.tag == "paper"
    snapshot_dir = case_dir / "snapshots"
    snapshot_done = [
        (snapshot_dir / f"milestone_d{round(100 * level):03d}.npz").is_file()
        for level in SNAPSHOT_DAMAGE_LEVELS
    ]
    snapshot_io_s = 0.0

    print("=" * 78)
    print(f"Double-notched ductile fracture | {geom.title}")
    print(f"  JAX {jax.__version__}; devices={jax.devices()}; dtype=float64")
    print(f"  E={E_MODULUS:.1f} MPa, nu={POISSON:.4f}, G={G:.1f}, K={K_BULK:.1f}")
    print(f"  sigma_y={SIGMA_Y0:.1f}, H={H_HARD:.1f}, Gc={GC}, l={LC}")
    print(f"  mesh={nnode} nodes/{nel} triangles; steps={args.steps}; du={du:.3e}")
    print("=" * 78)

    for step in range(start_step + 1, args.steps + 1):
        step_t0 = time.perf_counter()
        u_value = step * du
        u_previous = (step - 1) * du
        delta = u_value - u_previous
        nsub = max(1, int(np.ceil(abs(delta) / 0.001)))
        u_linear_residual = np.inf
        plastic_iterations = 0
        for sub in range(1, nsub + 1):
            u_current = u_previous + delta * sub / nsub
            boundary = np.zeros(ndof)
            # Gridap tag order: bottom, right, top, left. The final assignment
            # governs a shared corner (only relevant to the load_bias case).
            boundary[2 * boundary_nodes[3] + 1] = u_current
            boundary[2 * boundary_nodes[4] + 1] = geom.left_scale * u_current
            boundary_j = jnp.asarray(boundary, dtype=jnp.float64)
            base_epsp, base_epsp33, base_epspeq = epsp, epsp33, epspeq
            for plastic_iteration in range(1, args.plastic_maxiter + 1):
                old_displacement = displacement
                displacement, u_res = solve_u(
                    damage, epsp, boundary_j, displacement
                )
                _, _, epsp, epsp33, epspeq = return_map(
                    displacement,
                    damage,
                    base_epsp,
                    base_epsp33,
                    base_epspeq,
                    True,
                )
                change = jnp.linalg.norm(displacement - old_displacement) / (
                    jnp.linalg.norm(displacement) + 1.0e-10
                )
                jax.block_until_ready(change)
                u_linear_residual = float(jax.device_get(u_res))
                plastic_iterations = plastic_iteration
                if float(jax.device_get(change)) < args.plastic_tol:
                    break

        stress, sse, _, _, _ = return_map(
            displacement, damage, epsp, epsp33, epspeq, False
        )
        plastic_energy = SIGMA_Y0 * epspeq + 0.5 * H_HARD * epspeq**2
        history = jnp.maximum(history, sse + plastic_energy)

        d_linear_residual = np.inf
        stagger_residual = np.inf
        for stagger_iteration in range(1, args.max_stagger + 1):
            old_damage = damage
            solved_damage, d_res = solve_d(history, damage)
            damage = jnp.clip(jnp.maximum(solved_damage, old_damage), 0.0, 1.0)
            stagger = jnp.linalg.norm(damage - old_damage) / (
                jnp.linalg.norm(damage) + 1.0e-10
            )
            jax.block_until_ready(stagger)
            d_linear_residual = float(jax.device_get(d_res))
            stagger_residual = float(jax.device_get(stagger))
            if stagger_residual < args.stagger_tol:
                break

        if max(u_linear_residual, d_linear_residual) > args.linear_residual_tol:
            raise RuntimeError(
                f"step {step}: linear residual too large "
                f"(u={u_linear_residual:.3e}, d={d_linear_residual:.3e})"
            )
        force = float(jax.device_get(reaction(stress)))
        dmax = float(jax.device_get(jnp.max(damage)))
        epmax = float(jax.device_get(jnp.max(epspeq)))
        peak_force = max(peak_force, force)
        elapsed = elapsed_offset + time.perf_counter() - run_t0 - snapshot_io_s
        row = {
            "step": step,
            "time_s": elapsed,
            "step_time_s": time.perf_counter() - step_t0,
            "u_mm": u_value,
            "F_kN": force,
            "d_max": dmax,
            "epspeq_max": epmax,
            "plastic_iterations": plastic_iterations,
            "stagger_iterations": stagger_iteration,
            "stagger_residual": stagger_residual,
            "u_linear_residual": u_linear_residual,
            "d_linear_residual": d_linear_residual,
        }
        rows.append(row)
        if step <= 5 or step % args.print_every == 0:
            print(
                f"  step {step:4d}/{args.steps} u={u_value:.5f} "
                f"F={force:.4f} kN dmax={dmax:.4f} epspeq={epmax:.5f}"
            )

        current_state = (displacement, damage, history, epsp, epsp33, epspeq)
        for index, level in enumerate(SNAPSHOT_DAMAGE_LEVELS):
            if snapshots_enabled and not snapshot_done[index] and dmax >= level:
                snapshot_t0 = time.perf_counter()
                save_scientific_snapshot(
                    snapshot_dir / f"milestone_d{round(100 * level):03d}.npz",
                    nodes,
                    tri,
                    current_state,
                    {
                        "cycle": 0,
                        "step": step,
                        "load": u_value,
                        "qoi_force_kN": force,
                        "qoi_d_max": dmax,
                        "milestone": f"d_max_{level:.2f}",
                    },
                )
                snapshot_io_s += time.perf_counter() - snapshot_t0
                if not args.no_vtk:
                    tag = f"d{round(100 * level):03d}_step{step:04d}"
                    save_vtu(
                        case_dir / f"result_{tag}.vtu",
                        nodes,
                        tri,
                        *host_state(current_state)[:3],
                        host_state(current_state)[5],
                    )
                snapshot_done[index] = True
        if args.vtu_every > 0 and step % args.vtu_every == 0 and not args.no_vtk:
            hs = host_state(current_state)
            save_vtu(case_dir / f"result_step{step:04d}.vtu", nodes, tri, hs[0], hs[1], hs[2], hs[5])
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            write_csv(csv_path, rows)
            save_checkpoint(checkpoint, step, host_state(current_state), rows)

        if stop_trigger is None:
            if peak_force > 1.0e-8 and dmax >= 0.99 and force <= STOP_FORCE_RATIO * peak_force:
                stop_trigger = step
        elif step >= stop_trigger + STOP_AFTER_STEPS:
            break

    final_state = (displacement, damage, history, epsp, epsp33, epspeq)
    final_host = host_state(final_state)
    write_csv(csv_path, rows)
    save_checkpoint(checkpoint, int(rows[-1]["step"]), final_host, rows)
    if snapshots_enabled:
        snapshot_t0 = time.perf_counter()
        final_row = rows[-1]
        save_scientific_snapshot(
            snapshot_dir / "final.npz",
            nodes,
            tri,
            final_state,
            {
                "cycle": 0,
                "step": int(final_row["step"]),
                "load": float(final_row["u_mm"]),
                "qoi_force_kN": float(final_row["F_kN"]),
                "qoi_d_max": float(final_row["d_max"]),
                "milestone": "final",
            },
        )
        snapshot_io_s += time.perf_counter() - snapshot_t0
    if not args.no_vtk:
        save_vtu(
            case_dir / "result_final.vtu",
            nodes,
            tri,
            final_host[0],
            final_host[1],
            final_host[2],
            final_host[5],
        )
    if not args.no_plot:
        save_plot(case_dir / "F_u_curve_double_notched.png", rows)
    timing = {
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "devices": [str(device) for device in jax.devices()],
        "device_strings": [str(device) for device in jax.devices()],
        "device_kinds": [device.device_kind for device in jax.devices()],
        "device_slug": device_slug(jax.devices()[0]),
        "dtype": "float64",
        "nodes": nnode,
        "triangles": nel,
        "runtime_s": time.perf_counter() - run_t0 - snapshot_io_s,
        "benchmark_mode": bool(args.benchmark),
        "mesh_file": str(mesh_file),
        "n_steps": len(rows),
        "first_step_s": rows[0]["step_time_s"] if rows else None,
        "post_first_mean_step_s": (
            float(np.mean([row["step_time_s"] for row in rows[1:]]))
            if len(rows) > 1
            else None
        ),
    }
    with (case_dir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    return {
        "case_tag": geom.tag,
        "H_hard_MPa": H_HARD,
        "bl_radius_mm": geom.bl_radius,
        "tr_radius_mm": geom.tr_radius,
        "u_total_mm": u_total,
        "left_scale": geom.left_scale,
        "runtime_s": timing["runtime_s"],
        "n_steps": len(rows),
        "f_peak_kN": max(row["F_kN"] for row in rows),
        "d_final": rows[-1]["d_max"],
        "epspeq_final": rows[-1]["epspeq_max"],
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.steps = 1
        if args.u_total is None:
            args.u_total = 0.001
        args.max_stagger = min(args.max_stagger, 2)
        args.plastic_maxiter = min(args.plastic_maxiter, 2)
        args.no_vtk = True
        args.no_plot = True
    if args.benchmark:
        args.no_vtk = True
        args.no_plot = True
    if args.steps < 1 or args.max_stagger < 1 or args.plastic_maxiter < 1:
        raise ValueError("iteration counts must be positive")
    if min(args.cg_tol, args.stagger_tol, args.plastic_tol) <= 0.0:
        raise ValueError("solver tolerances must be positive")
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if args.platform == "gpu" and all(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU requested, but available JAX devices are {devices}")
    if args.outdir is None:
        root = Path.cwd()
        args.outdir = (
            root / "outputs" / f"results_{device_slug(devices[0])}"
            / "plastic_fracture_double_notched" / "jax"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = CASES if args.case == "all" else tuple(c for c in CASES if c.tag == args.case)
    summaries = [run_case(args, geom) for geom in selected]
    summary_path = args.outdir / "scan_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"outputs: {args.outdir}")


if __name__ == "__main__":
    main()
