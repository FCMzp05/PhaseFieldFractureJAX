#!/usr/bin/env python3
"""
Standalone JAX/GPU finite-element solver for the SENS pure-shear fracture test.

This is a direct port of reference_julia_standalone/elastic_fracture_SENS.jl:
plane strain P1 triangles, hybrid spectral split, DG0 history, AT2 damage,
degree-two exact quadrature, staggered solves, and top-boundary shear traction.
The file imports no project-local module.  If no reference mesh is present it
generates the Julia Gmsh geometry and mesh itself.

GPU smoke test (WSL):
  python elastic_fracture_SENS_jax_gpu.py --platform gpu --steps 2 --no-plot --no-vtk

Dependencies: jax (CUDA build for GPU), numpy, meshio, gmsh; matplotlib optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path


def _preparse_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    args, _ = parser.parse_known_args()
    return args.platform


_PLATFORM_REQUEST = _preparse_platform()
if _PLATFORM_REQUEST != "auto":
    os.environ.setdefault(
        "JAX_PLATFORMS", "cuda" if _PLATFORM_REQUEST == "gpu" else _PLATFORM_REQUEST
    )
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "JAX is required. For NVIDIA CUDA under WSL, install for example:\n"
        "  python -m pip install 'jax[cuda13]' meshio gmsh\n"
        f"Original import error: {exc}"
    ) from exc


# Exact constants from elastic_fracture_SENS.jl.
W = 1.0
HEIGHT = 1.0
A0 = 0.5
Y_NOTCH = 0.5
CH_PRECRACK_FACTOR = 0.5
H_FINE = 0.005
H_COARSE = 0.05
REFINE_XMIN = 0.49
REFINE_XMAX = 1.0
REFINE_YMIN = 0.0
REFINE_YMAX = 1.0

LAMBDA = 121_150.0
MU = 80_770.0
GC = 2.7
L0 = 0.015
ETA = 0.0

DEFAULT_DU = 2.0e-5
DEFAULT_U_END = 20.0e-3
DEFAULT_STEPS = 1000
DEFAULT_STAGGER_TOL = 1.0e-4
DEFAULT_MIN_STAGGER = 2
DEFAULT_MAX_STAGGER = 20
F_STOP_TOL = 1.0e-3
VTK_ONSET_LEVEL = 0.999
VTK_FORCE_DROP_LEVELS = (0.75, 0.50, 0.25)
SNAPSHOT_THRESHOLDS = (0.20, 0.50, 0.80, 0.95, 0.99)


def script_root() -> Path:
    """Use the repository root in-tree, or the script directory standalone."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code" else script_dir


def build_parser() -> argparse.ArgumentParser:
    root = script_root()
    default_reference = (
        root
        / "data"
        / "julia_reference"
        / "elastic_fracture_sens"
        / "F_u_curve_split.csv"
    )
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU SENS pure-shear phase-field solver"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="existing exact mesh (default: generate in the output directory)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_reference,
        help="optional Julia F-u CSV (missing files are skipped)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: outputs/results_<device>/elastic_fracture_SENS/jax)",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--du", type=float, default=DEFAULT_DU)
    parser.add_argument("--u-end", type=float, default=DEFAULT_U_END)
    parser.add_argument("--stagger-tol", type=float, default=DEFAULT_STAGGER_TOL)
    parser.add_argument("--min-stagger", type=int, default=DEFAULT_MIN_STAGGER)
    parser.add_argument("--max-stagger", type=int, default=DEFAULT_MAX_STAGGER)
    parser.add_argument(
        "--reject-stagger-cap",
        action="store_true",
        help="reject unconverged steps instead of Julia's commit-at-20 behavior",
    )
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=3000)
    parser.add_argument("--linear-residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--force-stop-tol", type=float, default=F_STOP_TOL)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regenerate-mesh", action="store_true")
    parser.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="disable sparse compressed scientific state snapshots",
    )
    parser.set_defaults(snapshots=True)
    parser.add_argument("--vtk", action="store_true", help="write optional VTU fields (off by default)")
    parser.add_argument("--plot", action="store_true", help="write optional comparison plot (off by default)")
    parser.add_argument("--no-vtk", dest="vtk", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--no-plot", dest="plot", action="store_false", help=argparse.SUPPRESS)
    return parser


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def generate_mesh(mesh_file: Path) -> None:
    """Generate the exact geometry, tags, and Gmsh settings used by Julia."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit(
            f"mesh does not exist ({mesh_file}) and gmsh is unavailable; "
            "install it with: python -m pip install gmsh"
        ) from exc

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    ch = CH_PRECRACK_FACTOR * H_FINE
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("SENS_shear")
        geo = gmsh.model.geo
        geo.addPoint(0.0, 0.0, 0.0, H_COARSE, 1)
        geo.addPoint(W, 0.0, 0.0, H_COARSE, 2)
        geo.addPoint(W, Y_NOTCH, 0.0, H_COARSE, 10)
        geo.addPoint(W, HEIGHT, 0.0, H_COARSE, 3)
        geo.addPoint(0.0, HEIGHT, 0.0, H_COARSE, 4)
        geo.addPoint(0.0, Y_NOTCH + ch, 0.0, H_COARSE, 5)
        geo.addPoint(A0, Y_NOTCH + ch, 0.0, H_FINE, 6)
        geo.addPoint(A0, Y_NOTCH, 0.0, H_FINE, 7)
        geo.addPoint(A0, Y_NOTCH - ch, 0.0, H_FINE, 8)
        geo.addPoint(0.0, Y_NOTCH - ch, 0.0, H_COARSE, 9)
        geo.addLine(1, 2, 1)
        geo.addLine(2, 10, 2)
        geo.addLine(10, 3, 10)
        geo.addLine(3, 4, 3)
        geo.addLine(4, 5, 4)
        geo.addLine(5, 6, 5)
        geo.addLine(6, 7, 6)
        geo.addLine(7, 8, 7)
        geo.addLine(8, 9, 8)
        geo.addLine(9, 1, 9)
        geo.addCurveLoop([1, 2, 10, 3, 4, 5, 6, 7, 8, 9], 1)
        geo.addPlaneSurface([1], 1)
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        gmsh.model.addPhysicalGroup(1, [1], 1)
        gmsh.model.setPhysicalName(1, 1, "bottom")
        gmsh.model.addPhysicalGroup(1, [3], 2)
        gmsh.model.setPhysicalName(1, 2, "top")
        gmsh.model.addPhysicalGroup(1, [4, 9], 3)
        gmsh.model.setPhysicalName(1, 3, "left")
        field = gmsh.model.mesh.field
        field.add("Box", 1)
        field.setNumber(1, "VIn", H_FINE)
        field.setNumber(1, "VOut", H_COARSE)
        field.setNumber(1, "XMin", REFINE_XMIN)
        field.setNumber(1, "XMax", REFINE_XMAX)
        field.setNumber(1, "YMin", REFINE_YMIN)
        field.setNumber(1, "YMax", REFINE_YMAX)
        field.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.finalize()


def load_mesh(mesh_file: Path, regenerate: bool):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    if regenerate or not mesh_file.is_file():
        print(f"generating canonical SENS mesh: {mesh_file}")
        generate_mesh(mesh_file)
    mesh = meshio.read(str(mesh_file))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangles = [
        np.asarray(block.data, dtype=np.int64)
        for block in mesh.cells
        if block.type == "triangle"
    ]
    if not triangles:
        raise RuntimeError("mesh has no P1 triangles")
    tri = np.concatenate(triangles)

    bottom_nodes: set[int] = set()
    top_nodes: set[int] = set()
    top_edges: list[np.ndarray] = []
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is not None:
        for block_index, block in enumerate(mesh.cells):
            if block.type != "line":
                continue
            for edge, tag in zip(block.data, physical[block_index]):
                if int(tag) == 1:
                    bottom_nodes.update(int(i) for i in edge)
                elif int(tag) == 2:
                    top_nodes.update(int(i) for i in edge)
                    top_edges.append(np.asarray(edge, dtype=np.int64))
    tol = 1.0e-10
    if not bottom_nodes:
        bottom_nodes.update(np.flatnonzero(np.abs(nodes[:, 1]) <= tol).tolist())
    if not top_nodes:
        top_nodes.update(
            np.flatnonzero(np.abs(nodes[:, 1] - HEIGHT) <= tol).tolist()
        )
    if not top_edges:
        ordered = sorted(top_nodes, key=lambda i: nodes[i, 0])
        top_edges = [
            np.asarray([a, b], dtype=np.int64)
            for a, b in zip(ordered[:-1], ordered[1:])
        ]

    xy = nodes[tri]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0.0
    if np.any(flip):
        tri[flip] = tri[flip][:, [0, 2, 1]]
        xy = nodes[tri]
        twice_area = (
            (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
            - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
        )
    areas = 0.5 * twice_area
    if np.any(areas <= 0.0):
        raise RuntimeError("mesh contains zero/negative-area triangles")
    bottom = np.asarray(sorted(bottom_nodes), dtype=np.int64)
    top = np.asarray(sorted(top_nodes), dtype=np.int64)
    if bottom.size == 0 or top.size == 0:
        raise RuntimeError("could not identify top/bottom boundary nodes")
    return nodes, tri, areas, bottom, top, np.asarray(top_edges, dtype=np.int64)


def find_boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edge_to_cell: dict[tuple[int, int], int] = {}
    for cell, element in enumerate(tri):
        for a, b in ((element[0], element[1]), (element[1], element[2]), (element[2], element[0])):
            edge_to_cell[tuple(sorted((int(a), int(b))))] = cell
    cells = []
    for edge in edges:
        key = tuple(sorted((int(edge[0]), int(edge[1]))))
        if key not in edge_to_cell:
            raise RuntimeError(f"top edge {key} has no adjacent triangle")
        cells.append(edge_to_cell[key])
    return np.asarray(cells, dtype=np.int64)


def preprocess_elements(nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray):
    x, y = nodes[:, 0], nodes[:, 1]
    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
    gx = np.stack([y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]], axis=1)
    gy = np.stack([x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]], axis=1)
    gx /= 2.0 * areas[:, None]
    gy /= 2.0 * areas[:, None]
    nel = tri.shape[0]
    bmat = np.zeros((nel, 3, 6), dtype=np.float64)
    for a in range(3):
        bmat[:, 0, 2 * a] = gx[:, a]
        bmat[:, 1, 2 * a + 1] = gy[:, a]
        bmat[:, 2, 2 * a] = gy[:, a]
        bmat[:, 2, 2 * a + 1] = gx[:, a]
    constitutive = np.asarray(
        [
            [LAMBDA + 2.0 * MU, LAMBDA, 0.0],
            [LAMBDA, LAMBDA + 2.0 * MU, 0.0],
            [0.0, 0.0, MU],
        ],
        dtype=np.float64,
    )
    ke0 = areas[:, None, None] * np.einsum(
        "eia,ij,ejb->eab", bmat, constitutive, bmat, optimize=True
    )
    mass = areas[:, None, None] * (
        np.asarray([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.float64) / 12.0
    )
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    dofs = np.empty((nel, 6), dtype=np.int64)
    for a in range(3):
        dofs[:, 2 * a] = 2 * tri[:, a]
        dofs[:, 2 * a + 1] = 2 * tri[:, a] + 1
    return bmat, ke0, mass, grad, dofs


def make_jax_solvers(
    *,
    tri_np,
    dofs_np,
    bmat_np,
    ke0_np,
    mass_np,
    grad_np,
    areas_np,
    nodes_np,
    top_edges_np,
    top_cells_np,
    free_mask_np,
    dtype,
    eta,
    cg_tol,
    cg_maxiter,
):
    tri = jnp.asarray(tri_np)
    dofs = jnp.asarray(dofs_np)
    bmat = jnp.asarray(bmat_np, dtype=dtype)
    ke0 = jnp.asarray(ke0_np, dtype=dtype)
    mass = jnp.asarray(mass_np, dtype=dtype)
    grad = jnp.asarray(grad_np, dtype=dtype)
    areas = jnp.asarray(areas_np, dtype=dtype)
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    top_edges = jnp.asarray(top_edges_np)
    top_cells = jnp.asarray(top_cells_np)
    edge_lengths = jnp.linalg.norm(
        nodes[top_edges[:, 1]] - nodes[top_edges[:, 0]], axis=1
    )
    free = jnp.asarray(free_mask_np, dtype=dtype)
    ndof = free_mask_np.size
    nnode = ndof // 2
    eta_j = jnp.asarray(eta, dtype=dtype)

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(values.reshape(-1))

    def weights(damage):
        # Exact degree-2 triangle average of (1-d)^2+eta for linear P1 d.
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        d_mean = jnp.mean(dv, axis=1)
        d2_mean = (
            jnp.sum(dv * dv, axis=1)
            + dv[:, 0] * dv[:, 1]
            + dv[:, 1] * dv[:, 2]
            + dv[:, 2] * dv[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * d_mean + d2_mean + eta_j

    def u_matvec(x, w):
        values = w[:, None] * jnp.einsum("eij,ej->ei", ke0, x[dofs])
        return scatter(dofs, values, ndof)

    @jax.jit
    def solve_displacement(damage, boundary, x0):
        w = weights(damage)

        def kmv(x):
            return u_matvec(x, w)

        rhs = -free * kmv(boundary)

        def operator(x):
            return free * kmv(free * x) + (1.0 - free) * x

        diagonal = free * scatter(
            dofs, w[:, None] * jnp.diagonal(ke0, axis1=1, axis2=2), ndof
        ) + (1.0 - free)
        diagonal = jnp.maximum(diagonal, jnp.finfo(dtype).tiny)
        correction, _ = cg(
            operator,
            rhs,
            x0=free * x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / diagonal,
        )
        displacement = boundary + free * correction
        residual = operator(correction) - rhs
        relative = jnp.linalg.norm(residual) / (
            jnp.linalg.norm(rhs) + jnp.asarray(1.0e-30, dtype=dtype)
        )
        return displacement, relative

    def d_matvec(x, element_matrix):
        return scatter(
            tri, jnp.einsum("eij,ej->ei", element_matrix, x[tri]), nnode
        )

    @jax.jit
    def solve_phase(history, x0):
        coefficient = GC / L0 + 2.0 * history
        element_matrix = GC * L0 * grad + coefficient[:, None, None] * mass
        rhs = scatter(
            tri,
            (2.0 * history * areas / 3.0)[:, None]
            * jnp.ones((1, 3), dtype=dtype),
            nnode,
        )

        def operator(x):
            return d_matvec(x, element_matrix)

        diagonal = scatter(
            tri, jnp.diagonal(element_matrix, axis1=1, axis2=2), nnode
        )
        diagonal = jnp.maximum(diagonal, jnp.finfo(dtype).tiny)
        damage, _ = cg(
            operator,
            rhs,
            x0=x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / diagonal,
        )
        residual = operator(damage) - rhs
        relative = jnp.linalg.norm(residual) / (
            jnp.linalg.norm(rhs) + jnp.asarray(1.0e-30, dtype=dtype)
        )
        return damage, relative

    @jax.jit
    def update_history(displacement, old_history):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        e11, e22, e12 = strain[:, 0], strain[:, 1], 0.5 * strain[:, 2]
        average = 0.5 * (e11 + e22)
        radius = jnp.sqrt((0.5 * (e11 - e22)) ** 2 + e12**2)
        p1, p2 = average + radius, average - radius
        psi_plus = (
            0.5 * LAMBDA * jnp.maximum(e11 + e22, 0.0) ** 2
            + MU * (jnp.maximum(p1, 0.0) ** 2 + jnp.maximum(p2, 0.0) ** 2)
        )
        return jnp.maximum(old_history, psi_plus)

    @jax.jit
    def boundary_reaction(displacement, damage):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        sigma_12 = MU * strain[:, 2]
        edge_damage = jnp.clip(damage[top_edges], 0.0, 1.0)
        d0, d1 = edge_damage[:, 0], edge_damage[:, 1]
        # Exact degree-2 line average of g(d).
        g_edge = (
            1.0
            - d0
            - d1
            + (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
            + eta_j
        )
        return jnp.sum(g_edge * sigma_12[top_cells] * edge_lengths)

    return solve_displacement, solve_phase, update_history, boundary_reaction


CSV_FIELDS = (
    "step",
    "time_s",
    "step_time_s",
    "u_mm",
    "F_N",
    "F_kN",
    "d_max",
    "stagger_iterations",
    "stagger_converged",
    "stagger_error",
    "u_linear_residual",
    "d_linear_residual",
)


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, step, displacement, damage, history, rows) -> None:
    np.savez_compressed(
        path,
        step=np.asarray(step),
        displacement=displacement,
        damage=damage,
        history=history,
        rows_json=np.asarray(json.dumps(rows)),
    )


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    return (
        int(data["step"]),
        np.asarray(data["displacement"]),
        np.asarray(data["damage"]),
        np.asarray(data["history"]),
        json.loads(str(data["rows_json"])),
    )


def snapshot_token(threshold: float) -> str:
    return f"{round(100 * threshold):03d}"


def save_snapshot(
    path: Path,
    *,
    nodes: np.ndarray,
    triangles: np.ndarray,
    displacement: np.ndarray,
    damage: np.ndarray,
    history: np.ndarray,
    step: int,
    applied_displacement: float,
    reaction_force: float,
    d_max: float,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.asarray("phase_field_state_snapshot_v1"),
        benchmark=np.asarray("SENS_spectral_eta0_split"),
        nodes=nodes,
        triangles=triangles,
        displacement=displacement,
        damage=damage,
        history=history,
        step=np.asarray(step),
        applied_displacement=np.asarray(applied_displacement),
        reaction_force=np.asarray(reaction_force),
        d_max=np.asarray(d_max),
        threshold=np.asarray(threshold),
    )


def save_vtu(path, nodes, tri, displacement, damage, history) -> None:
    import meshio

    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])
    vectors = np.column_stack(
        [displacement[0::2], displacement[1::2], np.zeros(nodes.shape[0])]
    )
    meshio.write_points_cells(
        str(path),
        points,
        [("triangle", tri)],
        point_data={"d": damage, "uh": vectors},
        cell_data={"H": [history]},
    )


def load_reference(path: Path | None):
    if path is None or not path.is_file():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = set(data.dtype.names or ())
    required = {"u_mm", "F_N"}
    if not required.issubset(names):
        raise RuntimeError(f"reference CSV lacks columns {sorted(required)}: {path}")
    return {
        "u": np.atleast_1d(np.asarray(data["u_mm"], dtype=np.float64)),
        "force": np.atleast_1d(np.asarray(data["F_N"], dtype=np.float64)),
    }


def compare_reference(rows, reference):
    if reference is None or not rows:
        return None
    u = np.asarray([row["u_mm"] for row in rows])
    force = np.asarray([row["F_N"] for row in rows])
    ref_force = np.interp(u, reference["u"], reference["force"])
    relative_l2 = np.linalg.norm(force - ref_force) / (
        np.linalg.norm(ref_force) + 1.0e-30
    )
    covered = reference["u"] <= np.max(u) + 1.0e-14
    reference_peak = float(np.max(reference["force"][covered]))
    peak = float(np.max(force))
    return {
        "relative_l2_force": float(relative_l2),
        "peak_force_N": peak,
        "reference_peak_force_N": reference_peak,
        "peak_force_relative_error": (peak - reference_peak) / reference_peak,
    }


def save_plot(outdir: Path, rows, reference) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot skipped: matplotlib is not installed")
        return
    u = np.asarray([row["u_mm"] for row in rows])
    force = np.asarray([row["F_N"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(1.0e3 * u, 1.0e-3 * force, label="JAX", lw=2.0)
    if reference is not None:
        ax.plot(
            1.0e3 * reference["u"],
            1.0e-3 * reference["force"],
            "--",
            label="Julia/Gridap",
            lw=1.6,
        )
    ax.set_xlabel(r"Horizontal displacement $u$ ($10^{-3}$ mm)")
    ax.set_ylabel("Top shear reaction (kN)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "F_u_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.du <= 0.0 or args.u_end <= 0.0:
        raise ValueError("steps, du, and u-end must be positive")
    if args.min_stagger < 1 or args.max_stagger < args.min_stagger:
        raise ValueError("invalid stagger iteration limits")
    if args.cg_tol <= 0.0 or args.linear_residual_tol <= 0.0:
        raise ValueError("linear solver tolerances must be positive")
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("FP64 could not be enabled")
    dtype = jnp.float64
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    if args.platform == "gpu" and all(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU requested, but JAX devices are {devices}")
    selected_device_slug = device_slug(devices[0])
    if args.outdir is None:
        root = script_root()
        args.outdir = (
            root / "outputs" / f"results_{selected_device_slug}"
            / "elastic_fracture_SENS" / "jax"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh or args.outdir / "mesh_SENS.msh"

    nodes, tri, areas, bottom, top, top_edges = load_mesh(
        mesh_path, args.regenerate_mesh
    )
    top_cells = find_boundary_cells(tri, top_edges)
    bmat, ke0, mass, grad, dofs = preprocess_elements(nodes, tri, areas)
    nnode = nodes.shape[0]
    ndof = 2 * nnode
    # Exact executable Julia model: both components fixed on bottom/top only.
    # Top ux is prescribed; top uy and both bottom components are zero.
    fixed_dofs = np.unique(
        np.concatenate([2 * bottom, 2 * bottom + 1, 2 * top, 2 * top + 1])
    )
    free_mask = np.ones(ndof, dtype=np.float64)
    free_mask[fixed_dofs] = 0.0
    top_x_dofs = 2 * top

    solve_u, solve_d, update_history, boundary_reaction = make_jax_solvers(
        tri_np=tri,
        dofs_np=dofs,
        bmat_np=bmat,
        ke0_np=ke0,
        mass_np=mass,
        grad_np=grad,
        areas_np=areas,
        nodes_np=nodes,
        top_edges_np=top_edges,
        top_cells_np=top_cells,
        free_mask_np=free_mask,
        dtype=dtype,
        eta=ETA,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
    )

    checkpoint_file = args.outdir / "checkpoint_latest.npz"
    csv_file = args.outdir / "F_u_curve_jax.csv"
    snapshot_dir = args.outdir / "snapshots"
    if args.resume and checkpoint_file.is_file():
        start_step, u_np, d_np, h_np, rows = load_checkpoint(checkpoint_file)
        print(f"resuming after step {start_step}: {checkpoint_file}")
    else:
        start_step = 0
        u_np = np.zeros(ndof, dtype=np.float64)
        d_np = np.zeros(nnode, dtype=np.float64)
        h_np = np.zeros(tri.shape[0], dtype=np.float64)
        rows: list[dict[str, float]] = []
    displacement = jnp.asarray(u_np, dtype=dtype)
    damage = jnp.asarray(d_np, dtype=dtype)
    history = jnp.asarray(h_np, dtype=dtype)
    completed_thresholds: set[float] = set()
    if args.snapshots and args.resume:
        resumed_d_max = float(np.max(d_np)) if d_np.size else 0.0
        completed_thresholds.update(
            threshold
            for threshold in SNAPSHOT_THRESHOLDS
            if resumed_d_max >= threshold
        )
        completed_thresholds.update(
            threshold
            for threshold in SNAPSHOT_THRESHOLDS
            if (snapshot_dir / f"snapshot_dmax_{snapshot_token(threshold)}.npz").is_file()
        )
    reference = load_reference(args.reference)

    print("=" * 78)
    print("Standalone JAX SENS pure-shear brittle phase-field fracture")
    print(f"  JAX {jax.__version__}; Python {platform.python_version()}; {devices}")
    print(
        f"  dtype=float64, eta={ETA:g}, mesh={nnode} nodes/"
        f"{tri.shape[0]} triangles, free displacement DOFs={int(free_mask.sum())}"
    )
    print(
        f"  du={args.du:.3e} mm, u_end={args.u_end:.3e} mm, "
        f"stagger=[{args.min_stagger},{args.max_stagger}], "
        f"tol={args.stagger_tol:g}, cap={'reject' if args.reject_stagger_cap else 'commit'}"
    )
    print("=" * 78)

    solve_t0 = time.perf_counter()
    elapsed_offset = float(rows[-1]["time_s"]) if rows else 0.0
    compile_time = None
    completed = 0
    crack_started = any(row["d_max"] >= VTK_ONSET_LEVEL for row in rows)
    force_onset = 0.0
    next_drop = 0

    for step in range(start_step + 1, args.steps + 1):
        step_t0 = time.perf_counter()
        prescribed_u = min(step * args.du, args.u_end)
        boundary_np = np.zeros(ndof, dtype=np.float64)
        boundary_np[top_x_dofs] = prescribed_u
        boundary = jnp.asarray(boundary_np, dtype=dtype)

        damage_step_old = damage
        history_step_old = history
        u_iter_old = None
        stagger_error = math.inf
        u_residual = math.inf
        d_residual = math.inf
        history_trial = history

        for stagger_iteration in range(1, args.max_stagger + 1):
            displacement_new, u_res = solve_u(damage, boundary, displacement)
            # Julia compares every current-step trial to H_n, not to a history
            # accumulated during the same stagger loop.
            history_trial = update_history(displacement_new, history_step_old)
            damage_trial, d_res = solve_d(history_trial, damage)
            damage_new = jnp.clip(
                jnp.maximum(damage_trial, damage_step_old), 0.0, 1.0
            )
            jax.block_until_ready(damage_new)
            u_residual = float(jax.device_get(u_res))
            d_residual = float(jax.device_get(d_res))
            if u_iter_old is None:
                u_error = math.inf
            else:
                u_error = float(
                    jax.device_get(
                        jnp.linalg.norm(
                            jnp.asarray(free_mask, dtype=dtype)
                            * (displacement_new - u_iter_old)
                        )
                        / (
                            jnp.linalg.norm(
                                jnp.asarray(free_mask, dtype=dtype)
                                * displacement_new
                            )
                            + jnp.asarray(1.0e-30, dtype=dtype)
                        )
                    )
                )
            d_error = float(
                jax.device_get(
                    jnp.linalg.norm(damage_new - damage)
                    / (
                        jnp.linalg.norm(damage_new)
                        + jnp.asarray(1.0e-30, dtype=dtype)
                    )
                )
            )
            stagger_error = max(u_error, d_error)
            u_iter_old = displacement_new
            displacement = displacement_new
            damage = damage_new
            if (
                stagger_iteration >= args.min_stagger
                and stagger_error < args.stagger_tol
            ):
                break
        history = history_trial
        stagger_converged = bool(
            np.isfinite(stagger_error) and stagger_error < args.stagger_tol
        )
        if not stagger_converged and args.reject_stagger_cap:
            raise RuntimeError(
                f"step {step}: stagger failed after {args.max_stagger} iterations "
                f"(error={stagger_error:.3e})"
            )
        if (
            not np.isfinite(u_residual)
            or not np.isfinite(d_residual)
            or max(u_residual, d_residual) > args.linear_residual_tol
        ):
            raise RuntimeError(
                f"step {step}: linear residual too large: "
                f"u={u_residual:.3e}, d={d_residual:.3e}"
            )

        reaction_value = boundary_reaction(displacement, damage)
        jax.block_until_ready(reaction_value)
        reaction = float(jax.device_get(reaction_value))
        damage_max = float(jax.device_get(jnp.max(damage)))
        step_time = time.perf_counter() - step_t0
        elapsed = elapsed_offset + time.perf_counter() - solve_t0
        if compile_time is None:
            compile_time = step_time
        row = {
            "step": step,
            "time_s": elapsed,
            "step_time_s": step_time,
            "u_mm": prescribed_u,
            "F_N": reaction,
            "F_kN": 1.0e-3 * reaction,
            "d_max": damage_max,
            "stagger_iterations": stagger_iteration,
            "stagger_converged": int(stagger_converged),
            "stagger_error": stagger_error,
            "u_linear_residual": u_residual,
            "d_linear_residual": d_residual,
        }
        rows.append(row)
        completed += 1

        crossed_thresholds = [
            threshold
            for threshold in SNAPSHOT_THRESHOLDS
            if threshold not in completed_thresholds and damage_max >= threshold
        ]
        if args.snapshots and crossed_thresholds:
            u_snapshot = np.asarray(jax.device_get(displacement))
            d_snapshot = np.asarray(jax.device_get(damage))
            h_snapshot = np.asarray(jax.device_get(history))
            for threshold in crossed_thresholds:
                save_snapshot(
                    snapshot_dir / f"snapshot_dmax_{snapshot_token(threshold)}.npz",
                    nodes=nodes,
                    triangles=tri,
                    displacement=u_snapshot,
                    damage=d_snapshot,
                    history=h_snapshot,
                    step=step,
                    applied_displacement=prescribed_u,
                    reaction_force=reaction,
                    d_max=damage_max,
                    threshold=threshold,
                )
                completed_thresholds.add(threshold)

        if (
            step <= 5
            or step % args.print_every == 0
            or not stagger_converged
            or step == args.steps
        ):
            print(
                f"  step={step:4d} u={prescribed_u:.3e} F={reaction:10.4f} N "
                f"dmax={damage_max:.5f} stag={stagger_iteration:2d} "
                f"{'OK' if stagger_converged else 'CAP'} err={stagger_error:.2e} "
                f"lin=({u_residual:.1e},{d_residual:.1e}) dt={step_time:.3f}s"
            )

        u_host = d_host = h_host = None
        if args.vtk and not crack_started and damage_max >= VTK_ONSET_LEVEL:
            crack_started = True
            force_onset = max(reaction, args.force_stop_tol)
            u_host = np.asarray(jax.device_get(displacement))
            d_host = np.asarray(jax.device_get(damage))
            h_host = np.asarray(jax.device_get(history))
            save_vtu(
                args.outdir / f"result_onset_split_step{step:04d}.vtu",
                nodes,
                tri,
                u_host,
                d_host,
                h_host,
            )
        while (
            args.vtk
            and crack_started
            and next_drop < len(VTK_FORCE_DROP_LEVELS)
            and reaction <= VTK_FORCE_DROP_LEVELS[next_drop] * force_onset
        ):
            ratio = VTK_FORCE_DROP_LEVELS[next_drop]
            if u_host is None:
                u_host = np.asarray(jax.device_get(displacement))
                d_host = np.asarray(jax.device_get(damage))
                h_host = np.asarray(jax.device_get(history))
            save_vtu(
                args.outdir / f"result_drop{round(100 * ratio):02d}_split_step{step:04d}.vtu",
                nodes,
                tri,
                u_host,
                d_host,
                h_host,
            )
            next_drop += 1

        if args.save_every > 0 and (
            step % args.save_every == 0 or step == args.steps
        ):
            if u_host is None:
                u_host = np.asarray(jax.device_get(displacement))
                d_host = np.asarray(jax.device_get(damage))
                h_host = np.asarray(jax.device_get(history))
            write_history(csv_file, rows)
            save_checkpoint(
                checkpoint_file, step, u_host, d_host, h_host, rows
            )
        if step > 1 and reaction <= args.force_stop_tol:
            print(f"stopping: top shear reaction {reaction:.6g} N")
            break
        if prescribed_u >= args.u_end:
            break

    if not rows:
        raise RuntimeError("no load steps were completed")
    total_time = time.perf_counter() - solve_t0
    u_host = np.asarray(jax.device_get(displacement))
    d_host = np.asarray(jax.device_get(damage))
    h_host = np.asarray(jax.device_get(history))
    write_history(csv_file, rows)
    save_checkpoint(
        checkpoint_file, int(rows[-1]["step"]), u_host, d_host, h_host, rows
    )
    if args.snapshots:
        final_row = rows[-1]
        save_snapshot(
            snapshot_dir / "snapshot_final.npz",
            nodes=nodes,
            triangles=tri,
            displacement=u_host,
            damage=d_host,
            history=h_host,
            step=int(final_row["step"]),
            applied_displacement=float(final_row["u_mm"]),
            reaction_force=float(final_row["F_N"]),
            d_max=float(final_row["d_max"]),
            threshold=np.nan,
        )
    if args.vtk:
        save_vtu(
            args.outdir / "result_final.vtu",
            nodes,
            tri,
            u_host,
            d_host,
            h_host,
        )

    comparison = compare_reference(rows, reference)
    run_step_times = np.asarray(
        [row["step_time_s"] for row in rows[-completed:]], dtype=np.float64
    )
    timing = {
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "device_strings": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "device_slug": selected_device_slug,
        "dtype": "float64",
        "eta": ETA,
        "mesh": str(mesh_path),
        "nodes": nnode,
        "triangles": int(tri.shape[0]),
        "free_displacement_dofs": int(free_mask.sum()),
        "steps_this_run": completed,
        "total_s": total_time,
        "first_step_compile_and_solve_s": compile_time,
        "post_first_step_mean_s": (
            float(np.mean(run_step_times[1:])) if run_step_times.size > 1 else None
        ),
        "median_step_s": float(np.median(run_step_times)),
        "comparison": comparison,
    }
    with (args.outdir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    if args.plot:
        save_plot(args.outdir, rows, reference)

    print("-" * 78)
    print(f"completed {completed} steps in {total_time:.3f} s")
    print(f"first step (compile + solve): {compile_time:.3f} s")
    if run_step_times.size > 1:
        print(f"post-first mean step: {np.mean(run_step_times[1:]):.3f} s")
    if comparison is not None:
        print(
            "Julia comparison: "
            f"L2(F)={comparison['relative_l2_force']:.3%}, "
            f"peak error={comparison['peak_force_relative_error']:.3%}"
        )
    else:
        if args.reference is None:
            print("Julia comparison skipped: no reference CSV supplied")
        else:
            print(f"Julia comparison skipped: reference CSV not found at {args.reference}")
    print(f"outputs: {args.outdir}")


if __name__ == "__main__":
    main()
