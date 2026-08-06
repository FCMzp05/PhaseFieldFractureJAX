#!/usr/bin/env python3
"""Standalone matrix-free JAX FEM port of the Liu (2025) Fig. 7 SENT case.

The fixed canonical case is a 1 x 1 mm plane-strain SENT specimen with a
0.5 mm notch, AT2 damage, Amor volumetric/deviatoric split, configurable
crack-face hydrogen concentration, stress-assisted steady diffusion, and
hydrogen-dependent fracture toughness. P1 triangles are used for u, d and
cH. No project-local module or data file is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from pathlib import Path


def _requested_platform() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    return p.parse_known_args()[0].platform


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
    raise SystemExit(f"JAX is required (tested with JAX 0.6.2): {exc}") from exc


# Canonical Liu 2025 Fig. 7 values; units are mm, N, MPa, s and mol.
W, HEIGHT, YC, A0 = 1.0, 1.0, 0.0, 0.5
E, NU = 210000.0, 0.3
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
K_BULK = LAMBDA + 2.0 * MU / 3.0
L0, GC0, KAPPA = 0.010, 2.7, 1.0e-5
H_FINE, H_COARSE, NOTCH_OPENING = 0.0025, 0.01, 0.020
FINE_XMIN, FINE_XMAX, FINE_YHALF = A0 - 4.0 * L0, W, 1.25 * L0
U_MAX, NOMINAL_STEPS = 0.0065, 120
MAX_STAGGER, STAGGER_TOL = 200, 1.0e-6
STOP_CRACK_LENGTH = 0.45
CH_ENV_1WPPM, CH_ENRICHMENT_CAP = 5.5e-6, 4.0
D_H, V_H, R_GAS, TEMPERATURE = 0.0127, 2000.0, 8.314, 300.0
# sigma is in MPa and V_H in mm^3/mol, so V_H*sigma is N*mm/mol.
# R_GAS is in J/(mol*K); 1 J = 1000 N*mm.
JOULE_TO_N_MM = 1000.0
DELTA_GB, HYDROGEN_CHI = 30000.0, 0.89
SNAPSHOT_THRESHOLDS = (0.20, 0.50, 0.80, 0.95, 0.99)
VERSION = "canonical_hydrogen_fracture_SENT_jax_v3"


def script_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "code" else here


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--outdir", type=Path)
    p.add_argument("--mesh", type=Path, help="optional existing Gmsh mesh")
    p.add_argument("--reference", type=Path, help="optional reference data.csv")
    p.add_argument("--reference-dir", type=Path, help="optional directory containing data.csv")
    p.add_argument(
        "--hydrogen-wtppm",
        type=float,
        default=1.0,
        help="environmental hydrogen concentration in wt-ppm (default: 1)",
    )
    p.add_argument(
        "--ch-env-1wppm",
        type=float,
        default=CH_ENV_1WPPM,
        help="model lattice concentration corresponding to 1 wt-ppm, mol/mm^3",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-snapshots", dest="snapshots", action="store_false")
    p.add_argument("--reject-stagger-cap", action="store_true")
    p.add_argument("--no-adaptive-steps", action="store_true")
    p.add_argument("--cg-tol", type=float, default=1.0e-10)
    p.add_argument("--cg-maxiter", type=int, default=5000)
    p.add_argument("--linear-residual-tol", type=float, default=2.0e-6)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=NOMINAL_STEPS,
                   help="stop after this many accepted canonical steps")
    p.add_argument("--inner-progress-seconds", type=float, default=15.0,
                   help="report stagger progress at this wall-time interval (0 disables)")
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="one-step coarse-mesh integration smoke test (noncanonical resolution)",
    )
    p.add_argument("--mesh-only", action="store_true", help="generate/load and validate the mesh")
    p.set_defaults(snapshots=True)
    return p


def concentration_tag(value: float) -> str:
    """Return a stable filesystem tag such as 0p5wppm or 2wppm."""
    text = f"{value:.8g}".replace("-", "m").replace(".", "p")
    return f"{text}wppm"


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def generate_mesh(path: Path, smoke: bool = False) -> None:
    """Generate the exact canonical Gmsh geometry/physical groups."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh is required: python -m pip install gmsh") from exc
    hf, hc = (0.04, 0.08) if smoke else (H_FINE, H_COARSE)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("liu2025_hydrogen_SENT")
        g = gmsh.model.geo
        yb, yt, co = -0.5, 0.5, NOTCH_OPENING
        for tag, x, y, h in (
            (1, 0.0, yb, hc), (2, W, yb, hc), (3, W, YC, hf),
            (4, W, yt, hc), (5, 0.0, yt, hc), (6, 0.0, 0.5 * co, hc),
            (7, A0, 0.5 * co, hf), (8, A0, YC, hf),
            (9, A0, -0.5 * co, hf), (10, 0.0, -0.5 * co, hc),
        ):
            g.addPoint(x, y, 0.0, h, tag)
        for tag, a, b in (
            (1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 5, 6),
            (6, 6, 7), (7, 7, 8), (8, 8, 9), (9, 9, 10), (10, 10, 1),
            (11, 8, 3),
        ):
            g.addLine(a, b, tag)
        g.addCurveLoop(list(range(1, 11)), 1)
        g.addPlaneSurface([1], 1)
        g.synchronize()
        gmsh.model.mesh.embed(1, [11], 2, 1)
        for dim, entities, tag, name in (
            (2, [1], 1, "domain"), (1, [1], 1, "bottom"),
            (1, [4], 2, "top"), (1, [6, 7, 8, 9], 3, "crack"),
            (1, [2, 3], 4, "right"),
        ):
            gmsh.model.addPhysicalGroup(dim, entities, tag)
            gmsh.model.setPhysicalName(dim, tag, name)
        f = gmsh.model.mesh.field
        f.add("Box", 1)
        for key, value in (
            ("VIn", hf), ("VOut", hc), ("XMin", FINE_XMIN),
            ("XMax", FINE_XMAX), ("YMin", -FINE_YHALF),
            ("YMax", FINE_YHALF), ("ZMin", -1.0), ("ZMax", 1.0),
        ):
            f.setNumber(1, key, value)
        f.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def load_mesh(path: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    tri_blocks = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "triangle"]
    if not tri_blocks:
        raise RuntimeError("mesh contains no linear triangles")
    tri = np.concatenate(tri_blocks)
    tags_by_name = {name.lower(): int(v[0]) for name, v in mesh.field_data.items()}
    expected = {"domain", "bottom", "top", "crack", "right"}
    if not expected.issubset(tags_by_name):
        raise RuntimeError(f"missing physical groups: {sorted(expected - tags_by_name.keys())}")
    boundaries = {name: [] for name in ("bottom", "top", "crack", "right")}
    physical = mesh.cell_data.get("gmsh:physical", [])
    for ib, block in enumerate(mesh.cells):
        if block.type != "line":
            continue
        for edge, tag in zip(block.data, physical[ib]):
            for name in boundaries:
                if int(tag) == tags_by_name[name]:
                    boundaries[name].append(np.asarray(edge[:2], dtype=np.int64))
    edges = {k: np.asarray(v, dtype=np.int64).reshape((-1, 2)) for k, v in boundaries.items()}
    for name in ("bottom", "top", "crack"):
        if edges[name].size == 0:
            raise RuntimeError(f"physical group {name!r} has no line elements")
    xy = nodes[tri]
    twice = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
             - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    flip = twice < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    xy = nodes[tri]
    areas = 0.5 * ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                   - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    if np.any(areas <= 0):
        raise RuntimeError("nonpositive triangle area")
    groups = {k: np.unique(v) for k, v in edges.items()}
    return nodes, tri, areas, edges, groups, sorted(expected)


def boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lookup = {}
    for ie, t in enumerate(tri):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            lookup[tuple(sorted((int(a), int(b))))] = ie
    return np.asarray([lookup[tuple(sorted(map(int, e)))] for e in edges], dtype=np.int64)


def preprocess(nodes, tri, areas):
    x, y = nodes[:, 0], nodes[:, 1]
    i, j, k = tri.T
    gx = np.stack((y[j] - y[k], y[k] - y[i], y[i] - y[j]), axis=1) / (2 * areas[:, None])
    gy = np.stack((x[k] - x[j], x[i] - x[k], x[j] - x[i]), axis=1) / (2 * areas[:, None])
    b = np.zeros((len(tri), 3, 6))
    for a in range(3):
        b[:, 0, 2 * a] = gx[:, a]
        b[:, 1, 2 * a + 1] = gy[:, a]
        b[:, 2, 2 * a] = gy[:, a]
        b[:, 2, 2 * a + 1] = gx[:, a]
    dofs = np.empty((len(tri), 6), dtype=np.int64)
    dofs[:, 0::2], dofs[:, 1::2] = 2 * tri, 2 * tri + 1
    mass = areas[:, None, None] * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    return gx, gy, b, dofs, mass, grad


def make_solvers(*, nodes_np, tri_np, areas_np, gx_np, gy_np, b_np, dofs_np,
                 mass_np, grad_np, free_u_np, free_c_np, top_edges_np,
                 top_cells_np, dtype, tol, maxiter, c_boundary_value,
                 c_enrichment_cap):
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    areas, gx, gy = map(lambda z: jnp.asarray(z, dtype=dtype), (areas_np, gx_np, gy_np))
    b, mass, grad = map(lambda z: jnp.asarray(z, dtype=dtype), (b_np, mass_np, grad_np))
    free_u, free_c = jnp.asarray(free_u_np, dtype=dtype), jnp.asarray(free_c_np, dtype=dtype)
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    top_edges, top_cells = jnp.asarray(top_edges_np), jnp.asarray(top_cells_np)
    edge_len = jnp.linalg.norm(nodes[top_edges[:, 1]] - nodes[top_edges[:, 0]], axis=1)
    nnode, ndof = len(nodes_np), 2 * len(nodes_np)
    tiny = jnp.asarray(1.0e-30, dtype=dtype)

    def scatter(ids, values, size):
        return jnp.zeros(size, dtype=dtype).at[ids.reshape(-1)].add(values.reshape(-1))

    def cell_state(u, d):
        strain = jnp.einsum("eij,ej->ei", b, u[dofs])
        tr = strain[:, 0] + strain[:, 1]
        dev2 = ((strain[:, 0] - tr / 3) ** 2 + (strain[:, 1] - tr / 3) ** 2
                + (tr / 3) ** 2 + 0.5 * strain[:, 2] ** 2)
        psi = 0.5 * K_BULK * jnp.maximum(tr, 0) ** 2 + MU * dev2
        dm = jnp.mean(jnp.clip(d[tri], 0, 1), axis=1)
        g = (1 - dm) ** 2 + KAPPA
        kv = K_BULK * jnp.where(tr >= 0, g, 1.0)
        c11, c12, c66 = kv + 4 * MU * g / 3, kv - 2 * MU * g / 3, MU * g
        constit = jnp.stack(
            (jnp.stack((c11, c12, jnp.zeros_like(g)), axis=1),
             jnp.stack((c12, c11, jnp.zeros_like(g)), axis=1),
             jnp.stack((jnp.zeros_like(g), jnp.zeros_like(g), c66), axis=1)),
            axis=1,
        )
        stress = jnp.einsum("eij,ej->ei", constit, strain)
        sigma_h = (stress[:, 0] + stress[:, 1]) / 3.0
        return strain, psi, constit, stress, sigma_h

    def solve_u(d, boundary, x0):
        _, _, constit, _, _ = cell_state(x0 + boundary, d)
        ke = areas[:, None, None] * jnp.einsum("eia,eij,ejb->eab", b, constit, b)
        def kmv(x):
            return scatter(dofs, jnp.einsum("eij,ej->ei", ke, x[dofs]), ndof)
        rhs = -free_u * kmv(boundary)
        def op(x):
            return free_u * kmv(free_u * x) + (1 - free_u) * x
        diag = free_u * scatter(dofs, jnp.diagonal(ke, axis1=1, axis2=2), ndof) + 1 - free_u
        sol, _ = cg(op, rhs, x0=free_u * x0, tol=tol, atol=0.0, maxiter=maxiter,
                    M=lambda r: r / jnp.maximum(diag, tiny))
        residual = jnp.linalg.norm(op(sol) - rhs) / (jnp.linalg.norm(rhs) + tiny)
        return boundary + free_u * sol, residual

    def project_sigma(sigma):
        rhs = scatter(tri, jnp.broadcast_to((areas * sigma / 3)[:, None], tri.shape), nnode)
        def op(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", mass, x[tri]), nnode)
        diag = scatter(tri, jnp.diagonal(mass, axis1=1, axis2=2), nnode)
        sol, _ = cg(op, rhs, tol=tol, atol=0.0, maxiter=maxiter,
                    M=lambda r: r / jnp.maximum(diag, tiny))
        return sol

    def solve_c(sigma_node, boundary, x0):
        # Slotboom transform y=c*exp(-beta*sigma_h) converts
        # div[D(grad(c)-beta*c*grad(sigma_h))]=0 into the symmetric,
        # positive-definite problem div[D*exp(beta*sigma_h)*grad(y)]=0.
        # This exponential fitting is robust where a direct Galerkin
        # advection-diffusion matrix is non-normal and poorly conditioned.
        beta = V_H / (JOULE_TO_N_MM * R_GAS * TEMPERATURE)
        exponent = beta * sigma_node
        weight = jnp.exp(jnp.mean(exponent[tri], axis=1))
        ke = D_H * weight[:, None, None] * grad
        ke = ke + 1.0e-14 * weight[:, None, None] * mass
        def kmv(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", ke, x[tri]), nnode)
        y_boundary = boundary * jnp.exp(-exponent)
        rhs = -free_c * kmv(y_boundary)
        def op(x):
            return free_c * kmv(free_c * x) + (1 - free_c) * x
        diag = free_c * scatter(tri, jnp.diagonal(ke, axis1=1, axis2=2), nnode) + 1 - free_c
        y0 = free_c * x0 * jnp.exp(-exponent)
        sol, _ = cg(op, rhs, x0=y0, tol=tol, atol=0.0, maxiter=maxiter,
                    M=lambda r: r / jnp.maximum(diag, tiny))
        y = y_boundary + free_c * sol
        c = jnp.exp(exponent) * y
        residual = jnp.linalg.norm(op(sol) - rhs) / (jnp.linalg.norm(rhs) + tiny)
        return jnp.clip(c, 0.0, c_enrichment_cap * c_boundary_value), residual

    def gc_cells(c):
        cm = jnp.mean(c[tri], axis=1)
        theta = cm / (cm + jnp.exp(-DELTA_GB / (R_GAS * TEMPERATURE)))
        return jnp.clip(1 - HYDROGEN_CHI * theta, 0, 1)

    def solve_d(history, gc_factor, x0):
        coeff = gc_factor * GC0 / L0 + 2 * history
        ke = gc_factor[:, None, None] * GC0 * L0 * grad + coeff[:, None, None] * mass
        rhs = scatter(tri, jnp.broadcast_to((2 * history * areas / 3)[:, None], tri.shape), nnode)
        def op(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", ke, x[tri]), nnode)
        diag = scatter(tri, jnp.diagonal(ke, axis1=1, axis2=2), nnode)
        sol, _ = cg(op, rhs, x0=x0, tol=tol, atol=0.0, maxiter=maxiter,
                    M=lambda r: r / jnp.maximum(diag, tiny))
        residual = jnp.linalg.norm(op(sol) - rhs) / (jnp.linalg.norm(rhs) + tiny)
        return sol, residual

    def qois(u, d, history, c):
        strain, psi, _, stress, sigma_h = cell_state(u, d)
        gc_f = gc_cells(c)
        reaction = jnp.sum(stress[top_cells, 1] * edge_len)
        dv = d[tri]
        gd_x, gd_y = jnp.sum(dv * gx, axis=1), jnp.sum(dv * gy, axis=1)
        gamma = jnp.sum(areas * (jnp.mean(dv * dv, axis=1) / (2 * L0)
                                 + 0.5 * L0 * (gd_x * gd_x + gd_y * gd_y)))
        return psi, sigma_h, gc_f, reaction, gamma

    return tuple(jax.jit(f) for f in (solve_u, project_sigma, solve_c, gc_cells, solve_d, qois))


CSV_FIELDS = (
    "step", "hydrogen_wtppm", "ch_env_1wppm_mol_per_mm3",
    "cH_boundary_mol_per_mm3", "top_displacement_mm", "reaction_force_N_per_mm",
    "crack_length_gamma_mm", "crack_length95_mm", "x_tip95_mm", "max_d",
    "max_cH", "min_Gc_factor", "max_sigma_h_mpa", "stagger_iterations",
    "stagger_converged", "stagger_error", "u_residual", "cH_residual",
    "d_residual", "step_wall_s", "cumulative_wall_s",
)


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def snapshot(path, *, nodes, tri, u, d, history, c_h, sigma_h, gc_factor, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, format=np.asarray("hydrogen_SENT_state_v1"),
        mesh_nodes=nodes, mesh_triangles=tri, u=u.reshape((-1, 2)), d=d,
        history=history, cH=c_h, sigma_h=sigma_h, Gc_factor=gc_factor,
        metadata_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )


def checkpoint(path, step, u, d, history, c_h, rows, completed):
    np.savez_compressed(
        path, step=np.asarray(step), u=u, d=d, history=history, cH=c_h,
        rows_json=np.asarray(json.dumps(rows)),
        thresholds_json=np.asarray(json.dumps(sorted(completed))),
    )


def load_checkpoint(path):
    z = np.load(path, allow_pickle=False)
    return (int(z["step"]), z["u"], z["d"], z["history"], z["cH"],
            json.loads(str(z["rows_json"])), set(json.loads(str(z["thresholds_json"]))))


def crack_tip95(nodes, tri, d):
    dm = np.mean(d[tri], axis=1)
    centers = np.mean(nodes[tri], axis=1)
    mask = ((centers[:, 0] >= A0 - 2 * L0) & (abs(centers[:, 1] - YC) <= 4 * L0)
            & (dm >= 0.95))
    tip = max(A0, float(np.max(nodes[tri[mask], 0])) if np.any(mask) else A0)
    return tip - A0, tip


def reference_data(args):
    path = args.reference
    if path is None and args.reference_dir is not None:
        candidate = args.reference_dir / "data.csv"
        path = candidate if candidate.is_file() else None
    if path is None or not path.is_file():
        return None, path
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data, path


def main() -> None:
    args = parser().parse_args()
    if args.hydrogen_wtppm < 0:
        raise ValueError("--hydrogen-wtppm must be non-negative")
    if args.ch_env_1wppm <= 0:
        raise ValueError("--ch-env-1wppm must be positive")
    c_boundary_value = args.hydrogen_wtppm * args.ch_env_1wppm
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX found no devices")
    if args.platform == "gpu" and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError(f"GPU requested, available devices: {devices}")
    slug = device_slug(devices[0])
    case_root = (
        script_root()
        / "outputs"
        / f"results_{slug}"
        / "hydrogen_fracture_SENT"
        / "jax"
    )
    if args.outdir is None:
        args.outdir = case_root / concentration_tag(args.hydrogen_wtppm)
    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh or case_root / (
        "mesh_smoke.msh" if args.smoke_only else "mesh_SENT.msh"
    )
    if not mesh_path.is_file():
        generate_mesh(mesh_path, smoke=args.smoke_only)
    nodes, tri, areas, edges, groups, physical = load_mesh(mesh_path)
    if args.mesh_only:
        print(f"mesh: {len(nodes)} nodes, {len(tri)} triangles")
        print(f"physical groups: {physical}")
        print("boundary nodes: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
        return
    gx, gy, b, dofs, mass, grad = preprocess(nodes, tri, areas)
    top_cells = boundary_cells(tri, edges["top"])
    ndof, nnode = 2 * len(nodes), len(nodes)
    fixed_u = np.unique(np.concatenate((2 * groups["bottom"], 2 * groups["bottom"] + 1,
                                        2 * groups["top"] + 1)))
    free_u = np.ones(ndof); free_u[fixed_u] = 0
    free_c = np.ones(nnode); free_c[groups["crack"]] = 0
    solve_u, project_sigma, solve_c, gc_cells, solve_d, qois = make_solvers(
        nodes_np=nodes, tri_np=tri, areas_np=areas, gx_np=gx, gy_np=gy, b_np=b,
        dofs_np=dofs, mass_np=mass, grad_np=grad, free_u_np=free_u,
        free_c_np=free_c, top_edges_np=edges["top"], top_cells_np=top_cells,
        dtype=dtype, tol=args.cg_tol, maxiter=args.cg_maxiter,
        c_boundary_value=c_boundary_value,
        c_enrichment_cap=CH_ENRICHMENT_CAP,
    )
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    steps, max_stagger = (1, 2) if args.smoke_only else (
        min(args.max_steps, NOMINAL_STEPS), MAX_STAGGER)
    adaptive = not args.no_adaptive_steps and not args.smoke_only
    cp_path, csv_path = args.outdir / "checkpoint_latest.npz", args.outdir / "data.csv"
    snap_dir = args.outdir / "snapshots"
    if args.resume and cp_path.is_file():
        start, u0, d0, h0, c0, rows, completed = load_checkpoint(cp_path)
    else:
        start, rows, completed = 0, [], set()
        u0, d0, h0 = np.zeros(ndof), np.zeros(nnode), np.zeros(len(tri))
        c0 = np.zeros(nnode)
        c0[groups["crack"]] = c_boundary_value
    u, d, history, c_h = map(lambda z: jnp.asarray(z, dtype=dtype), (u0, d0, h0, c0))
    c_boundary_np = np.zeros(nnode)
    c_boundary_np[groups["crack"]] = c_boundary_value
    c_boundary = jnp.asarray(c_boundary_np, dtype=dtype)
    top_y = 2 * groups["top"] + 1
    du_base, du_min, du = U_MAX / NOMINAL_STEPS, U_MAX / NOMINAL_STEPS / 16, U_MAX / NOMINAL_STEPS
    u_prev = float(rows[-1]["top_displacement_mm"]) if rows else 0.0
    prev_d = float(rows[-1]["max_d"]) if rows else 0.0
    prev_l = float(rows[-1]["crack_length95_mm"]) if rows else 0.0
    run_t0, total_stagger, accepted, attempts = time.perf_counter(), 0, start, 0
    max_attempts = NOMINAL_STEPS * 16 * 4 + 50
    print("=" * 76)
    print("Liu 2025 Fig.7 hydrogen SENT — matrix-free JAX FEM")
    print(f"JAX {jax.__version__}; Python {platform.python_version()}; devices={devices}")
    print(f"mesh: {nnode} nodes, {len(tri)} triangles; groups={physical}")
    print(
        f"case: {args.hydrogen_wtppm:g} wt-ppm; "
        f"cH_boundary={c_boundary_value:.6e} mol/mm^3; "
        f"dtype={args.dtype}; stagger cap={max_stagger}"
    )
    print("=" * 76)
    while ((accepted < 1) if args.smoke_only else
           (accepted < steps and u_prev < U_MAX - 1e-14 and attempts < max_attempts)):
        attempts += 1
        step_t0 = time.perf_counter()
        target = min(u_prev + du, U_MAX)
        old = (u, d, history, c_h)
        d_step, h_step, c_step = d, history, c_h
        boundary_np = np.zeros(ndof); boundary_np[top_y] = target
        boundary = jnp.asarray(boundary_np, dtype=dtype)
        err = np.inf
        progress_t = step_t0
        for it in range(1, max_stagger + 1):
            d_iter = d
            u, ur = solve_u(d, boundary, u)
            _, sigma_h0, _, _, _ = qois(u, d, history, c_h)
            sigma_node = project_sigma(sigma_h0)
            c_h, cr = solve_c(sigma_node, c_boundary, c_h)
            gc_f = gc_cells(c_h)
            psi, _, _, _, _ = qois(u, d, history, c_h)
            history = jnp.maximum(history, psi)
            d_trial, dr = solve_d(history, gc_f, d)
            d = jnp.clip(jnp.maximum(d_trial, d_step), 0, 1)
            jax.block_until_ready(d)
            err = float(jax.device_get(jnp.linalg.norm(d - d_iter) /
                                       (jnp.linalg.norm(d) + 1e-30)))
            now = time.perf_counter()
            if (args.inner_progress_seconds > 0
                    and now - progress_t >= args.inner_progress_seconds):
                inner_residuals = tuple(float(jax.device_get(v)) for v in (ur, cr, dr))
                print(f"  target={target:.6e} fp={it:3d} err={err:.3e} "
                      f"res(u,cH,d)=({inner_residuals[0]:.3e},"
                      f"{inner_residuals[1]:.3e},{inner_residuals[2]:.3e}) "
                      f"elapsed={now-step_t0:.1f}s", flush=True)
                progress_t = now
            if err < STAGGER_TOL:
                break
        converged = np.isfinite(err) and err < STAGGER_TOL
        if not converged and args.reject_stagger_cap:
            u, d, history, c_h = old
            raise RuntimeError(f"stagger cap at target u={target:.8g}, error={err:.3e}")
        residuals = tuple(float(jax.device_get(v)) for v in (ur, cr, dr))
        if not all(np.isfinite(v) and v <= args.linear_residual_tol for v in residuals):
            u, d, history, c_h = old
            raise RuntimeError(
                "linear residual limit exceeded: "
                f"u={residuals[0]:.3e}, cH={residuals[1]:.3e}, d={residuals[2]:.3e}"
            )
        psi, sigma_h, gc_f, reaction, gamma = qois(u, d, history, c_h)
        jax.block_until_ready(reaction)
        uh, dh, hh, ch, sh, gf = map(
            lambda z: np.asarray(jax.device_get(z)), (u, d, history, c_h, sigma_h, gc_f)
        )
        dmax, cmax = float(dh.max()), float(ch.max())
        l95, tip = crack_tip95(nodes, tri, dh)
        delta_d, delta_l = dmax - prev_d, l95 - prev_l
        if adaptive and du > du_min * (1 + 1e-9) and (delta_d > 0.08 or delta_l > 0.02):
            u, d, history, c_h = old
            du = max(du / 2, du_min)
            continue
        accepted += 1
        cumulative = time.perf_counter() - run_t0
        row = {
            "step": accepted,
            "hydrogen_wtppm": args.hydrogen_wtppm,
            "ch_env_1wppm_mol_per_mm3": args.ch_env_1wppm,
            "cH_boundary_mol_per_mm3": c_boundary_value,
            "top_displacement_mm": target,
            "reaction_force_N_per_mm": float(reaction),
            "crack_length_gamma_mm": max(float(gamma), 0.0),
            "crack_length95_mm": l95, "x_tip95_mm": tip, "max_d": dmax,
            "max_cH": cmax, "min_Gc_factor": float(gf.min()),
            "max_sigma_h_mpa": float(sh.max()), "stagger_iterations": it,
            "stagger_converged": int(converged), "stagger_error": err,
            "u_residual": residuals[0], "cH_residual": residuals[1],
            "d_residual": residuals[2],
            "step_wall_s": time.perf_counter() - step_t0, "cumulative_wall_s": cumulative,
        }
        rows.append(row); total_stagger += it
        meta = dict(version=VERSION, step=accepted, threshold=None,
                    top_displacement_mm=target, reaction_force_N_per_mm=float(reaction),
                    hydrogen_wtppm=args.hydrogen_wtppm,
                    ch_env_1wppm_mol_per_mm3=args.ch_env_1wppm,
                    cH_boundary_mol_per_mm3=c_boundary_value,
                    units="mm,N,MPa,s,mol", canonical=not args.smoke_only)
        crossed = [t for t in SNAPSHOT_THRESHOLDS if t not in completed and dmax >= t]
        if args.snapshots:
            for threshold in crossed:
                meta_t = dict(meta, threshold=threshold)
                snapshot(snap_dir / f"snapshot_dmax_{round(100*threshold):03d}.npz",
                         nodes=nodes, tri=tri, u=uh, d=dh, history=hh, c_h=ch,
                         sigma_h=sh, gc_factor=gf, meta=meta_t)
                completed.add(threshold)
        if accepted % args.checkpoint_every == 0 or accepted == steps:
            write_csv(csv_path, rows)
            checkpoint(cp_path, accepted, uh, dh, hh, ch, rows, completed)
        if accepted <= 5 or accepted % args.print_every == 0:
            print(f"step={accepted:4d} u={target:.6e} F={float(reaction):.6e} "
                  f"dmax={dmax:.4f} cHmax={cmax:.3e} fHmin={gf.min():.4f} "
                  f"fp={it} {'OK' if converged else 'CAP'} dt={row['step_wall_s']:.2f}s")
        u_prev, prev_d, prev_l = target, dmax, l95
        if adaptive and delta_d < 0.02 and delta_l < 0.005:
            du = min(1.5 * du, du_base)
        if l95 >= STOP_CRACK_LENGTH:
            print(f"early stop: crack_length95={l95:.6f} mm")
            break
    if not rows:
        raise RuntimeError("no step completed")
    write_csv(csv_path, rows)
    checkpoint(cp_path, accepted, uh, dh, hh, ch, rows, completed)
    if args.snapshots:
        meta["label"] = "final"
        snapshot(snap_dir / "snapshot_final.npz", nodes=nodes, tri=tri, u=uh, d=dh,
                 history=hh, c_h=ch, sigma_h=sh, gc_factor=gf, meta=meta)
    ref, ref_path = reference_data(args)
    comparison = None
    if ref is not None:
        names = ref.dtype.names or ()
        u_name = "top_displacement_mm" if "top_displacement_mm" in names else "u_mm"
        f_name = "reaction_force_N_per_mm" if "reaction_force_N_per_mm" in names else "F_N"
        uu = np.asarray([r["top_displacement_mm"] for r in rows])
        ff = np.asarray([r["reaction_force_N_per_mm"] for r in rows])
        rf = np.interp(uu, np.asarray(ref[u_name]), np.asarray(ref[f_name]))
        comparison = {"relative_l2_force": float(np.linalg.norm(ff-rf)/(np.linalg.norm(rf)+1e-30))}
    timing = {
        "version": VERSION, "jax_version": jax.__version__, "device": str(devices[0]),
        "device_slug": slug, "dtype": args.dtype, "nodes": nnode, "triangles": len(tri),
        "physical_groups": physical, "steps": len(rows), "total_wall_s": time.perf_counter()-run_t0,
        "average_stagger": total_stagger/max(len(rows), 1), "comparison": comparison,
        "reference": str(ref_path) if ref_path else None, "smoke_only": args.smoke_only,
        "hydrogen_wtppm": args.hydrogen_wtppm,
        "ch_env_1wppm_mol_per_mm3": args.ch_env_1wppm,
        "cH_boundary_mol_per_mm3": c_boundary_value,
    }
    (args.outdir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"completed {len(rows)} step(s); outputs: {args.outdir}")
    if ref is None:
        print("reference comparison skipped (no existing --reference/--reference-dir data.csv)")


if __name__ == "__main__":
    main()
