#!/usr/bin/env python3
"""
Standalone JAX/GPU finite-element solver for the Miehe (2010) SENT test.

Model
-----
* 2-D plane strain, P1 triangular elements
* hybrid elasticity: g(d) C : eps(u), g(d) = (1-d)^2 + eta
* Miehe spectral tensile-energy history
* AT2 phase field
* displacement/phase-field staggered solution
* pointwise damage irreversibility

This file intentionally contains mesh generation/loading, element kernels,
linear solvers, time stepping, checkpointing and post-processing. It imports no
local project module or data, so copying this one file is sufficient apart from
normal third-party Python dependencies.

GPU execution (WSL/Linux):
  python elastic_fracture_SENT_jax_gpu.py --platform gpu

Short smoke test:
  python elastic_fracture_SENT_jax_gpu.py --platform gpu --steps 5 --no-plot

Dependencies:
  jax (CUDA build for GPU), numpy, gmsh, meshio; matplotlib is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
from pathlib import Path


# Select the backend before importing JAX.  "auto" leaves JAX's choice intact.
def _preparse_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    args, _ = parser.parse_known_args()
    return args.platform


_PLATFORM_REQUEST = _preparse_platform()
if _PLATFORM_REQUEST != "auto":
    # Current JAX exposes NVIDIA as backend "cuda"; Device.platform remains "gpu".
    backend_name = "cuda" if _PLATFORM_REQUEST == "gpu" else _PLATFORM_REQUEST
    os.environ.setdefault("JAX_PLATFORMS", backend_name)
# WSL shares VRAM with Windows graphics processes. Avoid JAX's large startup
# preallocation and allocate only what this FE solve actually needs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "JAX is required. For an NVIDIA GPU under WSL/Linux install, for example:\n"
        "  python -m pip install 'jax[cuda13]' meshio\n"
        f"Original import error: {exc}"
    ) from exc


# =============================================================================
# Reference model constants (match elastic_fracture_SENT.jl)
# =============================================================================

W = 1.0
HEIGHT = 1.0
A0 = 0.5
Y_NOTCH = 0.5
CH_PRECRACK_FACTOR = 0.5

H_FINE = 0.005
H_COARSE = 0.05
REFINE_XMIN_SENT = 0.45
REFINE_XMAX_SENT = 1.00
REFINE_YMIN_SENT = 0.35
REFINE_YMAX_SENT = 0.65

LAMBDA = 121_150.0
MU = 80_770.0
GC = 2.7
L0 = 0.015
ETA = 1.0e-6

DEFAULT_DU = 8.0e-6
DEFAULT_U_END = 8.0e-3
DEFAULT_STEPS = 1000
DEFAULT_STAGGER_TOL = 1.0e-4
DEFAULT_MAX_STAGGER = 20
SNAPSHOT_THRESHOLDS = (0.20, 0.50, 0.80, 0.95, 0.99)
DEFAULT_MIN_STAGGER = 2


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
        / "elastic_fracture_sent"
        / "F_u_curve_spectral_eta1em6.csv"
    )
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU SENT brittle phase-field fracture solver"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="existing Gmsh mesh (default: generate mesh_SENT.msh in --outdir)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=default_reference,
        help="optional reference force-displacement CSV (missing files are skipped)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: outputs/results_<device>/elastic_fracture_SENT/jax)",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--du", type=float, default=DEFAULT_DU)
    parser.add_argument("--u-end", type=float, default=DEFAULT_U_END)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--stagger-tol", type=float, default=DEFAULT_STAGGER_TOL)
    parser.add_argument("--min-stagger", type=int, default=DEFAULT_MIN_STAGGER)
    parser.add_argument("--max-stagger", type=int, default=DEFAULT_MAX_STAGGER)
    parser.add_argument(
        "--allow-stagger-cap",
        dest="allow_stagger_cap",
        action="store_true",
        default=True,
        help=(
            "commit at the stagger iteration cap (default; matches Julia)"
        ),
    )
    parser.add_argument(
        "--reject-stagger-cap",
        dest="allow_stagger_cap",
        action="store_false",
        help="strict mode: reject an irreversible step at the stagger cap",
    )
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=3000)
    parser.add_argument("--linear-residual-tol", type=float, default=1.0e-7)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="disable sparse compressed scientific state snapshots",
    )
    parser.set_defaults(snapshots=True)
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


# =============================================================================
# Mesh and element preprocessing
# =============================================================================

def generate_sent_mesh(mesh_file: Path) -> None:
    """Generate the canonical SENT mesh used by the Julia/Gridap solver."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh is required: python -m pip install gmsh") from exc

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("SENT_tension")

        crack_half_width = CH_PRECRACK_FACTOR * H_FINE
        geo = gmsh.model.geo
        geo.addPoint(0.0, 0.0, 0.0, H_COARSE, 1)
        geo.addPoint(W, 0.0, 0.0, H_COARSE, 2)
        geo.addPoint(W, Y_NOTCH, 0.0, H_COARSE, 10)
        geo.addPoint(W, HEIGHT, 0.0, H_COARSE, 3)
        geo.addPoint(0.0, HEIGHT, 0.0, H_COARSE, 4)
        geo.addPoint(0.0, Y_NOTCH + crack_half_width, 0.0, H_COARSE, 5)
        geo.addPoint(A0, Y_NOTCH + crack_half_width, 0.0, H_FINE, 6)
        geo.addPoint(A0, Y_NOTCH, 0.0, H_FINE, 7)
        geo.addPoint(A0, Y_NOTCH - crack_half_width, 0.0, H_FINE, 8)
        geo.addPoint(0.0, Y_NOTCH - crack_half_width, 0.0, H_COARSE, 9)

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

        field = gmsh.model.mesh.field
        field.add("Box", 1)
        field.setNumber(1, "VIn", H_FINE)
        field.setNumber(1, "VOut", H_COARSE)
        field.setNumber(1, "XMin", REFINE_XMIN_SENT)
        field.setNumber(1, "XMax", REFINE_XMAX_SENT)
        field.setNumber(1, "YMin", REFINE_YMIN_SENT)
        field.setNumber(1, "YMax", REFINE_YMAX_SENT)
        field.setAsBackgroundMesh(1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.finalize()


def load_mesh(mesh_file: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc

    if not mesh_file.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_file}")
    mesh = meshio.read(str(mesh_file))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)

    triangles = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "triangle"]
    if not triangles:
        raise RuntimeError("mesh has no P1 triangle cells")
    tri = np.concatenate(triangles, axis=0)

    bottom_nodes: set[int] = set()
    top_nodes: set[int] = set()
    top_edges: list[np.ndarray] = []
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is not None:
        for block_index, block in enumerate(mesh.cells):
            if block.type != "line":
                continue
            tags = physical[block_index]
            for edge, tag in zip(block.data, tags):
                if int(tag) == 1:
                    bottom_nodes.update(int(i) for i in edge)
                elif int(tag) == 2:
                    top_nodes.update(int(i) for i in edge)
                    top_edges.append(np.asarray(edge, dtype=np.int64))

    tol = 1.0e-10
    if not bottom_nodes:
        bottom_nodes.update(np.where(np.abs(nodes[:, 1]) <= tol)[0].tolist())
    if not top_nodes:
        top_nodes.update(np.where(np.abs(nodes[:, 1] - HEIGHT) <= tol)[0].tolist())
    if not top_edges:
        top_sorted = sorted(top_nodes, key=lambda index: nodes[index, 0])
        top_edges = [
            np.asarray([a, b], dtype=np.int64)
            for a, b in zip(top_sorted[:-1], top_sorted[1:])
        ]
    bottom = np.asarray(sorted(bottom_nodes), dtype=np.int64)
    top = np.asarray(sorted(top_nodes), dtype=np.int64)
    if bottom.size == 0 or top.size == 0:
        raise RuntimeError("could not identify bottom/top Dirichlet nodes")

    # Enforce counter-clockwise orientation.
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

    return (
        mesh,
        nodes,
        tri,
        areas,
        bottom,
        top,
        np.asarray(top_edges, dtype=np.int64),
    )


def find_boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edge_to_cell: dict[tuple[int, int], int] = {}
    for cell, element in enumerate(tri):
        for a, b in (
            (element[0], element[1]),
            (element[1], element[2]),
            (element[2], element[0]),
        ):
            edge_to_cell[tuple(sorted((int(a), int(b))))] = cell
    cells = []
    for edge in edges:
        key = tuple(sorted((int(edge[0]), int(edge[1]))))
        if key not in edge_to_cell:
            raise RuntimeError(f"boundary edge {key} has no adjacent triangle")
        cells.append(edge_to_cell[key])
    return np.asarray(cells, dtype=np.int64)


def preprocess_elements(nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray):
    x = nodes[:, 0]
    y = nodes[:, 1]
    i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]
    grad_x = np.stack(
        [y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]], axis=1
    ) / (2.0 * areas[:, None])
    grad_y = np.stack(
        [x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]], axis=1
    ) / (2.0 * areas[:, None])

    nel = tri.shape[0]
    bmat = np.zeros((nel, 3, 6), dtype=np.float64)
    for a in range(3):
        bmat[:, 0, 2 * a] = grad_x[:, a]
        bmat[:, 1, 2 * a + 1] = grad_y[:, a]
        bmat[:, 2, 2 * a] = grad_y[:, a]
        bmat[:, 2, 2 * a + 1] = grad_x[:, a]

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

    mass_template = np.asarray(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=np.float64,
    ) / 12.0
    mass = areas[:, None, None] * mass_template[None, :, :]
    grad = areas[:, None, None] * (
        grad_x[:, :, None] * grad_x[:, None, :]
        + grad_y[:, :, None] * grad_y[:, None, :]
    )

    dofs = np.empty((nel, 6), dtype=np.int64)
    for a in range(3):
        dofs[:, 2 * a] = 2 * tri[:, a]
        dofs[:, 2 * a + 1] = 2 * tri[:, a] + 1
    return bmat, ke0, mass, grad, dofs


# =============================================================================
# JAX kernels and matrix-free CG solvers
# =============================================================================

def make_jax_solvers(
    *,
    tri_np: np.ndarray,
    dofs_np: np.ndarray,
    bmat_np: np.ndarray,
    ke0_np: np.ndarray,
    mass_np: np.ndarray,
    grad_np: np.ndarray,
    areas_np: np.ndarray,
    nodes_np: np.ndarray,
    top_edges_np: np.ndarray,
    top_cells_np: np.ndarray,
    free_mask_np: np.ndarray,
    dtype,
    eta: float,
    cg_tol: float,
    cg_maxiter: int,
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
    top_edge_lengths = jnp.linalg.norm(
        nodes[top_edges[:, 1]] - nodes[top_edges[:, 0]], axis=1
    )
    free = jnp.asarray(free_mask_np, dtype=dtype)
    ndof = int(free_mask_np.size)
    nnode = ndof // 2
    eta_j = jnp.asarray(eta, dtype=dtype)

    def displacement_weights(damage):
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        d_mean = jnp.mean(dv, axis=1)
        d2_integral_mean = (
            jnp.sum(dv * dv, axis=1)
            + dv[:, 0] * dv[:, 1]
            + dv[:, 1] * dv[:, 2]
            + dv[:, 2] * dv[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * d_mean + d2_integral_mean + eta_j

    def scatter_vector(indices, values, size):
        return jnp.zeros((size,), dtype=dtype).at[indices.reshape(-1)].add(values.reshape(-1))

    def displacement_matvec(x, weights):
        xe = x[dofs]
        ye = weights[:, None] * jnp.einsum("eij,ej->ei", ke0, xe)
        return scatter_vector(dofs, ye, ndof)

    def displacement_diagonal(weights):
        diag_e = weights[:, None] * jnp.diagonal(ke0, axis1=1, axis2=2)
        return scatter_vector(dofs, diag_e, ndof)

    @jax.jit
    def solve_displacement(damage, boundary_values, x0):
        weights = displacement_weights(damage)

        def kmv(x):
            return displacement_matvec(x, weights)

        rhs = -free * kmv(boundary_values)

        def operator(x):
            return free * kmv(free * x) + (1.0 - free) * x

        diagonal = free * displacement_diagonal(weights) + (1.0 - free)
        diagonal = jnp.maximum(diagonal, jnp.finfo(dtype).tiny)

        def precondition(r):
            return r / diagonal

        correction, _ = cg(
            operator,
            rhs,
            x0=free * x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=precondition,
        )
        displacement = boundary_values + free * correction
        residual = operator(correction) - rhs
        relative_residual = jnp.linalg.norm(residual) / (
            jnp.linalg.norm(rhs) + jnp.asarray(1.0e-30, dtype=dtype)
        )
        return displacement, relative_residual

    def phase_matrix(history):
        coefficient = GC / L0 + 2.0 * history
        return GC * L0 * grad + coefficient[:, None, None] * mass

    def phase_matvec(x, ke):
        xe = x[tri]
        ye = jnp.einsum("eij,ej->ei", ke, xe)
        return scatter_vector(tri, ye, nnode)

    @jax.jit
    def solve_phase(history, x0):
        ke = phase_matrix(history)
        rhs_e = (2.0 * history * areas / 3.0)[:, None] * jnp.ones(
            (1, 3), dtype=dtype
        )
        rhs = scatter_vector(tri, rhs_e, nnode)
        diagonal = scatter_vector(
            tri, jnp.diagonal(ke, axis1=1, axis2=2), nnode
        )
        diagonal = jnp.maximum(diagonal, jnp.finfo(dtype).tiny)

        def operator(x):
            return phase_matvec(x, ke)

        def precondition(r):
            return r / diagonal

        damage, _ = cg(
            operator,
            rhs,
            x0=x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=precondition,
        )
        residual = operator(damage) - rhs
        relative_residual = jnp.linalg.norm(residual) / (
            jnp.linalg.norm(rhs) + jnp.asarray(1.0e-30, dtype=dtype)
        )
        return damage, relative_residual

    @jax.jit
    def strain_and_history(displacement, old_history):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        e11 = strain[:, 0]
        e22 = strain[:, 1]
        e12 = 0.5 * strain[:, 2]
        average = 0.5 * (e11 + e22)
        radius = jnp.sqrt((0.5 * (e11 - e22)) ** 2 + e12**2)
        principal_1 = average + radius
        principal_2 = average - radius
        trace_positive = jnp.maximum(e11 + e22, 0.0)
        psi_plus = (
            0.5 * LAMBDA * trace_positive**2
            + MU
            * (
                jnp.maximum(principal_1, 0.0) ** 2
                + jnp.maximum(principal_2, 0.0) ** 2
            )
        )
        return strain, jnp.maximum(old_history, psi_plus)

    @jax.jit
    def boundary_reaction(displacement, damage):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        sigma_22 = LAMBDA * strain[:, 0] + (LAMBDA + 2.0 * MU) * strain[:, 1]
        edge_damage = jnp.clip(damage[top_edges], 0.0, 1.0)
        d0 = edge_damage[:, 0]
        d1 = edge_damage[:, 1]
        # Exact line average of (1-d)^2 + eta for linear d on each edge.
        g_edge = (
            1.0
            - (d0 + d1)
            + (d0 * d0 + d0 * d1 + d1 * d1) / 3.0
            + eta_j
        )
        return jnp.sum(g_edge * sigma_22[top_cells] * top_edge_lengths)

    return (
        solve_displacement,
        solve_phase,
        strain_and_history,
        boundary_reaction,
    )


# =============================================================================
# Output helpers
# =============================================================================

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


def write_history(csv_file: Path, rows: list[dict[str, float]]) -> None:
    with csv_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    step: int,
    displacement: np.ndarray,
    damage: np.ndarray,
    history: np.ndarray,
    rows: list[dict[str, float]],
) -> None:
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
    rows = json.loads(str(data["rows_json"]))
    return (
        int(data["step"]),
        np.asarray(data["displacement"]),
        np.asarray(data["damage"]),
        np.asarray(data["history"]),
        rows,
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
        benchmark=np.asarray("SENT_spectral_eta1em6"),
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


def save_vtu(
    filename: Path,
    points: np.ndarray,
    tri: np.ndarray,
    displacement: np.ndarray,
    damage: np.ndarray,
    history: np.ndarray,
) -> None:
    import meshio

    points3 = np.column_stack([points, np.zeros(points.shape[0])])
    meshio.write_points_cells(
        str(filename),
        points3,
        [("triangle", tri)],
        point_data={
            "damage": damage,
            "displacement": np.column_stack(
                [
                    displacement[0::2],
                    displacement[1::2],
                    np.zeros(points.shape[0]),
                ]
            ),
        },
        cell_data={"history": [history]},
    )


def load_reference(path: Path | None):
    if path is None or not path.is_file():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "step": np.asarray(data["step"], dtype=np.int64),
        "u": np.asarray(data["u_mm"], dtype=np.float64),
        "force": np.asarray(data["F_N"], dtype=np.float64),
        "time": np.asarray(data["time_s"], dtype=np.float64),
    }


def compare_reference(rows: list[dict[str, float]], reference):
    if reference is None or not rows:
        return None
    u = np.asarray([r["u_mm"] for r in rows])
    force = np.asarray([r["F_N"] for r in rows])
    ref_force = np.interp(u, reference["u"], reference["force"])
    denominator = np.linalg.norm(ref_force) + 1.0e-30
    relative_l2 = np.linalg.norm(force - ref_force) / denominator
    peak = float(np.max(force))
    covered = reference["u"] <= np.max(u) + 1.0e-14
    peak_ref = float(np.max(reference["force"][covered]))
    return {
        "relative_l2_force": float(relative_l2),
        "peak_force_N": peak,
        "reference_peak_force_N": peak_ref,
        "peak_force_relative_error": (peak - peak_ref) / peak_ref,
    }


def save_plot(outdir: Path, rows: list[dict[str, float]], reference) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot skipped: matplotlib is not installed")
        return

    u = np.asarray([r["u_mm"] for r in rows])
    force = np.asarray([r["F_N"] for r in rows])
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
    ax.set_xlabel(r"Displacement $u$ ($10^{-3}$ mm)")
    ax.set_ylabel("Reaction force (kN)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "F_u_comparison.png", dpi=180)
    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.min_stagger < 1 or args.max_stagger < args.min_stagger:
        raise ValueError("invalid stagger iteration limits")
    if args.cg_tol <= 0.0 or args.linear_residual_tol <= 0.0:
        raise ValueError("linear solver tolerances must be positive")
    use_x64 = args.dtype == "float64"
    jax.config.update("jax_enable_x64", use_x64)
    dtype = jnp.float64 if use_x64 else jnp.float32
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    if args.platform == "gpu" and all(d.platform != "gpu" for d in devices):
        raise RuntimeError(f"GPU requested, but JAX devices are {devices}")
    selected_device_slug = device_slug(devices[0])
    if args.outdir is None:
        root = script_root()
        args.outdir = (
            root / "outputs" / f"results_{selected_device_slug}"
            / "elastic_fracture_SENT" / "jax"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = args.outdir / "checkpoint_latest.npz"
    csv_file = args.outdir / "F_u_curve_jax.csv"
    snapshot_dir = args.outdir / "snapshots"

    if args.mesh is None:
        args.mesh = args.outdir / "mesh_SENT.msh"
    if not args.mesh.is_file():
        print(f"generating canonical SENT mesh: {args.mesh}")
        generate_sent_mesh(args.mesh)

    _, nodes, tri, areas, bottom, top, top_edges = load_mesh(args.mesh)
    top_cells = find_boundary_cells(tri, top_edges)
    bmat, ke0, mass, grad, dofs = preprocess_elements(nodes, tri, areas)
    nnode = nodes.shape[0]
    ndof = 2 * nnode
    fixed_dofs = np.unique(
        np.concatenate(
            [
                2 * bottom,
                2 * bottom + 1,
                2 * top,
                2 * top + 1,
            ]
        )
    )
    free_mask = np.ones(ndof, dtype=np.float64)
    free_mask[fixed_dofs] = 0.0
    top_y_dofs = 2 * top + 1

    (
        solve_displacement,
        solve_phase,
        strain_and_history,
        boundary_reaction,
    ) = make_jax_solvers(
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
        eta=args.eta,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
    )

    if args.resume and checkpoint_file.is_file():
        start_step, u_np, d_np, history_np, rows = load_checkpoint(checkpoint_file)
        print(f"resuming after step {start_step}: {checkpoint_file}")
    else:
        start_step = 0
        u_np = np.zeros(ndof, dtype=np.float64)
        d_np = np.zeros(nnode, dtype=np.float64)
        history_np = np.zeros(tri.shape[0], dtype=np.float64)
        rows: list[dict[str, float]] = []

    displacement = jnp.asarray(u_np, dtype=dtype)
    damage = jnp.asarray(d_np, dtype=dtype)
    history = jnp.asarray(history_np, dtype=dtype)
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
    print("Standalone JAX SENT brittle phase-field fracture")
    print(f"  JAX {jax.__version__}; Python {platform.python_version()}")
    print(f"  devices: {devices}")
    print(f"  dtype={args.dtype}, eta={args.eta:g}, reaction=boundary traction")
    print(
        f"  mesh={nnode} nodes, {tri.shape[0]} triangles, "
        f"{int(np.sum(free_mask))} free displacement DOFs"
    )
    print(
        f"  steps={args.steps}, du={args.du:.3e} mm, "
        f"stagger=[{args.min_stagger},{args.max_stagger}], tol={args.stagger_tol:g}"
    )
    print("=" * 78)

    solve_t0 = time.perf_counter()
    elapsed_offset = float(rows[-1]["time_s"]) if rows else 0.0
    compile_time = None
    converged_steps = 0

    for step in range(start_step + 1, args.steps + 1):
        step_t0 = time.perf_counter()
        prescribed_u = min(step * args.du, args.u_end)
        boundary_np = np.zeros(ndof, dtype=np.float64)
        boundary_np[top_y_dofs] = prescribed_u
        boundary_values = jnp.asarray(boundary_np, dtype=dtype)

        damage_step_old = damage
        history_step_old = history
        u_iter_old = None
        stagger_error = np.inf
        u_linear_residual = np.inf
        d_linear_residual = np.inf

        for stagger_iteration in range(1, args.max_stagger + 1):
            displacement_new, u_res = solve_displacement(
                damage, boundary_values, displacement
            )
            _, history_trial = strain_and_history(
                displacement_new, history_step_old
            )
            damage_trial, d_res = solve_phase(history_trial, damage)
            damage_new = jnp.clip(
                jnp.maximum(damage_trial, damage_step_old), 0.0, 1.0
            )

            # Explicit synchronization: timing and convergence must include GPU work.
            jax.block_until_ready(damage_new)
            u_linear_residual = float(jax.device_get(u_res))
            d_linear_residual = float(jax.device_get(d_res))

            if u_iter_old is None:
                u_error = np.inf
            else:
                du_norm = jnp.linalg.norm(
                    free_mask * (displacement_new - u_iter_old)
                )
                u_norm = jnp.linalg.norm(free_mask * displacement_new)
                u_error = float(
                    jax.device_get(du_norm / (u_norm + jnp.asarray(1.0e-30, dtype=dtype)))
                )
            dd_norm = jnp.linalg.norm(damage_new - damage)
            d_norm = jnp.linalg.norm(damage_new)
            d_error = float(
                jax.device_get(dd_norm / (d_norm + jnp.asarray(1.0e-30, dtype=dtype)))
            )
            stagger_error = max(u_error, d_error)

            u_iter_old = displacement_new
            displacement = displacement_new
            damage = damage_new
            history = history_trial

            if (
                stagger_iteration >= args.min_stagger
                and stagger_error < args.stagger_tol
            ):
                break

        stagger_converged = bool(
            np.isfinite(stagger_error) and stagger_error < args.stagger_tol
        )
        if not stagger_converged and not args.allow_stagger_cap:
            raise RuntimeError(
                f"step {step}: stagger failed after {args.max_stagger} iterations "
                f"(error={stagger_error:.3e}); irreversible state not saved"
            )
        if (
            not np.isfinite(u_linear_residual)
            or not np.isfinite(d_linear_residual)
            or max(u_linear_residual, d_linear_residual)
            > args.linear_residual_tol
        ):
            raise RuntimeError(
                f"step {step}: linear solve residual too large: "
                f"u={u_linear_residual:.3e}, d={d_linear_residual:.3e}"
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
            "u_linear_residual": u_linear_residual,
            "d_linear_residual": d_linear_residual,
        }
        rows.append(row)
        converged_steps += 1

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
            or stagger_iteration > 20
            or not stagger_converged
            or step == args.steps
        ):
            print(
                f"  step={step:4d} u={prescribed_u:.3e} "
                f"F={reaction:10.4f} N dmax={damage_max:.5f} "
                f"stag={stagger_iteration:2d} "
                f"{'OK' if stagger_converged else 'CAP'} "
                f"err={stagger_error:.2e} "
                f"lin=({u_linear_residual:.1e},{d_linear_residual:.1e}) "
                f"dt={step_time:.3f}s"
            )

        if args.save_every > 0 and (
            step % args.save_every == 0 or step == args.steps
        ):
            u_host = np.asarray(jax.device_get(displacement))
            d_host = np.asarray(jax.device_get(damage))
            h_host = np.asarray(jax.device_get(history))
            write_history(csv_file, rows)
            save_checkpoint(
                checkpoint_file, step, u_host, d_host, h_host, rows
            )

        if prescribed_u >= args.u_end:
            break

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
    if not args.no_vtk:
        save_vtu(
            args.outdir / "result_final.vtu",
            nodes,
            tri,
            u_host,
            d_host,
            h_host,
        )

    comparison = compare_reference(rows, reference)
    step_times = np.asarray([r["step_time_s"] for r in rows[-converged_steps:]])
    timing = {
        "jax_version": jax.__version__,
        "devices": [str(d) for d in devices],
        "device_strings": [str(d) for d in devices],
        "device_kinds": [d.device_kind for d in devices],
        "device_slug": selected_device_slug,
        "dtype": args.dtype,
        "reaction": "boundary_traction",
        "nodes": nnode,
        "triangles": int(tri.shape[0]),
        "free_displacement_dofs": int(np.sum(free_mask)),
        "steps_this_run": converged_steps,
        "total_s": total_time,
        "first_step_compile_and_solve_s": compile_time,
        "post_first_step_mean_s": (
            float(np.mean(step_times[1:])) if step_times.size > 1 else None
        ),
        "median_step_s": float(np.median(step_times)),
        "comparison": comparison,
    }
    with (args.outdir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    if not args.no_plot:
        save_plot(args.outdir, rows, reference)

    print("-" * 78)
    print(f"completed {converged_steps} steps in {total_time:.3f} s")
    print(f"first step (compile + solve): {compile_time:.3f} s")
    if step_times.size > 1:
        print(f"post-first mean step: {np.mean(step_times[1:]):.3f} s")
    if comparison is not None:
        print(
            "Julia comparison: "
            f"L2(F)={comparison['relative_l2_force']:.3%}, "
            f"peak error={comparison['peak_force_relative_error']:.3%}"
        )
    else:
        print(f"Julia comparison skipped: reference CSV not found at {args.reference}")
    print(f"outputs: {args.outdir}")


if __name__ == "__main__":
    main()
