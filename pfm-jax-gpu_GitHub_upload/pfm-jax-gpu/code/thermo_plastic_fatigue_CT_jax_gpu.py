#!/usr/bin/env python3
"""
Standalone FP64 JAX finite-element solver for the canonical ΔT=200 K
thermo-elasto-plastic CT fatigue benchmark.

The default run generates the unchanged Julia/Gmsh CT mesh in the device-aware
output directory.  It implements P1 plane-strain elasticity, one Chaboche
state per triangle, the Julia tensile-mode histories, accumulated fatigue
history, AT2 phase field, and the original 21-evaluation-per-cycle algorithm.

Normal GPU run:
  python thermo_plastic_fatigue_CT_jax_gpu.py --platform gpu

Short execution check (one maximum-load constitutive/phase-field step):
  python thermo_plastic_fatigue_CT_jax_gpu.py --platform gpu --smoke

Dependencies: JAX (CUDA build for GPU), NumPy, meshio, gmsh; matplotlib is
optional.
There are deliberately no project-local imports.
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
    from jax import lax  # noqa: E402
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "JAX is required. Install a CUDA build, for example:\n"
        "  python -m pip install 'jax[cuda13]' meshio gmsh\n"
        f"Original import error: {exc}"
    ) from exc


# Canonical revision-baseline constants ---------------------------------------
W_CT, H_CT, THICKNESS = 50.0, 48.0, 25.0
PIN_RADIUS, PIN1_X, PIN1_Y, PIN2_X, PIN2_Y = 5.0, 8.0, 12.0, 8.0, 36.0
Y_CENTER, V_TIP_X, V_TIP_Y = 24.0, 14.0, 24.0
NOTCH_LEN, NOTCH_HALF = 12.0, 1.0
PRECRACK_LEN, PRECRACK_TRANS_LEN, PRECRACK_HALF = 2.0, 0.8, 0.12
PRECRACK_RECT_LEN = PRECRACK_LEN - PRECRACK_TRANS_LEN
H_COARSE, H_FINE, H_PIN = 2.0, 0.15, 0.8
E, NU = 210000.0, 0.3
G = E / (2.0 * (1.0 + NU))
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
SIGMA_Y0, Q_INF, VOCE_B, CHABOCHE_C, CHABOCHE_GAMMA = (
    440.0, 55.0, 2.38, 23_554.0, 139.0
)
L0, GC0, ETA, ALPHA_T0 = 0.3, 2.7, 1.0e-3, 20.0
K0_HEAT, K1_HEAT, CV_VOL = 0.05, 0.05e-6, 3.925e-3
ALPHA_TH, T0, T_RIGHT_MAX, HEAT_RAMP_CYCLES, CHI_EFF = (
    1.2e-5, 300.0, 500.0, 30, 0.85
)
F_MAX_3D = 20_000.0
F_MAX, R_LOAD, F_MIN = F_MAX_3D / THICKNESS, 0.0, 0.0
DEFAULT_CYCLES = 2000
DEFAULT_STEPS_PER_CYCLE = 20
DEFAULT_PHASE_EVERY = 1
DEFAULT_MAX_STAGGER = 20
DEFAULT_STAGGER_TOL = 1.0e-6
DEFAULT_STOP_EXTENSION = 30.0
DEFAULT_STOP_X = V_TIP_X + DEFAULT_STOP_EXTENSION
DEFAULT_STALL_CYCLES = 40
DEFAULT_STALL_TOL = 1.0e-5
DEFAULT_INIT_DA = 5.0e-3
THERMAL_SNAPSHOT_CYCLES = (1, 10, 20, 30)
VTK_FRACTIONS = (0.10, 0.30, 0.60, 0.90)


def script_root() -> Path:
    """Use the repository root in-tree, or the script directory standalone."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code" else script_dir


def build_parser() -> argparse.ArgumentParser:
    root = script_root()
    refdir = (
        root
        / "data"
        / "julia_reference"
        / "thermo_plastic_fatigue_CT"
    )
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU CT Chaboche phase-field fatigue solver"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="existing CT mesh (default: generate CT_mesh.msh under the output directory)",
    )
    parser.add_argument("--reference", type=Path, default=refdir / "data.csv")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "output directory (default: "
            "outputs/results_<device>/thermo_plastic_fatigue_CT/jax)"
        ),
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--cycles-total", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--steps-per-cycle", type=int, default=DEFAULT_STEPS_PER_CYCLE)
    parser.add_argument("--phase-every", type=int, default=DEFAULT_PHASE_EVERY)
    parser.add_argument("--max-stagger", type=int, default=DEFAULT_MAX_STAGGER)
    parser.add_argument("--stagger-tol", type=float, default=DEFAULT_STAGGER_TOL)
    parser.add_argument("--picard-tol", type=float, default=1.0e-4)
    parser.add_argument("--max-picard", type=int, default=10)
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=5000)
    parser.add_argument("--linear-residual-tol", type=float, default=5.0e-7)
    parser.add_argument("--stop-crack-x", type=float, default=DEFAULT_STOP_X)
    parser.add_argument("--stall-cycles", type=int, default=DEFAULT_STALL_CYCLES)
    parser.add_argument("--stall-tol", type=float, default=DEFAULT_STALL_TOL)
    parser.add_argument("--init-da", type=float, default=DEFAULT_INIT_DA)
    parser.add_argument("--save-every", type=int, default=1, help="checkpoint blocks")
    parser.add_argument("--print-every", type=int, default=1, help="print blocks")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    return parser


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


# Mesh and P1 preprocessing ----------------------------------------------------
def create_ct_mesh(path: Path) -> Path:
    """Generate the revision h=l0/2 slit-precracked CT mesh."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit(
            "gmsh is required to generate the default mesh: "
            "python -m pip install gmsh meshio"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("CT_thermo_fatigue_canonical")
        geo = gmsh.model.geo

        points = (
            (0.0, 0.0, H_COARSE),
            (W_CT, 0.0, H_COARSE),
            (W_CT, H_CT, H_COARSE),
            (0.0, H_CT, H_COARSE),
            (0.0, Y_CENTER + NOTCH_HALF, H_FINE),
            (NOTCH_LEN, Y_CENTER + NOTCH_HALF, H_FINE),
            (NOTCH_LEN, Y_CENTER - NOTCH_HALF, H_FINE),
            (0.0, Y_CENTER - NOTCH_HALF, H_FINE),
            (V_TIP_X - PRECRACK_RECT_LEN, Y_CENTER + PRECRACK_HALF, H_FINE),
            (V_TIP_X, Y_CENTER + PRECRACK_HALF, H_FINE),
            (V_TIP_X, Y_CENTER - PRECRACK_HALF, H_FINE),
            (V_TIP_X - PRECRACK_RECT_LEN, Y_CENTER - PRECRACK_HALF, H_FINE),
            (PIN1_X, PIN1_Y, H_COARSE),
            (PIN1_X + PIN_RADIUS, PIN1_Y, H_COARSE),
            (PIN1_X, PIN1_Y + PIN_RADIUS, H_COARSE),
            (PIN1_X - PIN_RADIUS, PIN1_Y, H_COARSE),
            (PIN1_X, PIN1_Y - PIN_RADIUS, H_COARSE),
            (PIN2_X, PIN2_Y, H_COARSE),
            (PIN2_X + PIN_RADIUS, PIN2_Y, H_COARSE),
            (PIN2_X, PIN2_Y + PIN_RADIUS, H_COARSE),
            (PIN2_X - PIN_RADIUS, PIN2_Y, H_COARSE),
            (PIN2_X, PIN2_Y - PIN_RADIUS, H_COARSE),
        )
        for tag, (x, y, size) in enumerate(points, start=1):
            geo.addPoint(x, y, 0.0, size, tag)

        straight_lines = (
            (1, 2), (2, 3), (3, 4), (4, 5), (8, 1),
            (5, 6), (6, 9), (12, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
        )
        for tag, (start, end) in enumerate(straight_lines, start=1):
            geo.addLine(start, end, tag)

        pin1_lines = (13, 14, 15, 16)
        pin2_lines = (17, 18, 19, 20)
        arcs = (
            (14, 13, 15), (15, 13, 16), (16, 13, 17), (17, 13, 14),
            (19, 18, 20), (20, 18, 21), (21, 18, 22), (22, 18, 19),
        )
        for tag, (start, center, end) in enumerate(arcs, start=13):
            geo.addCircleArc(start, center, end, tag)

        geo.addCurveLoop(list(range(1, 13)), 1)
        geo.addCurveLoop(list(pin1_lines), 2)
        geo.addCurveLoop(list(pin2_lines), 3)
        geo.addPlaneSurface([1, 2, 3], 1)
        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        gmsh.model.addPhysicalGroup(1, list(pin2_lines), 1)
        gmsh.model.setPhysicalName(1, 1, "pin_upper")
        gmsh.model.addPhysicalGroup(1, list(pin1_lines), 2)
        gmsh.model.setPhysicalName(1, 2, "pin_lower")
        gmsh.model.addPhysicalGroup(0, [2], 3)
        gmsh.model.setPhysicalName(0, 3, "fixed_point")
        gmsh.model.addPhysicalGroup(0, [1], 8)
        gmsh.model.setPhysicalName(0, 8, "rot_lock_y")
        gmsh.model.addPhysicalGroup(1, [4, 5], 4)
        gmsh.model.setPhysicalName(1, 4, "left")
        gmsh.model.addPhysicalGroup(1, [2], 5)
        gmsh.model.setPhysicalName(1, 5, "right")

        field = gmsh.model.mesh.field
        field.add("Box", 1)
        field.setNumber(1, "VIn", H_FINE)
        field.setNumber(1, "VOut", H_COARSE)
        field.setNumber(1, "XMin", V_TIP_X - 2.0)
        field.setNumber(1, "XMax", W_CT)
        field.setNumber(1, "YMin", V_TIP_Y - 1.25)
        field.setNumber(1, "YMax", V_TIP_Y + 1.25)
        for fid, px, py in ((2, PIN1_X, PIN1_Y), (3, PIN2_X, PIN2_Y)):
            field.add("Ball", fid)
            field.setNumber(fid, "Radius", 7.5)
            field.setNumber(fid, "VIn", H_PIN)
            field.setNumber(fid, "VOut", H_COARSE)
            field.setNumber(fid, "XCenter", px)
            field.setNumber(fid, "YCenter", py)
        field.add("Min", 4)
        field.setNumbers(4, "FieldsList", [1, 2, 3])
        field.setAsBackgroundMesh(4)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def load_mesh(path: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    if not path.is_file():
        raise FileNotFoundError(f"mesh not found: {path}")
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangles: list[np.ndarray] = []
    upper_edges: list[np.ndarray] = []
    lower_edges: list[np.ndarray] = []
    fixed_nodes: set[int] = set()
    rot_nodes: set[int] = set()
    left_nodes: set[int] = set()
    right_nodes: set[int] = set()
    physical = mesh.cell_data.get("gmsh:physical")
    for block_index, block in enumerate(mesh.cells):
        tags = None if physical is None else np.asarray(physical[block_index])
        if block.type == "triangle":
            triangles.append(np.asarray(block.data, dtype=np.int64))
        elif block.type == "line" and tags is not None:
            for edge, tag in zip(block.data, tags):
                if int(tag) == 1:
                    upper_edges.append(np.asarray(edge, dtype=np.int64))
                elif int(tag) == 2:
                    lower_edges.append(np.asarray(edge, dtype=np.int64))
                elif int(tag) == 4:
                    left_nodes.update(map(int, edge))
                elif int(tag) == 5:
                    right_nodes.update(map(int, edge))
        elif block.type == "vertex" and tags is not None:
            for point, tag in zip(block.data, tags):
                if int(tag) == 3:
                    fixed_nodes.add(int(point[0]))
                elif int(tag) == 8:
                    rot_nodes.add(int(point[0]))
    if not triangles:
        raise RuntimeError("mesh has no first-order triangles")
    tri = np.concatenate(triangles)
    if not fixed_nodes:
        fixed_nodes.update(
            np.where(
                np.isclose(nodes[:, 0], W_CT, atol=1.0e-10)
                & np.isclose(nodes[:, 1], 0.0, atol=1.0e-10)
            )[0].tolist()
        )
    if not (upper_edges and lower_edges and fixed_nodes and rot_nodes and left_nodes and right_nodes):
        raise RuntimeError("canonical mesh physical groups are incomplete")
    xy = nodes[tri]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0.0
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
        nodes,
        tri,
        areas,
        np.asarray(upper_edges, dtype=np.int64),
        np.asarray(lower_edges, dtype=np.int64),
        np.asarray(sorted(fixed_nodes), dtype=np.int64),
        np.asarray(sorted(rot_nodes), dtype=np.int64),
        np.asarray(sorted(left_nodes), dtype=np.int64),
        np.asarray(sorted(right_nodes), dtype=np.int64),
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
    mass = (
        areas[:, None, None]
        * np.asarray([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.float64)[None]
        / 12.0
    )
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    return bmat, dofs, ke0, mass, grad


def boundary_force(nodes: np.ndarray, edges: np.ndarray, traction_y: float) -> np.ndarray:
    force = np.zeros(2 * len(nodes), dtype=np.float64)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    values = 0.5 * traction_y * lengths
    np.add.at(force, 2 * edges[:, 0] + 1, values)
    np.add.at(force, 2 * edges[:, 1] + 1, values)
    return force


# JAX kernels -----------------------------------------------------------------
def make_kernels(
    *,
    tri_np,
    dofs_np,
    bmat_np,
    ke0_np,
    mass_np,
    grad_np,
    areas_np,
    free_np,
    thermal_free_np,
    thermal_left_np,
    thermal_right_np,
    upper_unit_np,
    lower_unit_np,
    cg_tol,
    cg_maxiter,
):
    dtype = jnp.float64
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    bmat, ke0 = jnp.asarray(bmat_np, dtype=dtype), jnp.asarray(ke0_np, dtype=dtype)
    mass, grad = jnp.asarray(mass_np, dtype=dtype), jnp.asarray(grad_np, dtype=dtype)
    areas = jnp.asarray(areas_np, dtype=dtype)
    free = jnp.asarray(free_np, dtype=dtype)
    thermal_free = jnp.asarray(thermal_free_np, dtype=dtype)
    thermal_left = jnp.asarray(thermal_left_np, dtype=dtype)
    thermal_right = jnp.asarray(thermal_right_np, dtype=dtype)
    upper_unit = jnp.asarray(upper_unit_np, dtype=dtype)
    lower_unit = jnp.asarray(lower_unit_np, dtype=dtype)
    ndof, nnode = len(free_np), len(free_np) // 2
    tiny = jnp.asarray(1.0e-30, dtype=dtype)

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(
            values.reshape(-1)
        )

    def degradation_integral_weight(damage):
        d = jnp.clip(damage[tri], 0.0, 1.0)
        # Exact element average of (1-d)^2+eta for linear P1 d.
        dmean = jnp.mean(d, axis=1)
        d2mean = (
            jnp.sum(d * d, axis=1)
            + d[:, 0] * d[:, 1]
            + d[:, 1] * d[:, 2]
            + d[:, 2] * d[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * dmean + d2mean + ETA

    def sigma_plastic(ep):
        trace = ep[:, 0] + ep[:, 1] + ep[:, 2]
        return jnp.stack(
            [
                2.0 * G * ep[:, 0] + LAMBDA * trace,
                2.0 * G * ep[:, 1] + LAMBDA * trace,
                2.0 * G * ep[:, 3],
            ],
            axis=1,
        )

    def kmv(x, weights):
        ye = weights[:, None] * jnp.einsum("eij,ej->ei", ke0, x[dofs])
        return scatter(dofs, ye, ndof)

    @jax.jit
    def solve_heat(temperature_old, damage, plastic_heat, right_value, dt):
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        qdamage = jnp.stack(
            [
                (4.0 * dv[:, 0] + dv[:, 1] + dv[:, 2]) / 6.0,
                (dv[:, 0] + 4.0 * dv[:, 1] + dv[:, 2]) / 6.0,
                (dv[:, 0] + dv[:, 1] + 4.0 * dv[:, 2]) / 6.0,
            ],
            axis=1,
        )
        svolos = jnp.maximum(qdamage * (2.0 - qdamage), 0.0) ** 1.5
        mt = jnp.clip((1.0 - svolos) / (1.0 + 0.5 * svolos), 0.0, 1.0)
        # Gridap's degree-2 triangle rule averages these three values.
        conductivity = K1_HEAT + jnp.mean(mt, axis=1) * (K0_HEAT - K1_HEAT)
        emat = conductivity[:, None, None] * grad + (CV_VOL / dt) * mass
        rhs_e = (CV_VOL / dt) * jnp.einsum(
            "eij,ej->ei", mass, temperature_old[tri]
        ) + (plastic_heat * areas / 3.0)[:, None]
        rhs = scatter(tri, rhs_e, nnode)
        prescribed = T0 * thermal_left + right_value * thermal_right

        def operator(x):
            full = thermal_free * x
            assembled = phase_matvec(full, emat)
            return thermal_free * assembled + (1.0 - thermal_free) * x

        boundary_action = phase_matvec(prescribed, emat)
        rhs = thermal_free * (rhs - boundary_action) + prescribed
        diagonal = thermal_free * scatter(
            tri, jnp.diagonal(emat, axis1=1, axis2=2), nnode
        ) + (1.0 - thermal_free)
        solved, _ = cg(
            operator, rhs, x0=temperature_old, tol=cg_tol, atol=0.0,
            maxiter=cg_maxiter, M=lambda r: r / jnp.maximum(diagonal, tiny)
        )
        residual = operator(solved) - rhs
        relres = jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + tiny)
        return thermal_free * solved + prescribed, relres

    @jax.jit
    def solve_displacement(damage, ep, temperature, load, x0):
        weights = degradation_integral_weight(damage)
        thermal_strain = ALPHA_TH * (jnp.mean(temperature[tri], axis=1) - T0)
        eigen = ep.at[:, 0].add(thermal_strain).at[:, 1].add(thermal_strain)
        eigen = eigen.at[:, 2].add(thermal_strain)
        sp = sigma_plastic(eigen)
        plastic_e = weights[:, None] * areas[:, None] * jnp.einsum(
            "eia,ei->ea", bmat, sp
        )
        rhs = (
            load * (upper_unit + lower_unit)
            + scatter(dofs, plastic_e, ndof)
        )

        def operator(x):
            return free * kmv(free * x, weights) + (1.0 - free) * x

        rhs = free * rhs
        diagonal = free * scatter(
            dofs, weights[:, None] * jnp.diagonal(ke0, axis1=1, axis2=2), ndof
        ) + (1.0 - free)
        solution, _ = cg(
            operator,
            rhs,
            x0=free * x0,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, tiny),
        )
        residual = operator(solution) - rhs
        relres = jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + tiny)
        return free * solution, relres

    def yield_stress(eq):
        return SIGMA_Y0 + Q_INF * (1.0 - jnp.exp(-VOCE_B * eq))

    def dyield(eq):
        return Q_INF * VOCE_B * jnp.exp(-VOCE_B * eq)

    @jax.jit
    def update_plastic(displacement, temperature, damage, ep_n, eq_n, alpha_n):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        eth = ALPHA_TH * (jnp.mean(temperature[tri], axis=1) - T0)
        ee11 = strain[:, 0] - ep_n[:, 0] - eth
        ee22 = strain[:, 1] - ep_n[:, 1] - eth
        ee33, ee12 = -ep_n[:, 2] - eth, 0.5 * strain[:, 2] - ep_n[:, 3]
        tr = ee11 + ee22 + ee33
        stress = jnp.stack(
            [
                2.0 * G * ee11 + LAMBDA * tr,
                2.0 * G * ee22 + LAMBDA * tr,
                2.0 * G * ee33 + LAMBDA * tr,
                2.0 * G * ee12,
            ],
            axis=1,
        )
        pressure = jnp.mean(stress[:, :3], axis=1)
        dev = stress.at[:, :3].add(-pressure[:, None])
        xi = dev - alpha_n
        xi_eq = jnp.sqrt(
            jnp.maximum(
                0.0,
                1.5
                * (
                    xi[:, 0] ** 2
                    + xi[:, 1] ** 2
                    + xi[:, 2] ** 2
                    + 2.0 * xi[:, 3] ** 2
                ),
            )
        )
        normal = 1.5 * xi / jnp.maximum(xi_eq[:, None], 1.0e-30)
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        damage_q1 = (4.0 * dv[:, 0] + dv[:, 1] + dv[:, 2]) / 6.0
        # The canonical return map samples the first degree-2 quadrature point.
        g = (1.0 - damage_q1) ** 2 + ETA
        active = (g * xi_eq - yield_stress(eq_n) > 0.0) & (xi_eq >= 1.0e-12)
        normal = jnp.where(active[:, None], normal, 0.0)
        ftrial = jnp.maximum(g * xi_eq - yield_stress(eq_n), 0.0)
        delta0 = ftrial / jnp.maximum(3.0 * g * G + CHABOCHE_C + dyield(eq_n), 1e-12)
        kin = (2.0 / 3.0) * CHABOCHE_C

        def consistency(delta):
            den = 1.0 + CHABOCHE_GAMMA * delta
            a = (alpha_n + kin * normal * delta[:, None]) / den[:, None]
            s = dev - 2.0 * G * delta[:, None] * normal
            xil = s - a
            xeq = jnp.sqrt(
                jnp.maximum(
                    1.0e-36,
                    1.5
                    * (
                        xil[:, 0] ** 2
                        + xil[:, 1] ** 2
                        + xil[:, 2] ** 2
                        + 2.0 * xil[:, 3] ** 2
                    ),
                )
            )
            da = (
                kin * normal - CHABOCHE_GAMMA * alpha_n
            ) / den[:, None] ** 2
            dxi = -2.0 * G * normal - da
            dot = (
                xil[:, 0] * dxi[:, 0]
                + xil[:, 1] * dxi[:, 1]
                + xil[:, 2] * dxi[:, 2]
                + 2.0 * xil[:, 3] * dxi[:, 3]
            )
            value = g * xeq - yield_stress(eq_n + delta)
            deriv = g * 1.5 * dot / xeq - dyield(eq_n + delta)
            return value, deriv

        hi0 = jnp.maximum(2.0 * delta0, 1.0e-12)

        def expand(_, hi):
            value, _ = consistency(hi)
            return jnp.where((value > 0.0) & active, 2.0 * hi, hi)

        hi = lax.fori_loop(0, 30, expand, hi0)
        ghi, _ = consistency(hi)
        bracket = ghi <= 0.0
        tol = 1.0e-10 * (xi_eq + SIGMA_Y0)

        def newton(_, state):
            delta, lo, upper, done = state
            value, deriv = consistency(delta)
            now_done = done | (jnp.abs(value) < tol) | (~active)
            lo2 = jnp.where(bracket & (value > 0.0), delta, lo)
            hi2 = jnp.where(bracket & (value <= 0.0), delta, upper)
            raw = jnp.where(
                jnp.abs(deriv) < 1.0e-30, 0.5 * (lo2 + hi2), delta - value / deriv
            )
            bracketed = jnp.clip(raw, lo2 + 1.0e-14, hi2 - 1.0e-14)
            unbracketed = jnp.maximum(
                jnp.where(jnp.abs(deriv) < 1.0e-30, 1.5 * delta, raw), 1.0e-14
            )
            candidate = jnp.where(bracket, bracketed, unbracketed)
            return jnp.where(now_done, delta, candidate), lo2, hi2, now_done

        delta, _, _, converged = lax.fori_loop(
            0,
            200,
            newton,
            (delta0, jnp.zeros_like(delta0), hi, ~active),
        )
        delta = jnp.where(active, delta, 0.0)
        den = 1.0 + CHABOCHE_GAMMA * delta
        ep_new = ep_n + delta[:, None] * normal
        alpha_new = (
            alpha_n + kin * normal * delta[:, None]
        ) / den[:, None]
        return strain, ep_new, eq_n + delta, alpha_new, jnp.all(converged)

    @jax.jit
    def energies(displacement, temperature, damage, ep, hi_old, hii_old, psi_d_old, alphabar):
        strain = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        tv = temperature[tri]
        qtemperature = jnp.stack(
            [
                (4.0 * tv[:, 0] + tv[:, 1] + tv[:, 2]) / 6.0,
                (tv[:, 0] + 4.0 * tv[:, 1] + tv[:, 2]) / 6.0,
                (tv[:, 0] + tv[:, 1] + 4.0 * tv[:, 2]) / 6.0,
            ],
            axis=1,
        )
        eth = ALPHA_TH * (qtemperature - T0)
        e11 = strain[:, 0, None] - ep[:, 0, None] - eth
        e22 = strain[:, 1, None] - ep[:, 1, None] - eth
        e12 = 0.5 * strain[:, 2, None] - ep[:, 3, None]
        avg = 0.5 * (e11 + e22)
        radius = jnp.sqrt(jnp.maximum(0.0, (0.5 * (e11 - e22)) ** 2 + e12**2))
        p1, p2 = avg + radius, avg - radius
        p3 = -ep[:, 2, None] - eth
        trace_pos = jnp.maximum(p1, 0.0) + jnp.maximum(p2, 0.0) + jnp.maximum(p3, 0.0)
        psi_i_q = 0.5 * LAMBDA * trace_pos**2
        psi_ii_q = G * (
            jnp.maximum(p1, 0.0) ** 2 + jnp.maximum(p2, 0.0) ** 2
            + jnp.maximum(p3, 0.0) ** 2
        )
        # Gridap order-2 triangle quadrature: permutations of (2/3,1/6,1/6).
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        qdamage = jnp.stack(
            [
                (4.0 * dv[:, 0] + dv[:, 1] + dv[:, 2]) / 6.0,
                (dv[:, 0] + 4.0 * dv[:, 1] + dv[:, 2]) / 6.0,
                (dv[:, 0] + dv[:, 1] + 4.0 * dv[:, 2]) / 6.0,
            ],
            axis=1,
        )
        psi_i = jnp.max(psi_i_q, axis=1)
        psi_ii = jnp.max(psi_ii_q, axis=1)
        psi_d = jnp.max(
            ((1.0 - qdamage) ** 2 + ETA) * (psi_i_q + psi_ii_q), axis=1
        )
        alphabar_new = alphabar + jnp.maximum(0.0, psi_d - psi_d_old)
        return (
            jnp.maximum(hi_old, psi_i),
            jnp.maximum(hii_old, psi_ii),
            psi_d,
            alphabar_new,
        )

    def phase_matvec(x, element_matrix):
        return scatter(
            tri, jnp.einsum("eij,ej->ei", element_matrix, x[tri]), nnode
        )

    @jax.jit
    def solve_phase(damage_old, temperature, hi, hii, eq, alphabar):
        tcell = jnp.mean(temperature[tri], axis=1)
        xi_t = jnp.clip((tcell - 300.0) / 1000.0, 0.0, 1.0)
        soft = jnp.maximum(1.0 - 1.80 * xi_t + 1.10 * xi_t**2, 0.05)
        gc = GC0 * soft
        alpha_t = ALPHA_T0 * soft
        psi_p = (1.0 - CHI_EFF) * (SIGMA_Y0 * eq + Q_INF * (
            eq + (jnp.exp(-VOCE_B * eq) - 1.0) / VOCE_B
        ))
        fatigue = (alpha_t / (alphabar + alpha_t + 1.0e-12)) ** 2
        drive = (hi + hii + psi_p) / gc
        emat = (
            fatigue[:, None, None] * L0 * grad
            + (fatigue / L0 + 2.0 * drive)[:, None, None] * mass
        )
        rhs = scatter(
            tri,
            (2.0 * drive * areas / 3.0)[:, None] * jnp.ones((1, 3), dtype=dtype),
            nnode,
        )
        diagonal = scatter(
            tri, jnp.diagonal(emat, axis1=1, axis2=2), nnode
        )
        operator = lambda x: phase_matvec(x, emat)
        solved, _ = cg(
            operator,
            rhs,
            x0=damage_old,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, tiny),
        )
        residual = operator(solved) - rhs
        relres = jnp.linalg.norm(residual) / (jnp.linalg.norm(rhs) + tiny)
        damage = jnp.clip(jnp.maximum(solved, damage_old), 0.0, 1.0)
        return damage, relres

    return solve_heat, solve_displacement, update_plastic, energies, solve_phase


# Outputs ---------------------------------------------------------------------
CSV_FIELDS = (
    "cycle",
    "time_s",
    "block_time_s",
    "a_mm",
    "delta_a_mm",
    "da_dN_mm_per_cycle",
    "tip_x_mm",
    "tip_y_mm",
    "T_right_K",
    "T_mean_K",
    "T_max_K",
    "Kmax_MPa_sqrt_mm",
    "Kmin_MPa_sqrt_mm",
    "deltaK_MPa_sqrt_mm",
    "s_max",
    "max_epspeq",
    "alphabar_max",
    "fatigue_f",
    "u_linear_residual",
    "s_linear_residual",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def crack_tip(nodes: np.ndarray, damage: np.ndarray) -> tuple[float, float]:
    mask = (
        (nodes[:, 0] > V_TIP_X - L0)
        & (np.abs(nodes[:, 1] - Y_CENTER) < 3.0 * L0)
        & (damage > 0.5)
    )
    if not np.any(mask):
        return V_TIP_X, V_TIP_Y
    candidates = np.where(mask)[0]
    index = candidates[np.argmax(nodes[candidates, 0])]
    return float(nodes[index, 0]), float(nodes[index, 1])


def save_checkpoint(path: Path, cycle: int, t_prev: float, state: dict, rows: list[dict]):
    host = {key: np.asarray(jax.device_get(value)) for key, value in state.items()}
    np.savez_compressed(
        path,
        cycle=np.asarray(cycle),
        t_prev=np.asarray(t_prev),
        rows_json=np.asarray(json.dumps(rows)),
        **host,
    )


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    keys = (
        "u", "damage", "temperature", "ep", "eq", "alpha", "hi", "hii",
        "psi_d", "alphabar", "plastic_heat"
    )
    state = {key: np.asarray(data[key]) for key in keys}
    return int(data["cycle"]), float(data["t_prev"]), state, json.loads(str(data["rows_json"]))


def save_vtu(path: Path, nodes: np.ndarray, tri: np.ndarray, state: dict) -> None:
    import meshio

    u = np.asarray(jax.device_get(state["u"]))
    damage = np.asarray(jax.device_get(state["damage"]))
    cell = {
        key: [np.asarray(jax.device_get(state[key]))]
        for key in ("hi", "hii", "alphabar", "eq")
    }
    cell["epsilon_p"] = [np.asarray(jax.device_get(state["ep"]))]
    cell["backstress"] = [np.asarray(jax.device_get(state["alpha"]))]
    meshio.write_points_cells(
        str(path),
        np.column_stack([nodes, np.zeros(len(nodes))]),
        [("triangle", tri)],
        point_data={
            "s": damage,
            "T": np.asarray(jax.device_get(state["temperature"])),
            "u": np.column_stack([u[0::2], u[1::2], np.zeros(len(nodes))]),
        },
        cell_data=cell,
    )


def save_scientific_snapshot(
    path: Path,
    nodes: np.ndarray,
    tri: np.ndarray,
    state: dict,
    metadata: dict,
) -> bool:
    if path.is_file() and metadata.get("milestone") != "final":
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    arrays = {
        key: np.asarray(jax.device_get(value))
        for key, value in state.items()
    }
    arrays["u"] = arrays["u"].reshape(-1, 2)
    np.savez_compressed(
        temporary,
        mesh=nodes,
        nodes=nodes,
        triangles=tri,
        u=arrays.pop("u"),
        d=arrays.pop("damage"),
        T=arrays.pop("temperature"),
        plastic_strain=arrays.pop("ep"),
        backstress=arrays.pop("alpha"),
        isotropic=arrays["eq"],
        fatigue=arrays["alphabar"],
        history_I=arrays["hi"],
        history_II=arrays["hii"],
        **arrays,
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    temporary.replace(path)
    return True


def load_reference(path: Path):
    if not path.is_file():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        name: np.atleast_1d(np.asarray(data[name], dtype=np.float64))
        for name in data.dtype.names
    }


def compare_reference(rows: list[dict], reference):
    if not rows or reference is None:
        return None
    by_cycle = {int(v): i for i, v in enumerate(reference["cycle"])}
    pairs = [(row, by_cycle[int(row["cycle"])]) for row in rows if int(row["cycle"]) in by_cycle]
    if not pairs:
        return None
    result = {}
    for ours, refname in (
        ("delta_a_mm", "delta_a_mm"),
        ("s_max", "max_damage"),
        ("max_epspeq", "max_epspeq"),
        ("T_mean_K", "T_mean_K"),
    ):
        if refname not in reference:
            continue
        a = np.asarray([row[ours] for row, _ in pairs])
        b = np.asarray([reference[refname][index] for _, index in pairs])
        result[f"{ours}_relative_l2"] = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
        result[f"{ours}_max_abs"] = float(np.max(np.abs(a - b)))
    result["matched_cycles"] = len(pairs)
    return result


def save_plot(outdir: Path, rows: list[dict], reference) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot([r["cycle"] for r in rows], [r["delta_a_mm"] for r in rows], label="JAX")
    if reference is not None:
        ax.plot(reference["cycle"], reference["delta_a_mm"], "--", label="Julia/Gridap")
    ax.set(xlabel="N (cycles)", ylabel="Crack extension Δa (mm)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "crack_growth_comparison.png", dpi=180)
    plt.close(fig)


# Driver ----------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    if args.steps_per_cycle <= 0 or args.steps_per_cycle % 2:
        raise ValueError("--steps-per-cycle must be a positive even integer")
    if min(args.cycles_total, args.max_stagger, args.max_picard, args.stall_cycles) <= 0:
        raise ValueError("cycle and iteration counts must be positive")

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if args.platform == "gpu" and all(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU requested, but JAX devices are {devices}")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("FP64 could not be enabled")
    selected_device_slug = device_slug(devices[0])
    if args.outdir is None:
        root = script_root()
        args.outdir = (
            root / "outputs" / f"results_{selected_device_slug}"
            / "thermo_plastic_fatigue_CT" / "jax"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.mesh is None:
        args.mesh = create_ct_mesh(args.outdir / "CT_mesh.msh")
        print(f"generated canonical CT mesh: {args.mesh}")
    checkpoint = args.outdir / "checkpoint_latest.npz"
    csv_path = args.outdir / "cycle_summary.csv"
    nodes, tri, areas, upper, lower, fixed, rot, left, right = load_mesh(args.mesh)
    bmat, dofs, ke0, mass, grad = preprocess(nodes, tri, areas)
    nnode, nel, ndof = len(nodes), len(tri), 2 * len(nodes)
    free = np.ones(ndof, dtype=np.float64)
    free[2 * fixed] = 0.0
    free[2 * fixed + 1] = 0.0
    free[2 * rot + 1] = 0.0
    thermal_free = np.ones(nnode, dtype=np.float64)
    thermal_free[left] = 0.0
    thermal_free[right] = 0.0
    thermal_qoi_mask = thermal_free.astype(bool)
    thermal_left = np.zeros(nnode, dtype=np.float64)
    thermal_right = np.zeros(nnode, dtype=np.float64)
    thermal_left[left] = 1.0
    thermal_right[right] = 1.0
    upper_unit = boundary_force(nodes, upper, +1.0 / (2.0 * np.pi * PIN_RADIUS))
    lower_unit = boundary_force(nodes, lower, -1.0 / (2.0 * np.pi * PIN_RADIUS))
    kernels = make_kernels(
        tri_np=tri,
        dofs_np=dofs,
        bmat_np=bmat,
        ke0_np=ke0,
        mass_np=mass,
        grad_np=grad,
        areas_np=areas,
        free_np=free,
        thermal_free_np=thermal_free,
        thermal_left_np=thermal_left,
        thermal_right_np=thermal_right,
        upper_unit_np=upper_unit,
        lower_unit_np=lower_unit,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
    )
    solve_t, solve_u, update_plastic, compute_energies, solve_s = kernels
    dtype = jnp.float64

    if args.resume and checkpoint.is_file():
        cycle, t_prev, state_np, rows = load_checkpoint(checkpoint)
        state = {key: jnp.asarray(value, dtype=dtype) for key, value in state_np.items()}
    else:
        cycle, t_prev, rows = 0, F_MIN, []
        state = {
            "u": jnp.zeros(ndof, dtype=dtype),
            "damage": jnp.zeros(nnode, dtype=dtype),
            "temperature": jnp.full(nnode, T0, dtype=dtype),
            "ep": jnp.zeros((nel, 4), dtype=dtype),
            "eq": jnp.zeros(nel, dtype=dtype),
            "alpha": jnp.zeros((nel, 4), dtype=dtype),
            "hi": jnp.zeros(nel, dtype=dtype),
            "hii": jnp.zeros(nel, dtype=dtype),
            "psi_d": jnp.zeros(nel, dtype=dtype),
            "alphabar": jnp.zeros(nel, dtype=dtype),
            "plastic_heat": jnp.zeros(nel, dtype=dtype),
        }
    reference = load_reference(args.reference)
    print("=" * 78)
    print("Standalone JAX CT Chaboche phase-field fatigue")
    print(f"  JAX {jax.__version__}; Python {platform.python_version()}; devices={devices}")
    print(f"  mesh={args.mesh} ({nnode} nodes, {nel} triangles)")
    print(
        f"  cycles={args.cycles_total}, no cycle jump, "
        f"substeps={args.steps_per_cycle + 1}, FP64=True"
    )
    print("=" * 78)

    n_half = args.steps_per_cycle // 2
    load_steps = np.linspace(F_MIN, F_MAX, n_half + 1)
    unload_steps = np.linspace(F_MAX, F_MIN, n_half + 1)[1:]
    run_t0 = time.perf_counter()
    elapsed_offset = float(rows[-1]["time_s"]) if rows else 0.0
    saved_vtu = [False] * len(VTK_FRACTIONS)
    snapshot_dir = args.outdir / "snapshots"
    snapshot_done = [
        (snapshot_dir / f"milestone_da{round(100 * fraction):02d}pct.npz").is_file()
        for fraction in VTK_FRACTIONS
    ]
    snapshot_io_s = 0.0
    blocks_this_run = 0
    last_u_res = last_s_res = 0.0
    last_t_res = 0.0
    solver_time = 0.0
    first_block_time = None
    post_compile_block_times = []
    crack_initiated = any(row["delta_a_mm"] > args.init_da for row in rows)
    stall_count = 0
    if crack_initiated:
        for row in reversed(rows):
            if row["da_dN_mm_per_cycle"] >= args.stall_tol:
                break
            stall_count += 1

    while cycle < args.cycles_total:
        block_t0 = time.perf_counter()
        jump = 1
        right_value = T0 + (T_RIGHT_MAX - T0) * min(
            (cycle + 1) / HEAT_RAMP_CYCLES, 1.0
        )
        sequence = np.concatenate([load_steps, unload_steps])
        if args.smoke:
            sequence = np.asarray([F_MAX])
            jump = 1
        substep_counter = 0
        for load in sequence:
            substep_counter += 1
            dt = 1.0 / len(sequence)
            state["temperature"], t_res = solve_t(
                state["temperature"], state["damage"], state["plastic_heat"],
                right_value, dt
            )
            last_t_res = float(jax.device_get(t_res))
            delta_load = float(load - t_prev)
            nsub = max(1, int(np.ceil(abs(delta_load) / 50.0)))
            eq_outer_old = state["eq"]
            for isub in range(1, nsub + 1):
                current_load = t_prev + delta_load * isub / nsub
                ep_base, eq_base, alpha_base = state["ep"], state["eq"], state["alpha"]
                ep_work, eq_work, alpha_work = ep_base, eq_base, alpha_base
                for _picard in range(1, args.max_picard + 1):
                    u_old = state["u"]
                    u_new, u_res = solve_u(
                        state["damage"], ep_work, state["temperature"],
                        current_load, state["u"]
                    )
                    _, ep_work, eq_work, alpha_work, plastic_ok = update_plastic(
                        u_new, state["temperature"], state["damage"],
                        ep_base, eq_base, alpha_base
                    )
                    relative = jnp.linalg.norm(u_new - u_old) / (
                        jnp.linalg.norm(u_new) + 1.0e-10
                    )
                    state["u"] = u_new
                    if float(jax.device_get(relative)) < args.picard_tol:
                        break
                if not bool(jax.device_get(plastic_ok)):
                    print("warning: at least one Chaboche return map reached iteration cap")
                state["ep"], state["eq"], state["alpha"] = ep_work, eq_work, alpha_work
                last_u_res = float(jax.device_get(u_res))
            sigma_old = SIGMA_Y0 + Q_INF * (1.0 - jnp.exp(-VOCE_B * eq_outer_old))
            sigma_new = SIGMA_Y0 + Q_INF * (1.0 - jnp.exp(-VOCE_B * state["eq"]))
            state["plastic_heat"] = (
                CHI_EFF * 0.5 * (sigma_old + sigma_new)
                * jnp.maximum(state["eq"] - eq_outer_old, 0.0) / dt
            )
            hi, hii, psi_d, alphabar = compute_energies(
                state["u"],
                state["temperature"],
                state["damage"],
                state["ep"],
                state["hi"],
                state["hii"],
                state["psi_d"],
                state["alphabar"],
            )
            state.update(hi=hi, hii=hii, psi_d=psi_d, alphabar=alphabar)
            if args.phase_every > 0 and substep_counter % args.phase_every == 0:
                for _stagger in range(1, args.max_stagger + 1):
                    old = state["damage"]
                    new, s_res = solve_s(
                        old, state["temperature"], hi, hii, state["eq"], alphabar
                    )
                    error = jnp.linalg.norm(new - old) / (jnp.linalg.norm(new) + 1.0e-10)
                    state["damage"] = new
                    if float(jax.device_get(error)) < args.stagger_tol:
                        break
                last_s_res = float(jax.device_get(s_res))
            t_prev = float(load)

        cycle += jump
        blocks_this_run += 1
        jax.block_until_ready(state["damage"])
        damage_host = np.asarray(jax.device_get(state["damage"]))
        temperature_host = np.asarray(jax.device_get(state["temperature"]))
        tip_x, tip_y = crack_tip(nodes, damage_host)
        alpha_max = float(jax.device_get(jnp.max(state["eq"])))
        alphabar_max = float(jax.device_get(jnp.max(state["alphabar"])))
        smax = float(np.max(damage_host))
        # Gridap's reference CSV reports free thermal DOFs (Dirichlet nodes excluded).
        t_mean = float(np.mean(temperature_host[thermal_qoi_mask]))
        t_max = float(np.max(temperature_host[thermal_qoi_mask]))
        soft = max(1.0 - 1.8 * max(t_mean - 300.0, 0.0) / 1000.0
                   + 1.1 * (max(t_mean - 300.0, 0.0) / 1000.0) ** 2, 0.05)
        alpha_t = ALPHA_T0 * soft
        fatigue = (alpha_t / (alphabar_max + alpha_t + 1.0e-12)) ** 2
        xratio = tip_x / W_CT
        geom = ((2.0 + xratio) / (1.0 - xratio) ** 1.5) * (
            0.886 + 4.64 * xratio - 13.32 * xratio**2
            + 14.72 * xratio**3 - 5.6 * xratio**4
        )
        kmax = F_MAX_3D / (THICKNESS * np.sqrt(W_CT)) * geom
        previous_a = rows[-1]["a_mm"] if rows else V_TIP_X
        block_time = time.perf_counter() - block_t0
        solver_time += block_time
        if first_block_time is None:
            first_block_time = block_time
        else:
            post_compile_block_times.append(block_time)
        elapsed = elapsed_offset + time.perf_counter() - run_t0 - snapshot_io_s
        row = {
            "cycle": cycle,
            "time_s": elapsed,
            "block_time_s": block_time,
            "a_mm": tip_x,
            "delta_a_mm": tip_x - V_TIP_X,
            "da_dN_mm_per_cycle": tip_x - previous_a,
            "tip_x_mm": tip_x,
            "tip_y_mm": tip_y,
            "T_right_K": right_value,
            "T_mean_K": t_mean,
            "T_max_K": t_max,
            "Kmax_MPa_sqrt_mm": kmax,
            "Kmin_MPa_sqrt_mm": 0.0,
            "deltaK_MPa_sqrt_mm": kmax,
            "s_max": smax,
            "max_epspeq": alpha_max,
            "alphabar_max": alphabar_max,
            "fatigue_f": fatigue,
            "u_linear_residual": last_u_res,
            "s_linear_residual": last_s_res,
        }
        rows.append(row)
        extension = tip_x - V_TIP_X
        crack_initiated = crack_initiated or extension > args.init_da
        if crack_initiated:
            stall_count = (
                stall_count + 1
                if row["da_dN_mm_per_cycle"] < args.stall_tol
                else 0
            )
        if blocks_this_run % args.print_every == 0 or args.smoke:
            print(
                f"  N={cycle:5d} da={row['delta_a_mm']:.4f} "
                f"Tmean={t_mean:.1f} smax={smax:.5f} ep={alpha_max:.5f} "
                f"abar={alphabar_max:.1f} "
                f"f={fatigue:.5f} dt={block_time:.2f}s"
            )
        if max(last_t_res, last_u_res, last_s_res) > args.linear_residual_tol:
            raise RuntimeError(
                f"linear residual exceeded tolerance: T={last_t_res:.3e}, "
                f"u={last_u_res:.3e}, s={last_s_res:.3e}"
            )
        if not args.no_vtk and cycle in THERMAL_SNAPSHOT_CYCLES:
            save_vtu(
                args.outdir / f"thermal_cycle{cycle:05d}.vtu",
                nodes,
                tri,
                state,
            )
        if args.save_every > 0 and blocks_this_run % args.save_every == 0:
            write_csv(csv_path, rows)
            save_checkpoint(checkpoint, cycle, t_prev, state, rows)
        for index, fraction in enumerate(VTK_FRACTIONS):
            if (
                not args.no_snapshots
                and not snapshot_done[index]
                and extension >= fraction * (args.stop_crack_x - V_TIP_X)
            ):
                snapshot_t0 = time.perf_counter()
                save_scientific_snapshot(
                    snapshot_dir / f"milestone_da{round(100 * fraction):02d}pct.npz",
                    nodes,
                    tri,
                    state,
                    {
                        "cycle": cycle,
                        "step": cycle,
                        "load": t_prev,
                        "qoi_crack_extension_mm": extension,
                        "qoi_crack_tip_x_mm": tip_x,
                        "qoi_crack_tip_y_mm": tip_y,
                        "stop_extension_mm": args.stop_crack_x - V_TIP_X,
                        "milestone": f"{round(100 * fraction)}pct_stop_extension",
                    },
                )
                snapshot_io_s += time.perf_counter() - snapshot_t0
                snapshot_done[index] = True
        for index, fraction in enumerate(VTK_FRACTIONS):
            if (
                not args.no_vtk
                and not saved_vtu[index]
                and extension >= fraction * (args.stop_crack_x - V_TIP_X)
            ):
                save_vtu(
                    args.outdir / f"result_da{round(100*fraction):02d}_cycle{cycle:05d}.vtu",
                    nodes,
                    tri,
                    state,
                )
                saved_vtu[index] = True
        if (
            args.smoke
            or tip_x > args.stop_crack_x
            or (crack_initiated and stall_count >= args.stall_cycles)
        ):
            if crack_initiated and stall_count >= args.stall_cycles:
                print(
                    f"  crack growth stalled for {args.stall_cycles} cycles "
                    f"(da={extension:.4f} mm), stop"
                )
            break

    total = time.perf_counter() - run_t0 - snapshot_io_s
    write_csv(csv_path, rows)
    save_checkpoint(checkpoint, cycle, t_prev, state, rows)
    if not args.no_snapshots:
        snapshot_t0 = time.perf_counter()
        final_row = rows[-1]
        save_scientific_snapshot(
            snapshot_dir / "final.npz",
            nodes,
            tri,
            state,
            {
                "cycle": cycle,
                "step": cycle,
                "load": t_prev,
                "qoi_crack_extension_mm": float(final_row["delta_a_mm"]),
                "qoi_crack_tip_x_mm": float(final_row["tip_x_mm"]),
                "qoi_crack_tip_y_mm": float(final_row["tip_y_mm"]),
                "stop_extension_mm": args.stop_crack_x - V_TIP_X,
                "milestone": "final",
            },
        )
        snapshot_io_s += time.perf_counter() - snapshot_t0
    if not args.no_vtk:
        save_vtu(args.outdir / "result_final.vtu", nodes, tri, state)
    comparison = compare_reference(rows, reference)
    final_qois = (
        {
            key: rows[-1][key]
            for key in (
                "cycle",
                "delta_a_mm",
                "tip_x_mm",
                "tip_y_mm",
                "s_max",
                "max_epspeq",
                "alphabar_max",
                "fatigue_f",
            )
        }
        if rows
        else None
    )
    timing = {
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "dtype": "float64",
        "mesh": str(args.mesh.resolve()),
        "reference": str(args.reference.resolve()),
        "nodes": nnode,
        "triangles": nel,
        "device_strings": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "device_slug": selected_device_slug,
        "blocks_this_run": blocks_this_run,
        "total_s": total,
        "solver_s_excluding_output": solver_time,
        "first_block_compile_and_solve_s": first_block_time,
        "post_compile_block_mean_s": (
            float(np.mean(post_compile_block_times))
            if post_compile_block_times
            else None
        ),
        "visualization_in_solver_timing": False,
        "final_qois": final_qois,
        "comparison": comparison,
        "canonical_revision_defaults": not args.smoke
        and args.cycles_total == DEFAULT_CYCLES
        and args.steps_per_cycle == DEFAULT_STEPS_PER_CYCLE,
    }
    with (args.outdir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    with (args.outdir / "qois.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "case": "thermo_plastic_fatigue_CT_dT200K",
                "final": final_qois,
                "reference_comparison": comparison,
            },
            stream,
            indent=2,
        )
    if not args.no_visualization:
        save_plot(args.outdir, rows, reference)
    print("-" * 78)
    print(f"completed through N={cycle} in {total:.3f}s; outputs={args.outdir}")
    if comparison:
        print(
            f"reference matched cycles={comparison['matched_cycles']}; "
            f"da L2={comparison['delta_a_mm_relative_l2']:.3e}; "
            f"smax L2={comparison['s_max_relative_l2']:.3e}"
        )


if __name__ == "__main__":
    main()
