#!/usr/bin/env python3
"""Standalone FP64 JAX/GPU port of ``plastic_fatigue_SENT.jl``.

The default case is the unchanged Julia l0=0.02 SENT calculation.  When no
mesh is supplied, this file generates the canonical Julia SENT mesh in the
device-specific output directory.  There are no project-local imports or
required repository data files; NumPy, JAX, meshio and gmsh are the runtime
packages.

Default run:
  python plastic_fatigue_SENT_jax_gpu.py --platform gpu
Small CPU smoke test:
  python plastic_fatigue_SENT_jax_gpu.py --platform cpu --cycles 1 \
      --steps-per-cycle 2 --no-vtk
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
    from jax.scipy.sparse.linalg import cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "JAX is required (for NVIDIA, install a CUDA JAX wheel) together with meshio"
    ) from exc


# Exact constants in plastic_fatigue_SENT.jl.
W = 1.0
HEIGHT = 1.0
A0 = 0.5
E = 210_000.0
NU = 0.3
G = E / (2.0 * (1.0 + NU))
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
SIGMA_Y0 = 440.0
HARDENING_N = 0.1
L0 = 0.02
GC = 2.7
ALPHA_T = 11.25
ETA_STAB = 1.0e-6
ETA_PLASTIC = 1.0
U_MAX = 0.001
R_LOAD = 0.0
U_MIN = R_LOAD * U_MAX
DEFAULT_CYCLES = 800
DEFAULT_STEPS_PER_CYCLE = 20
DEFAULT_PHASE_EVERY = 1
DEFAULT_MAX_STAGGER = 20
DEFAULT_STAGGER_TOL = 1.0e-6
TIP_X_BACK = 3.0 * L0
TIP_X_FRONT = L0
TIP_Y_HALF = 3.0 * L0
WAKE_X_BACK = 10.0 * L0
AHEAD_X_FRONT = 5.0 * L0
AHEAD_ALPHA_FRONT = 2.0 * L0
EPSP_ACTIVE_THRESH = 1.0e-4
EPSP_ZONE_THRESH = 1.0e-2
VTK_A_LEVELS = (0.02, 0.10, 0.20, 0.30, 0.40)
H_FINE = 0.005
H_COARSE = 0.05


def script_root() -> Path:
    """Use the repository root in-tree, or the script directory standalone."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code" else script_dir


def build_parser() -> argparse.ArgumentParser:
    root = script_root()
    refdir = root / "data" / "julia_reference" / "plastic_fatigue_sent"
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU elastoplastic fatigue phase-field SENT solver"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="existing mesh to preserve and use (default: generate under outdir)",
    )
    parser.add_argument("--reference-dir", type=Path, default=refdir)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: outputs/results_<device>/plastic_fatigue_SENT/jax)",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument(
        "--steps-per-cycle", type=int, default=DEFAULT_STEPS_PER_CYCLE
    )
    parser.add_argument("--phase-every", type=int, default=DEFAULT_PHASE_EVERY)
    parser.add_argument("--max-stagger", type=int, default=DEFAULT_MAX_STAGGER)
    parser.add_argument("--stagger-tol", type=float, default=DEFAULT_STAGGER_TOL)
    parser.add_argument("--picard-maxiter", type=int, default=10)
    parser.add_argument("--picard-tol", type=float, default=1.0e-4)
    parser.add_argument("--cg-tol", type=float, default=1.0e-10)
    parser.add_argument("--cg-maxiter", type=int, default=4000)
    parser.add_argument("--linear-residual-tol", type=float, default=1.0e-6)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="generate/load and validate the mesh, then exit before solving",
    )
    parser.add_argument(
        "--stop-after-substeps",
        type=int,
        default=0,
        help="testing aid; zero means complete every requested cycle",
    )
    return parser


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def generate_mesh(path: Path) -> None:
    """Generate the exact canonical mesh recipe from the Julia CPU solver."""
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit(
            "gmsh is required to generate the default mesh: "
            "python -m pip install gmsh"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    try:
        gmsh.initialize()
        initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("SENT")

        y_crack = HEIGHT / 2.0
        crack_height = H_FINE

        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, H_COARSE, 1)
        gmsh.model.geo.addPoint(W, 0.0, 0.0, H_COARSE, 2)
        gmsh.model.geo.addPoint(W, HEIGHT / 2.0, 0.0, H_COARSE, 22)
        gmsh.model.geo.addPoint(W, HEIGHT, 0.0, H_COARSE, 3)
        gmsh.model.geo.addPoint(0.0, HEIGHT, 0.0, H_COARSE, 4)
        gmsh.model.geo.addPoint(
            0.0, y_crack + 0.5 * crack_height, 0.0, H_COARSE, 5
        )
        gmsh.model.geo.addPoint(
            A0 - crack_height,
            y_crack + 0.5 * crack_height,
            0.0,
            H_FINE,
            6,
        )
        gmsh.model.geo.addPoint(A0, y_crack, 0.0, H_FINE, 7)
        gmsh.model.geo.addPoint(
            A0 - crack_height,
            y_crack - 0.5 * crack_height,
            0.0,
            H_FINE,
            8,
        )
        gmsh.model.geo.addPoint(
            0.0, y_crack - 0.5 * crack_height, 0.0, H_COARSE, 9
        )

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 22, 11)
        gmsh.model.geo.addLine(22, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 7, 6)
        gmsh.model.geo.addLine(7, 8, 7)
        gmsh.model.geo.addLine(8, 9, 8)
        gmsh.model.geo.addLine(9, 1, 9)

        gmsh.model.geo.addCurveLoop([1, 11, 2, 3, 4, 5, 6, 7, 8, 9], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        gmsh.model.addPhysicalGroup(1, [1], 1)
        gmsh.model.setPhysicalName(1, 1, "bottom")
        gmsh.model.addPhysicalGroup(1, [3], 2)
        gmsh.model.setPhysicalName(1, 2, "top")

        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", H_FINE)
        gmsh.model.mesh.field.setNumber(1, "VOut", H_COARSE)
        gmsh.model.mesh.field.setNumber(1, "XMin", A0 - 0.1)
        gmsh.model.mesh.field.setNumber(1, "XMax", W)
        gmsh.model.mesh.field.setNumber(1, "YMin", y_crack - 0.15)
        gmsh.model.mesh.field.setNumber(1, "YMax", y_crack + 0.15)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
        node_count = len(gmsh.model.mesh.getNodes()[0])
    finally:
        if initialized:
            gmsh.finalize()

    print(
        f"generated canonical Julia SENT mesh: {path} "
        f"({node_count} nodes; h_fine={H_FINE}, h_coarse={H_COARSE})"
    )


def load_mesh(path: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc
    if not path.is_file():
        raise FileNotFoundError(f"mesh not found: {path}")
    mesh = meshio.read(str(path))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    blocks = [np.asarray(c.data, dtype=np.int64) for c in mesh.cells if c.type == "triangle"]
    if not blocks:
        raise RuntimeError("mesh contains no linear triangles")
    tri = np.concatenate(blocks)
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
        raise RuntimeError("mesh contains degenerate triangles")

    bottom: set[int] = set()
    top: set[int] = set()
    physical = mesh.cell_data.get("gmsh:physical")
    if physical is not None:
        for block_index, block in enumerate(mesh.cells):
            if block.type != "line":
                continue
            for edge, tag in zip(block.data, physical[block_index]):
                if int(tag) == 1:
                    bottom.update(map(int, edge))
                elif int(tag) == 2:
                    top.update(map(int, edge))
    if not bottom:
        bottom.update(np.flatnonzero(np.abs(nodes[:, 1]) < 1.0e-10).tolist())
    if not top:
        top.update(np.flatnonzero(np.abs(nodes[:, 1] - HEIGHT) < 1.0e-10).tolist())
    if not bottom or not top:
        raise RuntimeError("could not identify Julia bottom/top physical boundaries")
    return nodes, tri, areas, np.asarray(sorted(bottom)), np.asarray(sorted(top))


def preprocess(nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray):
    x, y = nodes[:, 0], nodes[:, 1]
    i0, i1, i2 = tri.T
    gx = np.stack((y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]), axis=1)
    gy = np.stack((x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]), axis=1)
    gx /= 2.0 * areas[:, None]
    gy /= 2.0 * areas[:, None]
    nel = len(tri)
    bmat = np.zeros((nel, 3, 6))
    for a in range(3):
        bmat[:, 0, 2 * a] = gx[:, a]
        bmat[:, 1, 2 * a + 1] = gy[:, a]
        bmat[:, 2, 2 * a] = gy[:, a]
        bmat[:, 2, 2 * a + 1] = gx[:, a]
    cmat = np.array(
        [
            [LAMBDA + 2.0 * G, LAMBDA, 0.0],
            [LAMBDA, LAMBDA + 2.0 * G, 0.0],
            [0.0, 0.0, G],
        ]
    )
    ke0 = areas[:, None, None] * np.einsum(
        "eia,ij,ejb->eab", bmat, cmat, bmat, optimize=True
    )
    mass = areas[:, None, None] * np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    )[None, :, :] / 12.0
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    dofs = np.empty((nel, 6), dtype=np.int64)
    for a in range(3):
        dofs[:, 2 * a] = 2 * tri[:, a]
        dofs[:, 2 * a + 1] = 2 * tri[:, a] + 1
    centers = np.mean(nodes[tri], axis=1)
    return gx, gy, bmat, ke0, mass, grad, dofs, centers


def make_kernels(
    tri_np,
    dofs_np,
    bmat_np,
    ke0_np,
    mass_np,
    grad_np,
    areas_np,
    free_np,
    cg_tol,
    cg_maxiter,
):
    dtype = jnp.float64
    tri = jnp.asarray(tri_np)
    dofs = jnp.asarray(dofs_np)
    bmat = jnp.asarray(bmat_np, dtype=dtype)
    ke0 = jnp.asarray(ke0_np, dtype=dtype)
    mass = jnp.asarray(mass_np, dtype=dtype)
    grad = jnp.asarray(grad_np, dtype=dtype)
    areas = jnp.asarray(areas_np, dtype=dtype)
    free = jnp.asarray(free_np, dtype=dtype)
    ndof = len(free_np)
    nnode = ndof // 2

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(values.reshape(-1))

    def g_integral_mean(damage):
        # Exact cell mean of (1-d)^2+eta for a P1 damage field.
        dv = jnp.clip(damage[tri], 0.0, 1.0)
        mean_d = jnp.mean(dv, axis=1)
        mean_d2 = (
            jnp.sum(dv * dv, axis=1)
            + dv[:, 0] * dv[:, 1]
            + dv[:, 1] * dv[:, 2]
            + dv[:, 2] * dv[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * mean_d + mean_d2 + ETA_STAB

    def kmv(x, weights):
        local = weights[:, None] * jnp.einsum("eij,ej->ei", ke0, x[dofs])
        return scatter(dofs, local, ndof)

    @jax.jit
    def solve_u(damage, boundary, plastic, plastic33, x0):
        weights = g_integral_mean(damage)
        ep_engineering = jnp.stack(
            (plastic[:, 0], plastic[:, 1], 2.0 * plastic[:, 2]), axis=1
        )
        sigma_p = jnp.stack(
            (
                (LAMBDA + 2.0 * G) * ep_engineering[:, 0]
                + LAMBDA * ep_engineering[:, 1]
                + LAMBDA * plastic33,
                LAMBDA * ep_engineering[:, 0]
                + (LAMBDA + 2.0 * G) * ep_engineering[:, 1]
                + LAMBDA * plastic33,
                G * ep_engineering[:, 2],
            ),
            axis=1,
        )
        rhs_local = (
            weights[:, None]
            * areas[:, None]
            * jnp.einsum("eia,ei->ea", bmat, sigma_p)
        )
        rhs_plastic = scatter(dofs, rhs_local, ndof)
        rhs = free * (rhs_plastic - kmv(boundary, weights))

        def operator(x):
            return free * kmv(free * x, weights) + (1.0 - free) * x

        diag_local = weights[:, None] * jnp.diagonal(ke0, axis1=1, axis2=2)
        diagonal = free * scatter(dofs, diag_local, ndof) + (1.0 - free)
        correction, _ = cg(
            operator,
            rhs,
            x0=free * (x0 - boundary),
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, 1.0e-30),
        )
        result = boundary + free * correction
        residual = jnp.linalg.norm(operator(correction) - rhs) / (
            jnp.linalg.norm(rhs) + 1.0e-30
        )
        return result, residual

    @jax.jit
    def strain(displacement):
        eng = jnp.einsum("eij,ej->ei", bmat, displacement[dofs])
        return jnp.stack((eng[:, 0], eng[:, 1], 0.5 * eng[:, 2]), axis=1)

    @jax.jit
    def return_map(total_strain, ep0, ep330, eq0):
        ee11 = total_strain[:, 0] - ep0[:, 0]
        ee22 = total_strain[:, 1] - ep0[:, 1]
        ee12 = total_strain[:, 2] - ep0[:, 2]
        ee33 = -ep330
        trace = ee11 + ee22 + ee33
        s11_full = 2.0 * G * ee11 + LAMBDA * trace
        s22_full = 2.0 * G * ee22 + LAMBDA * trace
        s33_full = 2.0 * G * ee33 + LAMBDA * trace
        s12_full = 2.0 * G * ee12
        pressure = (s11_full + s22_full + s33_full) / 3.0
        dev11 = s11_full - pressure
        dev22 = s22_full - pressure
        dev33 = s33_full - pressure
        dev12 = s12_full
        seq = jnp.sqrt(
            1.5 * (dev11**2 + dev22**2 + dev33**2 + 2.0 * dev12**2)
        )
        sy0 = SIGMA_Y0 * (1.0 + (E / SIGMA_Y0) * eq0) ** HARDENING_N
        yielding = (seq - sy0 > 0.0) & (seq >= 1.0e-10)

        def newton(_, gamma):
            eq_new = eq0 + gamma
            term = 1.0 + (E / SIGMA_Y0) * eq_new
            sy = SIGMA_Y0 * term**HARDENING_N
            residual = seq - 3.0 * G * gamma - sy
            derivative = -3.0 * G - HARDENING_N * E * term ** (HARDENING_N - 1.0)
            trial = jnp.maximum(gamma - residual / derivative, 0.0)
            return jnp.where(jnp.abs(residual) < 1.0e-9 * SIGMA_Y0, gamma, trial)

        gamma = jax.lax.fori_loop(0, 50, newton, jnp.zeros_like(eq0))
        gamma = jnp.where(yielding, gamma, 0.0)
        inv_seq = jnp.where(seq > 1.0e-12, 1.0 / seq, 0.0)
        factor = 1.5 * gamma * inv_seq
        ep = ep0 + jnp.stack(
            (factor * dev11, factor * dev22, factor * dev12), axis=1
        )
        ep33 = ep330 + factor * dev33
        eq = eq0 + gamma
        return ep, ep33, eq

    @jax.jit
    def energies(total_strain, ep, ep33, eq):
        ee11 = total_strain[:, 0] - ep[:, 0]
        ee22 = total_strain[:, 1] - ep[:, 1]
        ee12 = total_strain[:, 2] - ep[:, 2]
        ee33 = -ep33
        trace = ee11 + ee22 + ee33
        mean = trace / 3.0
        dev2 = (
            (ee11 - mean) ** 2
            + (ee22 - mean) ** 2
            + (ee33 - mean) ** 2
            + 2.0 * ee12**2
        )
        bulk = LAMBDA + 2.0 * G / 3.0
        psi_e = 0.5 * bulk * jnp.maximum(trace, 0.0) ** 2 + G * dev2
        term = (1.0 + (E / SIGMA_Y0) * eq) ** (HARDENING_N + 1.0)
        psi_p = (
            SIGMA_Y0**2
            / E
            * (term - 1.0)
            / (HARDENING_N + 1.0)
        )
        return psi_e, psi_p

    def phase_matvec(x, local_matrix):
        values = jnp.einsum("eij,ej->ei", local_matrix, x[tri])
        return scatter(tri, values, nnode)

    @jax.jit
    def solve_phase(history, alpha_bar, old_damage):
        fatigue = jnp.where(
            alpha_bar <= ALPHA_T,
            1.0,
            (2.0 * ALPHA_T / (alpha_bar + ALPHA_T)) ** 2,
        )
        local = (
            fatigue[:, None, None] * GC * L0 * grad
            + (2.0 * history + fatigue * GC / L0)[:, None, None] * mass
        )
        rhs_local = (2.0 * history * areas / 3.0)[:, None] * jnp.ones(
            (1, 3), dtype=dtype
        )
        rhs = scatter(tri, rhs_local, nnode)
        diagonal = scatter(
            tri, jnp.diagonal(local, axis1=1, axis2=2), nnode
        )
        operator = lambda x: phase_matvec(x, local)
        solved, _ = cg(
            operator,
            rhs,
            x0=old_damage,
            tol=cg_tol,
            atol=0.0,
            maxiter=cg_maxiter,
            M=lambda r: r / jnp.maximum(diagonal, 1.0e-30),
        )
        residual = jnp.linalg.norm(operator(solved) - rhs) / (
            jnp.linalg.norm(rhs) + 1.0e-30
        )
        damage = jnp.clip(jnp.maximum(solved, old_damage), 0.0, 1.0)
        return damage, residual

    @jax.jit
    def equivalent_stress(total_strain, ep, ep33):
        ee11 = total_strain[:, 0] - ep[:, 0]
        ee22 = total_strain[:, 1] - ep[:, 1]
        ee12 = total_strain[:, 2] - ep[:, 2]
        ee33 = -ep33
        tr = ee11 + ee22 + ee33
        s11 = 2.0 * G * ee11 + LAMBDA * tr
        s22 = 2.0 * G * ee22 + LAMBDA * tr
        s33 = 2.0 * G * ee33 + LAMBDA * tr
        s12 = 2.0 * G * ee12
        p = (s11 + s22 + s33) / 3.0
        return jnp.sqrt(
            1.5 * ((s11 - p) ** 2 + (s22 - p) ** 2 + (s33 - p) ** 2 + 2 * s12**2)
        )

    return solve_u, strain, return_map, energies, solve_phase, equivalent_stress


DIAGNOSTIC_FIELDS = (
    "N,time_s,delta_N,alpha_bar_max,s_max,alpha_p_max,f,x_tip,delta_a,"
    "phi_dwep,phi_tip,phi_wake,phi_ahead,delta_phi_block,"
    "delta_phi_per_cycle_proxy,delta_alpha_block,delta_alpha_per_cycle_proxy,"
    "delta_a_block,delta_a_per_cycle_proxy,phi_tip_rate,alpha_bar_rate_linfit,"
    "psie_tip_peak_mean,psie_tip_unload_mean,psie_peak_unload_ratio,"
    "dpsip_tip_mean,psip_tip_mean,chi_e_peak,chi_e_unload,chi_e_peak_cum,"
    "chi_e_unload_cum,n_tip,alpha_bar_tip_ahead_mean,alpha_bar_tip_ahead_max,"
    "n_ahead_alpha_cells,n_plastic_active,r_p_estimate,max_grad_s,"
    "damage_zone_width,n_cells_s_above_01"
).split(",")
KHF_FIELDS = (
    "N,time_s,a,alpha_max,alpha_mean,alpha_var,alpha_tip_mean,"
    "psip_max,psip_mean,psip_var,psip_tip_mean,"
    "psie_max,psie_mean,psie_var,psie_tip_mean,d_max,d_mean,d_tip_mean"
).split(",")


def plastic_energy(eq):
    return (
        SIGMA_Y0**2
        / E
        * ((1.0 + E / SIGMA_Y0 * eq) ** (HARDENING_N + 1.0) - 1.0)
        / (HARDENING_N + 1.0)
    )


def fatigue_degradation(alpha):
    return 1.0 if alpha <= ALPHA_T else (2.0 * ALPHA_T / (alpha + ALPHA_T)) ** 2


def write_rows(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, cycle: int, state: dict, rows: dict):
    arrays = {key: np.asarray(value) for key, value in state.items()}
    np.savez_compressed(
        path,
        cycle=np.asarray(cycle),
        rows_json=np.asarray(json.dumps(rows)),
        **arrays,
    )


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    state = {key: np.asarray(data[key]) for key in data.files if key not in ("cycle", "rows_json")}
    return int(data["cycle"]), state, json.loads(str(data["rows_json"]))


def save_vtu(path, nodes, tri, displacement, damage, cell_data):
    import meshio

    points = np.column_stack((nodes, np.zeros(len(nodes))))
    vector_u = np.column_stack(
        (displacement[0::2], displacement[1::2], np.zeros(len(nodes)))
    )
    meshio.write_points_cells(
        str(path),
        points,
        [("triangle", tri)],
        point_data={"d": damage, "u": vector_u},
        cell_data={key: [np.asarray(value)] for key, value in cell_data.items()},
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
    arrays = {key: np.asarray(value) for key, value in state.items()}
    arrays["u"] = arrays["u"].reshape(-1, 2)
    np.savez_compressed(
        temporary,
        nodes=nodes,
        triangles=tri,
        displacement=arrays.pop("u"),
        damage=arrays.pop("d"),
        **arrays,
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    temporary.replace(path)
    return True


def linear_slope(x, y):
    if len(x) < 2:
        return 0.0
    xx, yy = np.asarray(x), np.asarray(y)
    denominator = np.sum((xx - xx.mean()) ** 2)
    return 0.0 if denominator <= 1.0e-30 else float(
        np.sum((xx - xx.mean()) * (yy - yy.mean())) / denominator
    )


def tip_energy_metrics(strain_np, ep, ep33, eq, centers, x_tip, dpsi_p):
    mask = (
        (centers[:, 0] >= x_tip - TIP_X_BACK)
        & (centers[:, 0] <= x_tip + TIP_X_FRONT)
        & (np.abs(centers[:, 1] - HEIGHT / 2.0) <= TIP_Y_HALF)
    )
    count = int(mask.sum())
    if not count:
        return 0.0, 0.0, 1.0, 0, 0.0, 1.0
    ee11 = strain_np[:, 0] - ep[:, 0]
    ee22 = strain_np[:, 1] - ep[:, 1]
    ee12 = strain_np[:, 2] - ep[:, 2]
    ee33 = -ep33
    trace = ee11 + ee22 + ee33
    mean = trace / 3.0
    psi_e = (
        0.5 * (LAMBDA + 2 * G / 3) * np.maximum(trace, 0) ** 2
        + G
        * (
            (ee11 - mean) ** 2
            + (ee22 - mean) ** 2
            + (ee33 - mean) ** 2
            + 2 * ee12**2
        )
    )
    e_mean = float(psi_e[mask].mean())
    dp_mean = float(dpsi_p[mask].mean())
    p_mean = float(plastic_energy(eq)[mask].mean())
    return (
        e_mean,
        dp_mean,
        e_mean / (e_mean + dp_mean + 1.0e-12),
        count,
        p_mean,
        e_mean / (e_mean + p_mean + 1.0e-12),
    )


def reference_comparison(reference_dir: Path, diagnostics):
    path = reference_dir / "diagnostics.csv"
    if not reference_dir.is_dir():
        print(
            f"reference comparison disabled: directory not found: {reference_dir}"
        )
        return None
    if not path.is_file():
        print(f"reference comparison disabled: diagnostics not found: {path}")
        return None
    if not diagnostics:
        print("reference comparison disabled: this run produced no diagnostics")
        return None
    try:
        reference = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        n = min(len(diagnostics), len(np.atleast_1d(reference)))
        fields = {
            "alpha_bar_max": "alpha_bar_max",
            "s_max": "s_max",
            "alpha_p_max": "alpha_p_max",
            "delta_a": "delta_a",
        }
        result = {"cycles_compared": n}
        for output_name, reference_name in fields.items():
            actual = np.asarray([float(row[output_name]) for row in diagnostics[:n]])
            expected = np.atleast_1d(reference[reference_name])[:n]
            result[f"{output_name}_relative_l2"] = float(
                np.linalg.norm(actual - expected)
                / (np.linalg.norm(expected) + 1.0e-30)
            )
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"reference comparison disabled: could not read {path}: {exc}")
        return None


def main() -> None:
    args = build_parser().parse_args()
    if args.cycles < 1 or args.steps_per_cycle < 2 or args.steps_per_cycle % 2:
        raise ValueError("cycles must be positive and steps-per-cycle positive/even")
    if min(args.cg_tol, args.stagger_tol, args.picard_tol) <= 0:
        raise ValueError("solver tolerances must be positive")
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if args.platform == "gpu" and all(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU requested but JAX devices are {devices}")
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("FP64 could not be enabled")
    selected_device_slug = device_slug(devices[0])
    if args.outdir is None:
        root = script_root()
        args.outdir = (
            root / "outputs" / f"results_{selected_device_slug}"
            / "plastic_fatigue_SENT" / "jax"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.mesh is None:
        args.mesh = args.outdir / "mesh.msh"
        generate_mesh(args.mesh)
    elif not args.mesh.is_file():
        raise FileNotFoundError(f"explicit mesh not found: {args.mesh}")
    else:
        print(f"using existing explicit mesh unchanged: {args.mesh}")
    checkpoint_path = args.outdir / "checkpoint_latest.npz"
    nodes, tri, areas, bottom, top = load_mesh(args.mesh)
    gx, gy, bmat, ke0, mass, grad, dofs, centers = preprocess(nodes, tri, areas)
    nnode, nel, ndof = len(nodes), len(tri), 2 * len(nodes)
    if args.mesh_only:
        print(
            f"mesh validation passed: {nnode} nodes, {nel} triangles, "
            f"{len(bottom)} bottom nodes, {len(top)} top nodes"
        )
        return
    # Julia masks: bottom=(x,y), top=(y only).
    fixed = np.unique(np.concatenate((2 * bottom, 2 * bottom + 1, 2 * top + 1)))
    free = np.ones(ndof)
    free[fixed] = 0.0
    top_y_dofs = 2 * top + 1
    solve_u, strain_kernel, return_map, energies, solve_phase, equivalent_stress = (
        make_kernels(
            tri,
            dofs,
            bmat,
            ke0,
            mass,
            grad,
            areas,
            free,
            args.cg_tol,
            args.cg_maxiter,
        )
    )

    if args.resume and checkpoint_path.is_file():
        start_cycle, state, all_rows = load_checkpoint(checkpoint_path)
        diagnostics = all_rows["diagnostics"]
        khf_rows = all_rows["khf"]
        summary_rows = all_rows["summary"]
    else:
        start_cycle = 0
        state = {
            "u": np.zeros(ndof),
            "d": np.zeros(nnode),
            "history": np.zeros(nel),
            "alpha_bar": np.zeros(nel),
            "alpha_inst_prev": np.zeros(nel),
            "ep": np.zeros((nel, 3)),
            "ep33": np.zeros(nel),
            "eq": np.zeros(nel),
            "psi_p_prev": np.zeros(nel),
            "psi_e": np.zeros(nel),
        }
        diagnostics, khf_rows, summary_rows = [], [], []

    u = jnp.asarray(state["u"])
    damage = jnp.asarray(state["d"])
    history = jnp.asarray(state["history"])
    alpha_bar = jnp.asarray(state["alpha_bar"])
    alpha_inst_prev = jnp.asarray(state["alpha_inst_prev"])
    ep = jnp.asarray(state["ep"])
    ep33 = jnp.asarray(state["ep33"])
    eq = jnp.asarray(state["eq"])
    psi_p_prev = jnp.asarray(state["psi_p_prev"])
    psi_e = jnp.asarray(state["psi_e"])
    elapsed_offset = float(summary_rows[-1]["time_s"]) if summary_rows else 0.0
    run_t0 = time.perf_counter()
    recent_n = [float(row["N"]) for row in diagnostics[-5:]]
    recent_alpha = [float(row["alpha_bar_max"]) for row in diagnostics[-5:]]
    prev_phi = float(diagnostics[-1]["phi_dwep"]) if diagnostics else 0.0
    prev_phi_tip = float(diagnostics[-1]["phi_tip"]) if diagnostics else 0.0
    prev_alpha_max = float(diagnostics[-1]["alpha_bar_max"]) if diagnostics else 0.0
    prev_da = float(diagnostics[-1]["delta_a"]) if diagnostics else 0.0
    next_vtk = sum(prev_da >= level for level in VTK_A_LEVELS)
    snapshot_dir = args.outdir / "snapshots"
    snapshot_done = [
        (snapshot_dir / f"milestone_da{round(1000 * level):04d}um.npz").is_file()
        for level in VTK_A_LEVELS
    ]
    snapshot_io_s = 0.0
    total_substeps = 0
    first_substep_time = None
    solver_time = 0.0
    post_compile_substep_times = []

    half = args.steps_per_cycle // 2
    load_steps = np.linspace(U_MIN, U_MAX, half + 1)
    unload_steps = np.linspace(U_MAX, U_MIN, half + 1)[1:]
    print("=" * 78)
    print("Standalone JAX SENT elastoplastic fatigue phase-field solver")
    print(f"JAX {jax.__version__}; Python {platform.python_version()}; devices={devices}")
    print(f"mesh={nnode} nodes/{nel} cells; FP64; l0={L0}; cycles={args.cycles}")
    print(f"reference={args.reference_dir}")
    print("=" * 78)

    completed_cycle = start_cycle
    for cycle in range(start_cycle + 1, args.cycles + 1):
        eq_before = eq
        peak_u = u
        best_peak = np.inf
        cycle_complete = True
        substep_counter = 0
        for prescribed in np.concatenate((load_steps, unload_steps)):
            sub_t0 = time.perf_counter()
            substep_counter += 1
            boundary_np = np.zeros(ndof)
            boundary_np[top_y_dofs] = prescribed
            boundary = jnp.asarray(boundary_np)
            ep_start, ep33_start, eq_start = ep, ep33, eq
            ep_work, ep33_work, eq_work = ep, ep33, eq
            last_u_res = np.inf
            for _picard in range(args.picard_maxiter):
                old_u = u
                u_new, u_res = solve_u(damage, boundary, ep_work, ep33_work, u)
                total_strain = strain_kernel(u_new)
                ep_new, ep33_new, eq_new = return_map(
                    total_strain, ep_start, ep33_start, eq_start
                )
                jax.block_until_ready(eq_new)
                change = float(
                    jax.device_get(
                        jnp.linalg.norm(u_new - old_u)
                        / (jnp.linalg.norm(u_new) + 1.0e-10)
                    )
                )
                u, ep_work, ep33_work, eq_work = u_new, ep_new, ep33_new, eq_new
                last_u_res = float(jax.device_get(u_res))
                if change < args.picard_tol:
                    break
            ep, ep33, eq = ep_work, ep33_work, eq_work
            total_strain = strain_kernel(u)
            psi_e, psi_p = energies(total_strain, ep, ep33, eq)
            psi_total = psi_e + ETA_PLASTIC * psi_p
            history = jnp.maximum(history, psi_total)
            # Gridap's first degree-2 triangle quadrature point has P1 values
            # (2/3,1/6,1/6); this choice matters only in g(d) for alpha.
            d_q0 = (
                2.0 * damage[tri[:, 0]]
                + 0.5 * damage[tri[:, 1]]
                + 0.5 * damage[tri[:, 2]]
            ) / 3.0
            alpha_inst = ((1.0 - jnp.clip(d_q0, 0, 1)) ** 2 + ETA_STAB) * psi_total
            alpha_bar = alpha_bar + jnp.maximum(alpha_inst - alpha_inst_prev, 0.0)
            alpha_inst_prev = alpha_inst
            last_d_res = 0.0
            if args.phase_every > 0 and substep_counter % args.phase_every == 0:
                for _stagger in range(1, args.max_stagger + 1):
                    old_damage = damage
                    damage, d_res = solve_phase(history, alpha_bar, damage)
                    jax.block_until_ready(damage)
                    phase_change = float(
                        jax.device_get(
                            jnp.linalg.norm(damage - old_damage)
                            / (jnp.linalg.norm(damage) + 1.0e-10)
                        )
                    )
                    last_d_res = float(jax.device_get(d_res))
                    if phase_change < args.stagger_tol:
                        break
            if max(last_u_res, last_d_res) > args.linear_residual_tol:
                raise RuntimeError(
                    f"linear residual exceeded tolerance: u={last_u_res:.3e}, "
                    f"d={last_d_res:.3e}"
                )
            if abs(prescribed - U_MAX) < best_peak:
                best_peak = abs(prescribed - U_MAX)
                peak_u = u
            total_substeps += 1
            sub_time = time.perf_counter() - sub_t0
            if first_substep_time is None:
                first_substep_time = sub_time
            else:
                post_compile_substep_times.append(sub_time)
            solver_time += sub_time
            if args.stop_after_substeps and total_substeps >= args.stop_after_substeps:
                cycle_complete = False
                break
        if not cycle_complete:
            print("stopped at requested substep limit; incomplete cycle not committed")
            break

        jax.block_until_ready(damage)
        completed_cycle = cycle
        elapsed = elapsed_offset + time.perf_counter() - run_t0 - snapshot_io_s
        host_u = np.asarray(jax.device_get(u))
        host_peak_u = np.asarray(jax.device_get(peak_u))
        host_d = np.asarray(jax.device_get(damage))
        host_h = np.asarray(jax.device_get(history))
        host_alpha = np.asarray(jax.device_get(alpha_bar))
        host_ep = np.asarray(jax.device_get(ep))
        host_ep33 = np.asarray(jax.device_get(ep33))
        host_eq = np.asarray(jax.device_get(eq))
        host_psi_e = np.asarray(jax.device_get(psi_e))
        psi_p_np = plastic_energy(host_eq)
        dpsi_p = np.maximum(psi_p_np - np.asarray(jax.device_get(psi_p_prev)), 0)
        psi_p_prev = jnp.asarray(psi_p_np)
        d_cell = host_d[tri].mean(axis=1)
        near_line = np.abs(nodes[:, 1] - HEIGHT / 2.0) < 2.0 * L0
        tip_nodes = (nodes[:, 0] > A0 - L0) & near_line & (host_d > 0.5)
        x_tip = max(A0, float(nodes[tip_nodes, 0].max()) if tip_nodes.any() else A0)
        delta_a = x_tip - A0
        tip_mask = (
            (centers[:, 0] >= x_tip - TIP_X_BACK)
            & (centers[:, 0] <= x_tip + TIP_X_FRONT)
            & (np.abs(centers[:, 1] - HEIGHT / 2) <= TIP_Y_HALF)
        )
        ahead = (
            (centers[:, 0] >= x_tip)
            & (centers[:, 0] <= x_tip + AHEAD_ALPHA_FRONT)
            & (np.abs(centers[:, 1] - HEIGHT / 2) <= TIP_Y_HALF)
        )
        weight = np.maximum(1.0 - ((1.0 - d_cell) ** 2 + ETA_STAB), 0)
        contribution = weight * host_eq
        yband = np.abs(centers[:, 1] - HEIGHT / 2) <= TIP_Y_HALF
        phi = float(contribution.sum())
        phi_tip = float(
            contribution[
                yband & (centers[:, 0] >= x_tip - L0) & (centers[:, 0] <= x_tip + 3 * L0)
            ].sum()
        )
        phi_wake = float(
            contribution[
                yband
                & (centers[:, 0] >= x_tip - WAKE_X_BACK)
                & (centers[:, 0] < x_tip - L0)
            ].sum()
        )
        phi_ahead = float(
            contribution[
                yband & (centers[:, 0] >= x_tip) & (centers[:, 0] <= x_tip + AHEAD_X_FRONT)
            ].sum()
        )
        peak_strain = np.einsum("eij,ej->ei", bmat, host_peak_u[dofs])
        peak_strain[:, 2] *= 0.5
        unload_strain = np.einsum("eij,ej->ei", bmat, host_u[dofs])
        unload_strain[:, 2] *= 0.5
        pe, dp, chi_peak, n_tip, pp, chi_peak_cum = tip_energy_metrics(
            peak_strain, host_ep, host_ep33, host_eq, centers, x_tip, dpsi_p
        )
        ue, _, chi_unload, _, _, chi_unload_cum = tip_energy_metrics(
            unload_strain, host_ep, host_ep33, host_eq, centers, x_tip, dpsi_p
        )
        active = int(np.sum(host_eq - np.asarray(jax.device_get(eq_before)) > EPSP_ACTIVE_THRESH))
        zone = host_eq > EPSP_ZONE_THRESH
        rp = (
            float(np.max(np.hypot(centers[zone, 0] - x_tip, centers[zone, 1] - HEIGHT / 2)))
            if zone.any()
            else 0.0
        )
        grad_d = np.sqrt(
            (np.sum(host_d[tri] * gx, axis=1)) ** 2
            + (np.sum(host_d[tri] * gy, axis=1)) ** 2
        )
        midband = yband & (d_cell > 0.1) & (d_cell < 0.9)
        zone_width = (
            float(np.ptp(centers[midband, 0])) if np.count_nonzero(midband) else 0.0
        )
        recent_n.append(cycle)
        recent_alpha.append(float(host_alpha.max()))
        recent_n, recent_alpha = recent_n[-5:], recent_alpha[-5:]
        delta_phi = phi - prev_phi
        delta_alpha = float(host_alpha.max()) - prev_alpha_max
        delta_a_block = delta_a - prev_da
        diag = {
            "N": cycle,
            "time_s": elapsed,
            "delta_N": 1,
            "alpha_bar_max": float(host_alpha.max()),
            "s_max": float(host_d.max()),
            "alpha_p_max": float(host_eq.max()),
            "f": fatigue_degradation(float(host_alpha.max())),
            "x_tip": x_tip,
            "delta_a": delta_a,
            "phi_dwep": phi,
            "phi_tip": phi_tip,
            "phi_wake": phi_wake,
            "phi_ahead": phi_ahead,
            "delta_phi_block": delta_phi,
            "delta_phi_per_cycle_proxy": delta_phi,
            "delta_alpha_block": delta_alpha,
            "delta_alpha_per_cycle_proxy": delta_alpha,
            "delta_a_block": delta_a_block,
            "delta_a_per_cycle_proxy": delta_a_block,
            "phi_tip_rate": (phi_tip - prev_phi_tip) / max(phi_tip, 1.0e-30),
            "alpha_bar_rate_linfit": linear_slope(recent_n, recent_alpha),
            "psie_tip_peak_mean": pe,
            "psie_tip_unload_mean": ue,
            "psie_peak_unload_ratio": pe / max(ue, 1.0e-12),
            "dpsip_tip_mean": dp,
            "psip_tip_mean": pp,
            "chi_e_peak": chi_peak,
            "chi_e_unload": chi_unload,
            "chi_e_peak_cum": chi_peak_cum,
            "chi_e_unload_cum": chi_unload_cum,
            "n_tip": n_tip,
            "alpha_bar_tip_ahead_mean": float(host_alpha[ahead].mean()) if ahead.any() else 0.0,
            "alpha_bar_tip_ahead_max": float(host_alpha[ahead].max()) if ahead.any() else 0.0,
            "n_ahead_alpha_cells": int(ahead.sum()),
            "n_plastic_active": active,
            "r_p_estimate": rp,
            "max_grad_s": float(grad_d.max()),
            "damage_zone_width": zone_width,
            "n_cells_s_above_01": int(np.sum(d_cell > 0.1)),
        }
        diagnostics.append(diag)
        khf_rows.append(
            {
                "N": cycle,
                "time_s": elapsed,
                "a": x_tip,
                "alpha_max": float(host_alpha.max()),
                "alpha_mean": float(host_alpha.mean()),
                "alpha_var": float(host_alpha.var()),
                "alpha_tip_mean": float(host_alpha[tip_mask].mean()) if tip_mask.any() else 0.0,
                "psip_max": float(psi_p_np.max()),
                "psip_mean": float(psi_p_np.mean()),
                "psip_var": float(psi_p_np.var()),
                "psip_tip_mean": float(psi_p_np[tip_mask].mean()) if tip_mask.any() else 0.0,
                "psie_max": float(host_psi_e.max()),
                "psie_mean": float(host_psi_e.mean()),
                "psie_var": float(host_psi_e.var()),
                "psie_tip_mean": float(host_psi_e[tip_mask].mean()) if tip_mask.any() else 0.0,
                "d_max": float(d_cell.max()),
                "d_mean": float(d_cell.mean()),
                "d_tip_mean": float(d_cell[tip_mask].mean()) if tip_mask.any() else 0.0,
            }
        )
        summary_rows.append({"N": cycle, "time_s": elapsed, "delta_a_mm": delta_a})
        prev_phi, prev_phi_tip = phi, phi_tip
        prev_alpha_max, prev_da = float(host_alpha.max()), delta_a
        eq_before = eq

        cell_output = {
            "s_cell": d_cell,
            "epspeq": host_eq,
            "psip": psi_p_np,
            "sigmaeq": np.asarray(
                jax.device_get(equivalent_stress(strain_kernel(u), ep, ep33))
            ),
            "alpha_bar": host_alpha,
            "history": host_h,
        }
        while not args.no_vtk and next_vtk < len(VTK_A_LEVELS) and delta_a >= VTK_A_LEVELS[next_vtk]:
            save_vtu(
                args.outdir / f"result_a{round(1000*VTK_A_LEVELS[next_vtk])}_N{cycle:06d}.vtu",
                nodes,
                tri,
                host_u,
                host_d,
                cell_output,
            )
            next_vtk += 1
        if not args.no_vtk and cycle % 10 == 0:
            save_vtu(
                args.outdir / f"result_N{cycle:06d}.vtu",
                nodes,
                tri,
                host_u,
                host_d,
                cell_output,
            )
        if cycle <= 5 or cycle % args.print_every == 0:
            print(
                f"N={cycle:4d} epspeq={host_eq.max():.6f} "
                f"s={host_d.max():.5f} alphabar={host_alpha.max():.4f} "
                f"da={delta_a:.6f} mm t={elapsed:.2f}s"
            )

        state = {
            "u": host_u,
            "d": host_d,
            "history": host_h,
            "alpha_bar": host_alpha,
            "alpha_inst_prev": np.asarray(jax.device_get(alpha_inst_prev)),
            "ep": host_ep,
            "ep33": host_ep33,
            "eq": host_eq,
            "psi_p_prev": psi_p_np,
            "psi_e": host_psi_e,
        }
        for index, level in enumerate(VTK_A_LEVELS):
            if not args.no_snapshots and not snapshot_done[index] and delta_a >= level:
                snapshot_t0 = time.perf_counter()
                save_scientific_snapshot(
                    snapshot_dir / f"milestone_da{round(1000 * level):04d}um.npz",
                    nodes,
                    tri,
                    state,
                    {
                        "cycle": cycle,
                        "step": cycle * args.steps_per_cycle,
                        "load": U_MIN,
                        "qoi_crack_extension_mm": delta_a,
                        "qoi_crack_tip_x_mm": x_tip,
                        "milestone": f"crack_extension_{level:.2f}_mm",
                    },
                )
                snapshot_io_s += time.perf_counter() - snapshot_t0
                snapshot_done[index] = True
        if args.save_every > 0 and (cycle % args.save_every == 0 or cycle == args.cycles):
            write_rows(args.outdir / "diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics)
            write_rows(args.outdir / "khf_data.csv", KHF_FIELDS, khf_rows)
            write_rows(
                args.outdir / "cycle_summary.csv",
                ("N", "time_s", "delta_a_mm"),
                summary_rows,
            )
            save_checkpoint(
                checkpoint_path,
                cycle,
                state,
                {"diagnostics": diagnostics, "khf": khf_rows, "summary": summary_rows},
            )
        if delta_a > 0.4 or x_tip > 0.95:
            print("crack approached the boundary; stopping as in Julia")
            break

    total_time = time.perf_counter() - run_t0 - snapshot_io_s
    if completed_cycle:
        write_rows(args.outdir / "diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostics)
        write_rows(args.outdir / "khf_data.csv", KHF_FIELDS, khf_rows)
        write_rows(
            args.outdir / "cycle_summary.csv",
            ("N", "time_s", "delta_a_mm"),
            summary_rows,
        )
        save_checkpoint(
            checkpoint_path,
            completed_cycle,
            state,
            {"diagnostics": diagnostics, "khf": khf_rows, "summary": summary_rows},
        )
        if not args.no_snapshots:
            snapshot_t0 = time.perf_counter()
            final_diag = diagnostics[-1]
            save_scientific_snapshot(
                snapshot_dir / "final.npz",
                nodes,
                tri,
                state,
                {
                    "cycle": completed_cycle,
                    "step": completed_cycle * args.steps_per_cycle,
                    "load": U_MIN,
                    "qoi_crack_extension_mm": float(final_diag["delta_a"]),
                    "qoi_crack_tip_x_mm": float(final_diag["x_tip"]),
                    "milestone": "final",
                },
            )
    comparison = reference_comparison(args.reference_dir, diagnostics)
    final_qois = (
        {
            key: diagnostics[-1][key]
            for key in (
                "N",
                "alpha_bar_max",
                "s_max",
                "alpha_p_max",
                "f",
                "x_tip",
                "delta_a",
            )
        }
        if diagnostics
        else None
    )
    timing = {
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "device_strings": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "device_slug": selected_device_slug,
        "dtype": "float64",
        "mesh": str(args.mesh),
        "reference_dir": str(args.reference_dir),
        "nodes": nnode,
        "triangles": nel,
        "completed_cycle": completed_cycle,
        "substeps_this_run": total_substeps,
        "total_s": total_time,
        "solver_s_excluding_output": solver_time,
        "first_substep_compile_and_solve_s": first_substep_time,
        "post_compile_substep_mean_s": (
            float(np.mean(post_compile_substep_times))
            if post_compile_substep_times
            else None
        ),
        "visualization_in_solver_timing": False,
        "final_qois": final_qois,
        "reference_comparison": comparison,
    }
    with (args.outdir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    with (args.outdir / "qois.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "case": "plastic_fatigue_SENT",
                "final": final_qois,
                "reference_comparison": comparison,
            },
            stream,
            indent=2,
        )
    print("-" * 78)
    print(f"completed through cycle {completed_cycle} in {total_time:.3f}s")
    if comparison:
        print(f"reference comparison: {comparison}")
    print(f"outputs: {args.outdir}")


if __name__ == "__main__":
    main()
