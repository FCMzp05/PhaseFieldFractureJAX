#!/usr/bin/env python3
"""
Standalone JAX/GPU port of elastic_fatigue_TPB_AT1.jl.

The reference discretization is retained: plane-strain P1 triangles, degree-two
triangle quadrature, continuous displacement and phase fields, quadrature-point
history/state fields, the rational length-scale-insensitive degradation law,
AT1, and five staggered iterations with a ten-iteration Newton phase solve.

Defaults are the Julia benchmark defaults (400 cycles x 10 steps).  The script
generates the same Gmsh geometry/size field unless --mesh is supplied.  It has
no project-local imports.  Dependencies are numpy, meshio, gmsh, and JAX
(a CUDA-enabled JAX installation is needed for --platform gpu).

Reference run:
  python elastic_fatigue_TPB_AT1_jax_gpu.py --platform gpu

Small non-reference smoke run:
  python elastic_fatigue_TPB_AT1_jax_gpu.py --platform cpu --max-steps 1 \
      --mesh-scale 100 --no-plot --no-vtk
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


def _requested_platform() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parsed, _ = parser.parse_known_args()
    return parsed.platform


_PLATFORM = _requested_platform()
if _PLATFORM != "auto":
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if _PLATFORM == "gpu" else "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax.scipy.sparse.linalg import bicgstab, cg  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "JAX is required. Install a CUDA build for GPU execution, plus gmsh and "
        f"meshio. Original import error: {exc}"
    ) from exc


# Exact constants from elastic_fatigue_TPB_AT1.jl.
E_MAT, NU_MAT, GC, LS, ETA = 20.8, 0.3, 5.0e-4, 0.06, 1.0e-15
L, HEIGHT, CH, CW = 8.0, 2.0, 0.4, 0.2
LL, LR = 0.475 * L, 0.525 * L
HFC, HF, H_COARSE = 0.03 / 100.0, 0.03 / 5.0, 50.0 * (0.03 / 5.0)
SIGMA_C = 0.2
M_PARAM = 3.0 * E_MAT * GC / (4.0 * LS * SIGMA_C**2)
PSI_CRIT = SIGMA_C**2 / (2.0 * E_MAT)
KF = 0.5
ALPHA_T = PSI_CRIT / KF
T_TOTAL = 1.0
N_CYCLES, N_STEPS_PER_CYCLE = 400, 10
INNER_MAX, REL_TOL, DAMAGE_TOL = 5, 1.0e-6, 0.05
U_MAX, R_RATIO = 0.025 / 2.0, 0.0
SAVE_CYCLE_INTERVAL = 40

# Degree-two symmetric triangle rule used by Gridap (weights A/3).
QP_N = np.asarray(
    [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
     [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
     [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]],
    dtype=np.float64,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Standalone FP64 JAX/GPU AT1 TPB elastic-fatigue benchmark"
    )
    result.add_argument(
        "--outdir", type=Path, default=None,
        help="output directory (default: outputs/results_<device>/elastic_fatigue_TPB_AT1/jax)",
    )
    result.add_argument("--mesh", type=Path, default=None, help="reuse a Gmsh .msh; default generates the Julia mesh")
    result.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    result.add_argument("--cycles", type=int, default=N_CYCLES)
    result.add_argument("--steps-per-cycle", type=int, default=N_STEPS_PER_CYCLE)
    result.add_argument("--max-steps", type=int, default=0, help="stop after this many global steps; 0 runs all")
    result.add_argument("--inner-max", type=int, default=INNER_MAX)
    result.add_argument("--rel-tol", type=float, default=REL_TOL)
    result.add_argument("--damage-tol", type=float, default=DAMAGE_TOL)
    result.add_argument("--u-max", type=float, default=U_MAX)
    result.add_argument("--r-ratio", type=float, default=R_RATIO)
    result.add_argument("--mesh-scale", type=float, default=1.0, help="multiply all Julia mesh sizes (testing only)")
    result.add_argument("--newton-max", type=int, default=10)
    result.add_argument("--newton-tol", type=float, default=1.0e-10)
    result.add_argument("--linear-tol", type=float, default=1.0e-10)
    result.add_argument("--linear-maxiter", type=int, default=4000)
    result.add_argument("--save-cycle-interval", type=int, default=SAVE_CYCLE_INTERVAL)
    result.add_argument("--checkpoint-every", type=int, default=N_STEPS_PER_CYCLE)
    result.add_argument("--print-every", type=int, default=N_STEPS_PER_CYCLE)
    result.add_argument("--resume", action="store_true")
    result.add_argument(
        "--no-snapshots", dest="snapshots", action="store_false",
        help="disable sparse scientific NPZ state snapshots",
    )
    result.set_defaults(snapshots=True)
    result.add_argument("--vtk", action="store_true", help="write optional VTU fields (off by default)")
    result.add_argument("--plot", action="store_true", help="write optional plots (off by default)")
    result.add_argument("--no-vtk", dest="vtk", action="store_false", help=argparse.SUPPRESS)
    result.add_argument("--no-plot", dest="plot", action="store_false", help=argparse.SUPPRESS)
    return result


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def create_mesh(filename: Path, scale: float) -> None:
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh is required to generate the benchmark mesh") from exc
    filename.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("BeamWithNotchSymThreePtBending")
        sizes = [HF, H_COARSE, H_COARSE, HF, HF, H_COARSE, H_COARSE, HF, HFC]
        points = [
            (L / 2 + CW / 2, 0.0), (L, 0.0), (L, HEIGHT), (LR, HEIGHT),
            (LL, HEIGHT), (0.0, HEIGHT), (0.0, 0.0),
            (L / 2 - CW / 2, 0.0), (L / 2, CH),
        ]
        for tag, ((x, y), size) in enumerate(zip(points, sizes), 1):
            gmsh.model.geo.addPoint(x, y, 0.0, scale * size, tag)
        for tag, (a, b) in enumerate(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
             (7, 8), (8, 9), (9, 1)],
            1,
        ):
            gmsh.model.geo.addLine(a, b, tag)
        gmsh.model.geo.addCurveLoop(list(range(1, 10)), 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [1], 1)
        gmsh.model.addPhysicalGroup(1, [4], 1)
        gmsh.model.addPhysicalGroup(0, [7], 2)
        gmsh.model.addPhysicalGroup(0, [2], 3)
        gmsh.model.setPhysicalName(2, 1, "Domain")
        gmsh.model.setPhysicalName(1, 1, "LoadLine")
        gmsh.model.setPhysicalName(0, 2, "LeftSupport")
        gmsh.model.setPhysicalName(0, 3, "RightSupport")
        gmsh.model.mesh.field.add("Box", 10)
        gmsh.model.mesh.field.setNumber(10, "VIn", scale * HF)
        gmsh.model.mesh.field.setNumber(10, "VOut", scale * H_COARSE)
        gmsh.model.mesh.field.setNumber(10, "XMin", L / 2 - CW)
        gmsh.model.mesh.field.setNumber(10, "XMax", L / 2 + CW)
        gmsh.model.mesh.field.setNumber(10, "YMin", 0.0)
        gmsh.model.mesh.field.setNumber(10, "YMax", HEIGHT)
        gmsh.model.mesh.field.setAsBackgroundMesh(10)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(filename))
        print(
            f"  generated mesh: {len(gmsh.model.mesh.getNodes()[0])} nodes, "
            f"{len(gmsh.model.mesh.getElementsByType(2)[0])} triangles"
        )
    finally:
        gmsh.finalize()


def load_mesh(filename: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required") from exc
    mesh = meshio.read(str(filename))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangles = [np.asarray(block.data, dtype=np.int64) for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        raise RuntimeError("mesh has no linear triangles")
    tri = np.concatenate(triangles)
    load_edges: list[np.ndarray] = []
    left: set[int] = set()
    right: set[int] = set()
    physical = mesh.cell_data.get("gmsh:physical", [])
    for index, block in enumerate(mesh.cells):
        tags = physical[index] if index < len(physical) else np.asarray([], dtype=int)
        if block.type == "line":
            for edge, tag in zip(block.data, tags):
                if int(tag) == 1:
                    load_edges.append(np.asarray(edge, dtype=np.int64))
        elif block.type == "vertex":
            for vertex, tag in zip(block.data, tags):
                if int(tag) == 2:
                    left.add(int(vertex[0]))
                elif int(tag) == 3:
                    right.add(int(vertex[0]))
    tol = 1.0e-9
    if not left:
        left.update(np.where((abs(nodes[:, 0]) < tol) & (abs(nodes[:, 1]) < tol))[0].tolist())
    if not right:
        right.update(np.where((abs(nodes[:, 0] - L) < tol) & (abs(nodes[:, 1]) < tol))[0].tolist())
    if not load_edges:
        top = np.where(
            (abs(nodes[:, 1] - HEIGHT) < tol)
            & (nodes[:, 0] >= LL - tol) & (nodes[:, 0] <= LR + tol)
        )[0]
        top = top[np.argsort(nodes[top, 0])]
        load_edges = [np.asarray(pair) for pair in zip(top[:-1], top[1:])]
    load_edges_np = np.asarray(load_edges, dtype=np.int64)
    load_nodes = np.unique(load_edges_np)
    if not left or not right or load_nodes.size == 0:
        raise RuntimeError("failed to identify Julia support/load boundary tags")

    xy = nodes[tri]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0.0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    xy = nodes[tri]
    areas = 0.5 * (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    if np.any(areas <= 0.0):
        raise RuntimeError("mesh contains degenerate triangles")
    return mesh, nodes, tri, areas, np.asarray(sorted(left)), np.asarray(sorted(right)), load_nodes, load_edges_np


def preprocess(nodes: np.ndarray, tri: np.ndarray, areas: np.ndarray):
    x, y = nodes[:, 0], nodes[:, 1]
    i, j, k = tri[:, 0], tri[:, 1], tri[:, 2]
    gx = np.stack([y[j] - y[k], y[k] - y[i], y[i] - y[j]], axis=1) / (2.0 * areas[:, None])
    gy = np.stack([x[k] - x[j], x[i] - x[k], x[j] - x[i]], axis=1) / (2.0 * areas[:, None])
    bmat = np.zeros((len(tri), 3, 6), dtype=np.float64)
    dofs = np.empty((len(tri), 6), dtype=np.int64)
    for a in range(3):
        bmat[:, 0, 2 * a] = gx[:, a]
        bmat[:, 1, 2 * a + 1] = gy[:, a]
        bmat[:, 2, 2 * a] = gy[:, a]
        bmat[:, 2, 2 * a + 1] = gx[:, a]
        dofs[:, 2 * a] = 2 * tri[:, a]
        dofs[:, 2 * a + 1] = 2 * tri[:, a] + 1
    grad = areas[:, None, None] * (
        gx[:, :, None] * gx[:, None, :] + gy[:, :, None] * gy[:, None, :]
    )
    mass = areas[:, None, None] * (
        np.asarray([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
    )
    return bmat, dofs, grad, mass


def boundary_cells(tri: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lookup: dict[tuple[int, int], int] = {}
    for cell, element in enumerate(tri):
        for a, b in ((element[0], element[1]), (element[1], element[2]), (element[2], element[0])):
            lookup[tuple(sorted((int(a), int(b))))] = cell
    return np.asarray([lookup[tuple(sorted(map(int, edge)))] for edge in edges], dtype=np.int64)


def point_qp_weights(nodes: np.ndarray, tri: np.ndarray, point=(L / 2, 2 * CH)):
    p = np.asarray(point)
    for cell, ids in enumerate(tri):
        vertices = nodes[ids]
        matrix = np.vstack([vertices.T, np.ones(3)])
        bary = np.linalg.solve(matrix, np.r_[p, 1.0])
        if np.min(bary) >= -1.0e-10:
            return cell, bary
    raise RuntimeError(f"diagnostic point {point} is outside the mesh")


def make_kernels(
    *, tri_np, dofs_np, bmat_np, grad_np, mass_np, areas_np, nodes_np,
    load_edges_np, load_cells_np, free_mask_np, linear_tol, linear_maxiter,
):
    dtype = jnp.float64
    tri, dofs = jnp.asarray(tri_np), jnp.asarray(dofs_np)
    bmat, grad, mass = (
        jnp.asarray(bmat_np), jnp.asarray(grad_np), jnp.asarray(mass_np)
    )
    areas, qn = jnp.asarray(areas_np), jnp.asarray(QP_N)
    free = jnp.asarray(free_mask_np)
    edges, edge_cells = jnp.asarray(load_edges_np), jnp.asarray(load_cells_np)
    load_y_dofs = jnp.asarray(2 * np.unique(load_edges_np) + 1)
    nodes = jnp.asarray(nodes_np)
    edge_lengths = jnp.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    ndof, nnode = len(free_mask_np), len(nodes_np)

    c11 = E_MAT * (1.0 - NU_MAT) / ((1.0 + NU_MAT) * (1.0 - 2.0 * NU_MAT))
    c12 = E_MAT * NU_MAT / ((1.0 + NU_MAT) * (1.0 - 2.0 * NU_MAT))
    shear = E_MAT / (2.0 * (1.0 + NU_MAT))
    cmat = jnp.asarray([[c11, c12, 0.0], [c12, c11, 0.0], [0.0, 0.0, shear]])
    cvol = jnp.asarray(
        [[0.5 * (c11 + c12), 0.5 * (c11 + c12), 0.0],
         [0.5 * (c11 + c12), 0.5 * (c11 + c12), 0.0],
         [0.0, 0.0, 0.0]]
    )
    cdev = cmat - cvol
    ke_full = areas[:, None, None] * jnp.einsum("eia,ij,ejb->eab", bmat, cmat, bmat)
    ke_vol = areas[:, None, None] * jnp.einsum("eia,ij,ejb->eab", bmat, cvol, bmat)
    ke_dev = areas[:, None, None] * jnp.einsum("eia,ij,ejb->eab", bmat, cdev, bmat)
    phase_grad = 0.75 * GC * LS * grad
    rhs_phase_e = ((3.0 / 8.0) * GC / LS) * (areas / 3.0)[:, None] * jnp.ones((1, 3))

    def scatter(indices, values, size):
        return jnp.zeros(size, dtype=dtype).at[indices.reshape(-1)].add(values.reshape(-1))

    @jax.jit
    def project_qp(values, x0):
        """Gridap :L2 P1 projection: three independent DOFs per element."""
        rhs_e = (areas / 3.0)[:, None] * jnp.einsum("qa,eq->ea", qn, values)
        projected = jnp.linalg.solve(mass, rhs_e[..., None]).squeeze(-1)
        residual = jnp.linalg.norm(
            jnp.einsum("eij,ej->ei", mass, projected) - rhs_e
        ) / jnp.maximum(
            jnp.linalg.norm(rhs_e), 1.0
        )
        return projected, residual

    def degradation(s):
        denominator = (
            0.25 * M_PARAM**2 * s**4 - 0.5 * M_PARAM**2 * s**2
            + 0.25 * M_PARAM**2 - M_PARAM * s**4 + M_PARAM * s**2 + s**4
        )
        return M_PARAM * s / denominator

    degradation_prime = jax.grad(lambda value: degradation(value))
    degradation_prime_v = jax.vmap(jax.vmap(degradation_prime))

    def phase_g(s):
        return s**2 / (s**2 + M_PARAM * (1.0 - s) * (0.5 + 0.5 * s))

    def element_stiffness(s, previous_u):
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        gmean = jnp.mean(phase_g(sq), axis=1)
        previous_strain = jnp.einsum("eij,ej->ei", bmat, previous_u[dofs])
        tension = (previous_strain[:, 0] + previous_strain[:, 1]) >= 0.0
        return jnp.where(
            tension[:, None, None],
            gmean[:, None, None] * ke_full,
            gmean[:, None, None] * ke_dev + ke_vol,
        )

    def u_matvec(x, ke):
        return scatter(dofs, jnp.einsum("eij,ej->ei", ke, x[dofs]), ndof)

    @jax.jit
    def solve_displacement(s, previous_u, boundary, x0):
        ke = element_stiffness(s, previous_u)
        rhs = -free * u_matvec(boundary, ke)

        def operator(x):
            return free * u_matvec(free * x, ke) + (1.0 - free) * x

        diagonal = free * scatter(dofs, jnp.diagonal(ke, axis1=1, axis2=2), ndof) + (1.0 - free)
        diagonal = jnp.maximum(jnp.abs(diagonal), 1.0e-30)
        correction, _ = cg(
            operator, rhs, x0=jnp.zeros_like(x0), tol=linear_tol, atol=0.0,
            maxiter=linear_maxiter, M=lambda r: r / diagonal,
        )
        result = boundary + free * correction
        residual = jnp.linalg.norm(operator(correction) - rhs) / (jnp.linalg.norm(rhs) + 1.0e-30)
        return result, residual

    def phase_residual(s, history):
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        history_q = jnp.einsum("qa,ea->eq", qn, history)
        nonlinear = (areas / 3.0)[:, None] * history_q * degradation(sq)
        re = jnp.einsum("eij,ej->ei", phase_grad, s[tri])
        re += jnp.einsum("qa,eq->ea", qn, nonlinear)
        re -= rhs_phase_e
        return scatter(tri, re, nnode)

    def phase_jacobian_data(s, history):
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        history_q = jnp.einsum("qa,ea->eq", qn, history)
        tangent_q = (
            (areas / 3.0)[:, None] * history_q * degradation_prime_v(sq)
        )
        return phase_grad + jnp.einsum("eq,qa,qb->eab", tangent_q, qn, qn)

    @jax.jit
    def newton_increment(s, history):
        residual = phase_residual(s, history)
        jacobian = phase_jacobian_data(s, history)

        def operator(x):
            return scatter(tri, jnp.einsum("eij,ej->ei", jacobian, x[tri]), nnode)

        diagonal = scatter(tri, jnp.diagonal(jacobian, axis1=1, axis2=2), nnode)
        safe_diagonal = jnp.where(jnp.abs(diagonal) > 1.0e-30, diagonal, 1.0)
        delta, _ = bicgstab(
            operator, -residual, x0=jnp.zeros_like(s), tol=linear_tol,
            atol=0.0, maxiter=linear_maxiter, M=lambda r: r / safe_diagonal,
        )
        linear_residual = jnp.linalg.norm(operator(delta) + residual) / (jnp.linalg.norm(residual) + 1.0e-30)
        return delta, jnp.linalg.norm(residual), linear_residual

    @jax.jit
    def residual_norm(s, history):
        return jnp.linalg.norm(phase_residual(s, history))

    @jax.jit
    def rel_error(s, history):
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        history_q = jnp.einsum("qa,ea->eq", qn, history)
        grad_term = jnp.einsum("ei,eij,ej->e", s[tri], phase_grad, s[tri])
        nonlinear = jnp.sum(
            (areas / 3.0)[:, None] * degradation(sq) * history_q * sq
        )
        rhs = ((3.0 / 8.0) * GC / LS) * jnp.sum(areas[:, None] / 3.0 * sq)
        return jnp.abs(jnp.sum(grad_term) + nonlinear - rhs) / (jnp.abs(rhs) + 1.0e-12)

    @jax.jit
    def update_energy(u, old_energy, fatigue_degradation):
        strain = jnp.einsum("eij,ej->ei", bmat, u[dofs])
        stress = jnp.einsum("ij,ej->ei", cmat, strain)
        full = 0.5 * jnp.sum(strain * stress, axis=1)
        dev_strain = strain.at[:, 0].set(0.5 * (strain[:, 0] - strain[:, 1]))
        dev_strain = dev_strain.at[:, 1].set(0.5 * (strain[:, 1] - strain[:, 0]))
        dev_stress = stress.at[:, 0].set(0.5 * (stress[:, 0] - stress[:, 1]))
        dev_stress = dev_stress.at[:, 1].set(0.5 * (stress[:, 1] - stress[:, 0]))
        dev = 0.5 * jnp.sum(dev_strain * dev_stress, axis=1)
        psi_org = jnp.where(strain[:, 0] + strain[:, 1] >= 0.0, full, dev)
        psi = psi_org[:, None] / fatigue_degradation
        return strain, psi_org, jnp.maximum(old_energy, psi)

    @jax.jit
    def update_fatigue(s, psi_org, old_alpha, old_alpha_bar):
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        alpha = phase_g(sq) * psi_org[:, None]
        old_alpha_q = jnp.einsum("qa,ea->eq", qn, old_alpha)
        old_alpha_bar_q = jnp.einsum("qa,ea->eq", qn, old_alpha_bar)
        alpha_bar = jnp.where(
            alpha >= old_alpha_q,
            old_alpha_bar_q + jnp.abs(alpha - old_alpha_q),
            old_alpha_bar_q,
        )
        return alpha, alpha_bar

    @jax.jit
    def diagnostics(s, u, fdeg):
        strain = jnp.einsum("eij,ej->ei", bmat, u[dofs])
        sq = jnp.einsum("qa,ea->eq", qn, s[tri])
        grad_energy = jnp.einsum("ei,eij,ej->e", s[tri], grad, s[tri])
        frac = (3.0 / 8.0) * jnp.sum(
            areas * (1.0 - jnp.mean(sq, axis=1)) / LS + LS * grad_energy
        )
        edge_qn = jnp.asarray(
            [[0.5 * (1.0 + 1.0 / np.sqrt(3.0)), 0.5 * (1.0 - 1.0 / np.sqrt(3.0))],
             [0.5 * (1.0 - 1.0 / np.sqrt(3.0)), 0.5 * (1.0 + 1.0 / np.sqrt(3.0))]]
        )
        edge_s = jnp.einsum("qa,ea->eq", edge_qn, s[edges])
        edge_g = phase_g(edge_s)
        edge_strain = strain[edge_cells]
        edge_stress_full = jnp.einsum("ij,ej->ei", cmat, edge_strain)
        edge_stress_dev = jnp.einsum("ij,ej->ei", cdev, edge_strain)
        edge_stress_vol = jnp.einsum("ij,ej->ei", cvol, edge_strain)
        tensile = edge_strain[:, 0] + edge_strain[:, 1] >= 0.0
        sigma_yy_q = jnp.where(
            tensile[:, None],
            edge_g * edge_stress_full[:, 1, None],
            edge_g * edge_stress_dev[:, 1, None] + edge_stress_vol[:, 1, None],
        )
        boundary_force_y = jnp.sum(edge_lengths * jnp.mean(sigma_yy_q, axis=1))
        internal_force = u_matvec(u, element_stiffness(s, u))
        reaction_force_y = jnp.sum(internal_force[load_y_dofs])
        return frac, -boundary_force_y, -reaction_force_y, strain

    return (
        solve_displacement, newton_increment, residual_norm, rel_error,
        update_energy, update_fatigue, diagnostics, project_qp,
    )


def solve_phase(s, history, kernels, maxiter: int, tolerance: float):
    _, increment, norm_fn = kernels[0], kernels[1], kernels[2]
    initial = float(jax.device_get(norm_fn(s, history)))
    residual = initial
    linear_residual = np.inf
    for iteration in range(1, maxiter + 1):
        delta, _, lin = increment(s, history)
        jax.block_until_ready(delta)
        linear_residual = float(jax.device_get(lin))
        step = 1.0
        accepted = s + delta
        accepted_norm = float(jax.device_get(norm_fn(accepted, history)))
        while accepted_norm > residual and step > 1.0 / 1024.0:
            step *= 0.5
            accepted = s + step * delta
            accepted_norm = float(jax.device_get(norm_fn(accepted, history)))
        s, residual = accepted, accepted_norm
        if residual <= tolerance * max(1.0, initial):
            break
    return s, iteration, residual, linear_residual


def require_finite(label: str, step: int, inner: int | None = None, **values) -> None:
    location = f"step {step}" + (f", stagger {inner}" if inner is not None else "")
    for name, value in values.items():
        finite = bool(jax.device_get(jnp.all(jnp.isfinite(jnp.asarray(value)))))
        if not finite:
            raise FloatingPointError(f"nonfinite {label}.{name} at {location}")


CSV_FIELDS = (
    "step", "time", "cycle", "u_app", "load", "x_tip", "y_tip", "delta_a",
    "alpha_bar", "f_degrad", "frac_energy", "inner_iterations", "rel_error",
    "phase_newton_iterations", "u_linear_residual", "phase_linear_residual",
    "step_time_s", "elapsed_s",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, step: int, u, s, energy, alpha, alpha_bar, rows) -> None:
    np.savez_compressed(
        path, step=np.asarray(step), u=np.asarray(jax.device_get(u)),
        s=np.asarray(jax.device_get(s)), energy=np.asarray(jax.device_get(energy)),
        alpha=np.asarray(jax.device_get(alpha)),
        alpha_bar=np.asarray(jax.device_get(alpha_bar)),
        rows_json=np.asarray(json.dumps(rows)),
    )


TPB_SNAPSHOT_MILESTONES = (0.05, 0.10, 0.20, 0.40, 0.60)


def save_snapshot(
    path: Path, nodes, tri, u, s, energy, alpha, alpha_bar, fdeg, strain,
    row: dict, milestone: str, milestone_value: float,
) -> None:
    """Atomically write one plotting state without touching solver state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    displacement = np.asarray(jax.device_get(u)).reshape(-1, 2)
    phase = np.asarray(jax.device_get(s))
    np.savez_compressed(
        temporary,
        nodes=np.asarray(nodes), triangles=np.asarray(tri),
        displacement=displacement, phase=phase, damage=1.0 - phase,
        history_energy=np.asarray(jax.device_get(energy)),
        fatigue_alpha=np.asarray(jax.device_get(alpha)),
        fatigue_alpha_bar=np.asarray(jax.device_get(alpha_bar)),
        fatigue_degradation=np.asarray(jax.device_get(fdeg)),
        strain=np.asarray(strain),
        step=np.asarray(row["step"]), cycle=np.asarray(row["cycle"]),
        time=np.asarray(row["time"]), applied_displacement=np.asarray(row["u_app"]),
        load=np.asarray(row["load"]), crack_tip=np.asarray([row["x_tip"], row["y_tip"]]),
        delta_a_mm=np.asarray(row["delta_a"]),
        milestone=np.asarray(milestone), milestone_value_mm=np.asarray(milestone_value),
    )
    temporary.replace(path)


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    return (
        int(data["step"]), data["u"], data["s"], data["energy"], data["alpha"],
        data["alpha_bar"], json.loads(str(data["rows_json"])),
    )


def save_vtu(path: Path, nodes, tri, u, s, strain, fdeg, alpha_bar) -> None:
    import meshio
    displacement = np.asarray(u).reshape(-1, 2)
    meshio.write_points_cells(
        str(path), np.column_stack([nodes, np.zeros(len(nodes))]), [("triangle", tri)],
        point_data={
            "uh": np.column_stack([displacement, np.zeros(len(nodes))]),
            "s": np.asarray(s), "damage": 1.0 - np.asarray(s),
        },
        cell_data={
            "epsi": [np.column_stack([np.asarray(strain), np.zeros((len(tri), 3))])],
            "Fdeg": [np.mean(np.asarray(fdeg), axis=1)],
            "alphabar": [np.mean(np.asarray(alpha_bar), axis=1)],
        },
    )


def save_plots(outdir: Path, nodes, tri, s, rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plots skipped: matplotlib is unavailable")
        return
    fig, ax = plt.subplots(figsize=(9, 3))
    image = ax.tripcolor(nodes[:, 0], nodes[:, 1], tri, 1.0 - np.asarray(s), shading="gouraud", cmap="jet", vmin=0, vmax=1)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Length-scale insensitive TPB fatigue, AT1, R=0.0")
    fig.colorbar(image, ax=ax, label="d"); fig.tight_layout()
    fig.savefig(outdir / "damage_final.png", dpi=200); plt.close(fig)
    if rows:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([r["cycle"] for r in rows], [r["delta_a"] for r in rows])
        ax.set(xlabel="N", ylabel="Δa (mm)", title="TPB fatigue crack growth (AT1)")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(outdir / "a_N_curve.png", dpi=220); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([r["u_app"] for r in rows], [r["load"] for r in rows])
        ax.set(xlabel="u (mm)", ylabel="Load", title="TPB load-displacement (AT1)")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(outdir / "F_u_curve.png", dpi=220); plt.close(fig)


def main() -> int:
    args = parser().parse_args()
    if args.cycles < 1 or args.steps_per_cycle < 1 or args.inner_max < 1:
        raise ValueError("cycle, step, and iteration counts must be positive")
    if args.mesh_scale <= 0.0 or args.linear_tol <= 0.0 or args.rel_tol <= 0.0:
        raise ValueError("mesh scale and tolerances must be positive")
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("FP64 could not be enabled")
    devices = jax.devices()
    if args.platform == "gpu" and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU requested but available devices are {devices}")
    selected_device_slug = device_slug(devices[0])
    if args.outdir is None:
        root = Path.cwd()
        args.outdir = (
            root / "outputs" / f"results_{selected_device_slug}"
            / "elastic_fatigue_TPB_AT1" / "jax"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_file = args.mesh or args.outdir / "BeamWithNotchSymThreePtBending.msh"
    if args.mesh is None and not mesh_file.is_file():
        create_mesh(mesh_file, args.mesh_scale)
    _, nodes, tri, areas, left, right, load_nodes, load_edges = load_mesh(mesh_file)
    bmat, dofs, grad, mass = preprocess(nodes, tri, areas)
    load_cells = boundary_cells(tri, load_edges)
    diagnostic_cell, diagnostic_weights = point_qp_weights(nodes, tri)

    ndof = 2 * len(nodes)
    free_mask = np.ones(ndof)
    free_mask[2 * left + 1] = 0.0
    free_mask[2 * right] = 0.0
    free_mask[2 * right + 1] = 0.0
    free_mask[2 * load_nodes + 1] = 0.0
    load_y_dofs = 2 * load_nodes + 1
    kernels = make_kernels(
        tri_np=tri, dofs_np=dofs, bmat_np=bmat, grad_np=grad, mass_np=mass,
        areas_np=areas,
        nodes_np=nodes, load_edges_np=load_edges, load_cells_np=load_cells,
        free_mask_np=free_mask, linear_tol=args.linear_tol,
        linear_maxiter=args.linear_maxiter,
    )
    (
        solve_u, _, _, relative_error, update_energy, update_fatigue,
        diagnostics, project_qp,
    ) = kernels

    checkpoint = args.outdir / "checkpoint_latest.npz"
    if args.resume and checkpoint.is_file():
        start, u0, s0, energy0, alpha0, alpha_bar0, rows = load_checkpoint(checkpoint)
        print(f"  resuming after step {start}")
    else:
        start = 0
        u0, s0 = np.zeros(ndof), np.ones(len(nodes))
        energy0 = np.full((len(tri), 3), PSI_CRIT)
        # Gridap's conformity=:L2 projection stores element-local P1 fields.
        alpha0 = np.zeros((len(tri), 3))
        alpha_bar0 = np.zeros((len(tri), 3))
        rows = [{
            "step": 0, "time": 0.0, "cycle": 0.0, "u_app": 0.0, "load": 0.0,
            "x_tip": L / 2, "y_tip": CH, "delta_a": 0.0, "alpha_bar": 0.0,
            "f_degrad": 1.0, "frac_energy": 0.0, "inner_iterations": 0,
            "rel_error": 0.0, "phase_newton_iterations": 0,
            "u_linear_residual": 0.0, "phase_linear_residual": 0.0,
            "step_time_s": 0.0, "elapsed_s": 0.0,
        }]
    u, s = jnp.asarray(u0), jnp.asarray(s0)
    energy, alpha = jnp.asarray(energy0), jnp.asarray(alpha0)
    alpha_bar = jnp.asarray(alpha_bar0)
    if alpha.shape != (len(tri), 3) or alpha_bar.shape != (len(tri), 3):
        raise RuntimeError(
            "checkpoint fatigue fields are incompatible with the canonical "
            "element-local L2 P1 state; start without --resume"
        )
    alpha_bar_q = jnp.einsum("qa,ea->eq", jnp.asarray(QP_N), alpha_bar)
    fdeg = jnp.where(
        alpha_bar_q >= ALPHA_T,
        (2.0 * ALPHA_T / (alpha_bar_q + ALPHA_T)) ** 2,
        1.0,
    )

    total_steps = args.cycles * args.steps_per_cycle
    final_step = total_steps if args.max_steps <= 0 else min(total_steps, start + args.max_steps)
    period = T_TOTAL / args.cycles
    dt = period / args.steps_per_cycle
    u_min = args.r_ratio * args.u_max
    print("=" * 78)
    print("Length-scale insensitive TPB fatigue | standalone JAX AT1")
    print(f"  JAX {jax.__version__}; Python {platform.python_version()}; devices={devices}")
    print(f"  FP64; E={E_MAT}; nu={NU_MAT}; Gc={GC}; ls={LS}; eta={ETA}")
    print(f"  sigma_c={SIGMA_C}; m={M_PARAM}; psiCrit={PSI_CRIT}; kf={KF}; alphaT={ALPHA_T}")
    print(f"  mesh={len(nodes)} nodes, {len(tri)} triangles; cycles={args.cycles} x {args.steps_per_cycle}")
    if args.mesh_scale != 1.0:
        print(f"  WARNING: mesh-scale={args.mesh_scale:g}; this is not the reference mesh")
    print("=" * 78)

    run_start = time.perf_counter()
    elapsed_offset = float(rows[-1]["elapsed_s"])
    step_times: list[float] = []
    last_strain = np.zeros((len(tri), 3))
    snapshot_dir = args.outdir / "snapshots"
    snapshot_io_s = 0.0
    for step in range(start + 1, final_step + 1):
        step_start = time.perf_counter()
        t = step * dt
        amp = args.u_max - u_min
        v_raw = u_min + abs(4.0 * amp / period * abs(((t - period / 4.0) % period) - period / 2.0) - amp)
        v_app = round(v_raw, 6)
        boundary = np.zeros(ndof)
        boundary[load_y_dofs] = -v_app
        boundary_j = jnp.asarray(boundary)
        alpha_bar_q = jnp.einsum(
        "qa,ea->eq", jnp.asarray(QP_N), alpha_bar
        )
        fdeg = jnp.where(
            alpha_bar_q >= ALPHA_T,
            (2.0 * ALPHA_T / (alpha_bar_q + ALPHA_T)) ** 2,
            1.0,
        )

        phase_newton_iterations = 0
        phase_linear_residual = np.inf
        u_linear_residual = np.inf
        rel = np.inf
        for inner in range(1, args.inner_max + 1):
            # Gridap's project_field is called at every stagger iteration.
            energy_p1, projection_residual = project_qp(
                energy, jnp.full((len(tri), 3), PSI_CRIT)
            )
            jax.block_until_ready(energy_p1)
            if float(jax.device_get(projection_residual)) > args.linear_tol * 10.0:
                raise RuntimeError(
                    "history L2 projection failed its linear residual check"
                )
            rel = float(jax.device_get(relative_error(s, energy_p1)))
            s, newton_it, _, phase_linear_residual = solve_phase(
                s, energy_p1, kernels, args.newton_max, args.newton_tol
            )
            require_finite(
                "phase", step, inner, s=s, rel_error=rel,
                linear_residual=phase_linear_residual,
            )
            phase_newton_iterations += newton_it
            u, u_res = solve_u(s, u, boundary_j, u)
            strain, psi_org, energy = update_energy(u, energy, fdeg)
            jax.block_until_ready(energy)
            u_linear_residual = float(jax.device_get(u_res))
            require_finite(
                "displacement", step, inner, u=u,
                linear_residual=u_linear_residual, strain=strain,
                psi_org=psi_org, history=energy,
            )
            if rel < args.rel_tol:
                break

        s_host = np.asarray(jax.device_get(s))
        damaged = np.where(s_host <= args.damage_tol)[0]
        if damaged.size:
            x_values = nodes[damaged, 0]
            selected = damaged[int(np.argmax(x_values))]
            x_tip, y_tip = map(float, nodes[selected])
        else:
            x_tip, y_tip = L / 2.0, CH

        old_alpha_point = float(
            np.asarray(jax.device_get(alpha[diagnostic_cell])) @ diagnostic_weights
        )
        old_alpha_bar_point = float(
            np.asarray(jax.device_get(alpha_bar[diagnostic_cell]))
            @ diagnostic_weights
        )
        s_point = float(s_host[tri[diagnostic_cell]] @ diagnostic_weights)
        g_point = s_point**2 / (
            s_point**2
            + M_PARAM * (1.0 - s_point) * (0.5 + 0.5 * s_point)
        )
        alpha_point = g_point * float(jax.device_get(psi_org[diagnostic_cell]))
        alpha_value = (
            old_alpha_bar_point + abs(alpha_point - old_alpha_point)
            if alpha_point >= old_alpha_point else old_alpha_bar_point
        )
        alpha_q, alpha_bar_q = update_fatigue(
            s, psi_org, alpha, alpha_bar
        )
        alpha, alpha_projection_residual = project_qp(alpha_q, alpha)
        alpha_bar, alpha_bar_projection_residual = project_qp(
            alpha_bar_q, alpha_bar
        )
        jax.block_until_ready(alpha_bar)
        require_finite(
            "fatigue", step, alpha=alpha, alpha_bar=alpha_bar,
            alpha_projection_residual=alpha_projection_residual,
            alpha_bar_projection_residual=alpha_bar_projection_residual,
        )
        if max(
            float(jax.device_get(alpha_projection_residual)),
            float(jax.device_get(alpha_bar_projection_residual)),
        ) > args.linear_tol * 10.0:
            raise RuntimeError("fatigue L2 projection failed its linear residual check")
        alpha_bar_q = jnp.einsum(
            "qa,ea->eq", jnp.asarray(QP_N), alpha_bar
        )
        fdeg = jnp.where(
            alpha_bar_q >= ALPHA_T,
            (2.0 * ALPHA_T / (alpha_bar_q + ALPHA_T)) ** 2,
            1.0,
        )
        frac, load, reaction_load, strain = diagnostics(s, u, fdeg)
        jax.block_until_ready(load)
        alpha_bar_host = np.asarray(jax.device_get(alpha_bar))
        fdeg_host = np.asarray(jax.device_get(fdeg))
        projected_alpha_bar_value = float(
            alpha_bar_host[diagnostic_cell] @ diagnostic_weights
        )
        fdeg_value = (
            (2.0 * ALPHA_T / (projected_alpha_bar_value + ALPHA_T)) ** 2
            if projected_alpha_bar_value >= ALPHA_T else 1.0
        )
        frac_value = float(jax.device_get(frac))
        load_value = float(jax.device_get(reaction_load))
        require_finite(
            "diagnostics", step, frac_energy=frac_value, load=load_value,
            f_degrad=fdeg, strain=strain,
        )
        last_strain = np.asarray(jax.device_get(strain))
        step_time = time.perf_counter() - step_start
        elapsed = elapsed_offset + time.perf_counter() - run_start - snapshot_io_s
        step_times.append(step_time)
        row = {
            "step": step, "time": t, "cycle": step / args.steps_per_cycle,
            "u_app": v_app, "load": load_value, "x_tip": x_tip, "y_tip": y_tip,
            "delta_a": y_tip - CH, "alpha_bar": alpha_value,
            "f_degrad": fdeg_value, "frac_energy": frac_value,
            "inner_iterations": inner, "rel_error": rel,
            "phase_newton_iterations": phase_newton_iterations,
            "u_linear_residual": u_linear_residual,
            "phase_linear_residual": phase_linear_residual,
            "step_time_s": step_time, "elapsed_s": elapsed,
        }
        rows.append(row)

        if args.snapshots:
            for milestone in TPB_SNAPSHOT_MILESTONES:
                snapshot = snapshot_dir / f"delta_a_{milestone:0.2f}mm.npz"
                if row["delta_a"] >= milestone and not snapshot.exists():
                    snapshot_t0 = time.perf_counter()
                    save_snapshot(
                        snapshot, nodes, tri, u, s, energy, alpha, alpha_bar,
                        fdeg, last_strain, row, "delta_a", milestone,
                    )
                    snapshot_io_s += time.perf_counter() - snapshot_t0

        completed_cycle = step % args.steps_per_cycle == 0
        cycle = step // args.steps_per_cycle
        if step <= 2 or step % args.print_every == 0 or step == final_step:
            print(
                f"  step {step:4d}/{total_steps} N={step / args.steps_per_cycle:7.1f} "
                f"u={v_app:.6f} F={load_value:.6f} da={y_tip - CH:.4f} "
                f"inner={inner} rel={rel:.2e} dt={step_time:.3f}s"
            )
        if args.vtk and completed_cycle and (
            cycle == 1 or cycle == args.cycles
            or (args.save_cycle_interval > 0 and cycle % args.save_cycle_interval == 0)
        ):
            save_vtu(
                args.outdir / f"cycle_{cycle:04d}.vtu", nodes, tri,
                np.asarray(jax.device_get(u)), s_host, last_strain, fdeg_host,
                alpha_bar_host,
            )
        if args.checkpoint_every > 0 and (
            step % args.checkpoint_every == 0 or step == final_step
        ):
            write_csv(args.outdir / "history.csv", rows)
            save_checkpoint(checkpoint, step, u, s, energy, alpha, alpha_bar, rows)

    total_time = time.perf_counter() - run_start - snapshot_io_s
    write_csv(args.outdir / "history.csv", rows)
    save_checkpoint(checkpoint, final_step, u, s, energy, alpha, alpha_bar, rows)
    final_snapshot = snapshot_dir / "final.npz"
    if args.snapshots and (step_times or not final_snapshot.exists()):
        save_snapshot(
            final_snapshot, nodes, tri, u, s, energy, alpha,
            alpha_bar, fdeg, last_strain, rows[-1], "final",
            float(rows[-1]["delta_a"]),
        )
    if args.vtk:
        save_vtu(
            args.outdir / "result_final.vtu", nodes, tri, np.asarray(jax.device_get(u)),
            np.asarray(jax.device_get(s)), last_strain,
            np.asarray(jax.device_get(fdeg)), np.asarray(jax.device_get(alpha_bar)),
        )
    if args.plot:
        save_plots(args.outdir, nodes, tri, np.asarray(jax.device_get(s)), rows)
    timing = {
        "jax_version": jax.__version__, "devices": [str(device) for device in devices],
        "device_strings": [str(device) for device in devices],
        "device_kinds": [device.device_kind for device in devices],
        "device_slug": selected_device_slug,
        "dtype": "float64", "nodes": len(nodes), "triangles": len(tri),
        "steps_this_run": len(step_times), "total_s": total_time,
        "first_step_compile_and_solve_s": step_times[0] if step_times else None,
        "post_first_step_mean_s": float(np.mean(step_times[1:])) if len(step_times) > 1 else None,
        "median_step_s": float(np.median(step_times)) if step_times else None,
        "reference_defaults": vars(args) | {"outdir": str(args.outdir), "mesh": str(mesh_file)},
    }
    with (args.outdir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, indent=2)
    print(f"completed {len(step_times)} steps in {total_time:.3f}s; outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
