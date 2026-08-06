#!/usr/bin/env python3
"""
Standalone JAX/GPU solver for the Kalthoff--Winkler dynamic fracture test.

Physics
-------
* 2-D plane strain, P1 triangular finite elements
* AT2 phase-field fracture with Miehe spectral tension/compression split
* explicit velocity-Verlet mechanics with a lumped mass
* matrix-free preconditioned CG phase-field solve
* irreversible history field and pointwise damage projection
* prescribed impact-velocity ramp on the left impact segment

Geometry
--------
The script constructs its own 100 x 100 mm half-plate geometry with Gmsh.
The bottom edge is the symmetry plane, a 50 mm edge notch is centred at
y = 25 mm, and the left edge below the notch is the impact segment.  No mesh
from another project is read or required.  A broad graded corridor resolves
the expected approximately 70-degree crack without embedding a crack path.

The default h_crack=0.06 mm mesh is intended for a large A100-class server
run.  Use --h-crack 0.5 --steps 5 for a local smoke test.

Server example
--------------
  python kalthoff_winkler_dynamic_jax_gpu.py --platform gpu

Five-level mesh-density validation (about 58k--988k nodes)
----------------------------------------------------------
  python kalthoff_winkler_dynamic_jax_gpu.py \\
    --platform gpu --mesh-sweep --resume

Local smoke test
----------------
  python kalthoff_winkler_dynamic_jax_gpu.py \\
    --platform gpu --h-crack 0.5 --steps 5 --no-plot

Dependencies
------------
  jax (CUDA build), numpy, gmsh, meshio; matplotlib is optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import subprocess
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
        "JAX_PLATFORMS",
        "cuda" if _PLATFORM_REQUEST == "gpu" else _PLATFORM_REQUEST,
    )
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402

try:
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from jax import lax  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "JAX is required. For an NVIDIA GPU under Linux/WSL install, e.g.:\n"
        "  python -m pip install 'jax[cuda13]' gmsh meshio\n"
        f"Original import error: {exc}"
    ) from exc


# =============================================================================
# Benchmark definition (mm, tonne, second, MPa)
# =============================================================================

W = 100.0
HEIGHT = 100.0
NOTCH_LENGTH = 50.0
NOTCH_Y = 25.0
NOTCH_HALF_WIDTH = 0.05
EXPECTED_CRACK_ANGLE_DEG = 70.0

E = 190_000.0
NU = 0.30
GC = 22.13
L0 = 0.195
RHO = 8.0e-9
ETA = 1.0e-7

LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))

DEFAULT_H_CRACK = 0.06
DEFAULT_H_COARSE = 5.0
DEFAULT_DT_SAFETY = 0.8
DEFAULT_TOTAL_TIME_US = 100.0
DEFAULT_IMPACT_VELOCITY_M_S = 16.5
DEFAULT_RAMP_TIME_US = 1.0
DEFAULT_CG_TOL = 1.0e-8
DEFAULT_CG_ATOL = 1.0e-10
DEFAULT_CG_MAXITER = 1000
DEFAULT_TIMING_BLOCK = 50
DEFAULT_HISTORY_EVERY = 100
DEFAULT_CHECKPOINT_EVERY = 5000
DEFAULT_SNAPSHOT_TIMES_US = "25,50,75,100"
DEFAULT_SWEEP_H_VALUES = "0.12,0.085,0.06,0.042,0.028"

PHYSICAL_TAGS = {
    "plate": 1,
    "bottom_symmetry": 11,
    "right": 12,
    "top": 13,
    "left_top": 14,
    "notch": 15,
    "left_impact": 16,
}


def script_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code" else script_dir


def device_slug(device) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", device.device_kind.lower()).strip("_")
    return slug or device.platform.lower()


def mesh_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_snapshot_times(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    values = tuple(sorted({float(item.strip()) for item in text.split(",") if item.strip()}))
    if any(value <= 0.0 for value in values):
        raise ValueError("snapshot times must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone JAX/GPU Kalthoff-Winkler dynamic fracture solver"
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="existing Gmsh mesh; default is a self-generated mesh in --outdir",
    )
    parser.add_argument("--regenerate-mesh", action="store_true")
    parser.add_argument("--h-crack", type=float, default=DEFAULT_H_CRACK)
    parser.add_argument("--h-coarse", type=float, default=DEFAULT_H_COARSE)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="result directory (device-aware default under outputs/)",
    )
    parser.add_argument(
        "--mesh-sweep",
        action="store_true",
        help="run independent full simulations over --sweep-h-values",
    )
    parser.add_argument(
        "--sweep-h-values",
        default=DEFAULT_SWEEP_H_VALUES,
        help=(
            "comma-separated fine mesh sizes for --mesh-sweep "
            "(default targets approximately 5e4 to 1e6 nodes)"
        ),
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help="root directory for mesh-sweep cases and aggregate results",
    )
    parser.add_argument(
        "--sweep-precheck-only",
        action="store_true",
        help="generate/count all sweep meshes without dynamic solves",
    )
    parser.add_argument(
        "--sweep-rerun",
        action="store_true",
        help="rerun completed sweep cases instead of skipping them",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="generate/load the mesh, write metadata, and stop before importing state",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the physical step count (intended for smoke tests)",
    )
    parser.add_argument("--total-time-us", type=float, default=DEFAULT_TOTAL_TIME_US)
    parser.add_argument(
        "--impact-velocity-m-s",
        type=float,
        default=DEFAULT_IMPACT_VELOCITY_M_S,
    )
    parser.add_argument("--ramp-time-us", type=float, default=DEFAULT_RAMP_TIME_US)
    parser.add_argument("--dt-safety", type=float, default=DEFAULT_DT_SAFETY)
    parser.add_argument("--damage-every", type=int, default=1)
    parser.add_argument("--cg-tol", type=float, default=DEFAULT_CG_TOL)
    parser.add_argument(
        "--cg-atol",
        type=float,
        default=DEFAULT_CG_ATOL,
        help="absolute CG tolerance for the near-zero initial damage RHS",
    )
    parser.add_argument("--cg-maxiter", type=int, default=DEFAULT_CG_MAXITER)
    parser.add_argument("--timing-block", type=int, default=DEFAULT_TIMING_BLOCK)
    parser.add_argument("--history-every", type=int, default=DEFAULT_HISTORY_EVERY)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="0 disables intermediate checkpoints; a final checkpoint is always written",
    )
    parser.add_argument(
        "--snapshot-times-us",
        default=DEFAULT_SNAPSHOT_TIMES_US,
        help="comma-separated physical times for compressed NPZ states",
    )
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--vtk",
        action="store_true",
        help="write final VTU (off by default to limit large-mesh I/O)",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="write final damage/history summary plot (default)",
    )
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


# =============================================================================
# Self-contained Gmsh geometry
# =============================================================================

def expected_crack_endpoint() -> tuple[float, float]:
    angle = math.radians(EXPECTED_CRACK_ANGLE_DEG)
    length = (HEIGHT - NOTCH_Y) / math.sin(angle)
    return (
        NOTCH_LENGTH + length * math.cos(angle),
        HEIGHT,
    )


def generate_mesh(mesh_file: Path, h_crack: float, h_coarse: float) -> None:
    try:
        import gmsh
    except ImportError as exc:
        raise SystemExit("gmsh is required: python -m pip install gmsh") from exc

    if h_crack <= 0.0 or h_coarse <= 0.0 or h_crack >= h_coarse:
        raise ValueError("require 0 < h_crack < h_coarse")

    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("kalthoff_winkler_half_plate")
        geo = gmsh.model.geo

        p1 = geo.addPoint(0.0, 0.0, 0.0, h_coarse)
        p2 = geo.addPoint(W, 0.0, 0.0, h_coarse)
        p3 = geo.addPoint(W, HEIGHT, 0.0, h_coarse)
        p4 = geo.addPoint(0.0, HEIGHT, 0.0, h_coarse)
        p5 = geo.addPoint(0.0, NOTCH_Y + NOTCH_HALF_WIDTH, 0.0, h_crack)
        p6 = geo.addPoint(
            NOTCH_LENGTH, NOTCH_Y + NOTCH_HALF_WIDTH, 0.0, h_crack
        )
        p7 = geo.addPoint(
            NOTCH_LENGTH, NOTCH_Y - NOTCH_HALF_WIDTH, 0.0, h_crack
        )
        p8 = geo.addPoint(0.0, NOTCH_Y - NOTCH_HALF_WIDTH, 0.0, h_crack)

        bottom = geo.addLine(p1, p2)
        right = geo.addLine(p2, p3)
        top = geo.addLine(p3, p4)
        left_top = geo.addLine(p4, p5)
        notch_upper = geo.addLine(p5, p6)
        notch_tip = geo.addLine(p6, p7)
        notch_lower = geo.addLine(p7, p8)
        left_impact = geo.addLine(p8, p1)
        loop = geo.addCurveLoop(
            [
                bottom,
                right,
                top,
                left_top,
                notch_upper,
                notch_tip,
                notch_lower,
                left_impact,
            ]
        )
        surface = geo.addPlaneSurface([loop])

        end_x, end_y = expected_crack_endpoint()
        guide_points = [
            geo.addPoint(
                NOTCH_LENGTH + fraction * (end_x - NOTCH_LENGTH),
                NOTCH_Y + fraction * (end_y - NOTCH_Y),
                0.0,
                h_crack,
            )
            for fraction in np.linspace(0.0, 1.0, 33)
        ]
        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], PHYSICAL_TAGS["plate"])
        gmsh.model.setPhysicalName(2, PHYSICAL_TAGS["plate"], "plate")
        boundary_groups = {
            "bottom_symmetry": [bottom],
            "right": [right],
            "top": [top],
            "left_top": [left_top],
            "notch": [notch_upper, notch_tip, notch_lower],
            "left_impact": [left_impact],
        }
        for name, curves in boundary_groups.items():
            gmsh.model.addPhysicalGroup(1, curves, PHYSICAL_TAGS[name])
            gmsh.model.setPhysicalName(1, PHYSICAL_TAGS[name], name)

        corridor_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            corridor_distance, "PointsList", guide_points
        )
        gmsh.model.mesh.field.setNumber(corridor_distance, "Sampling", 100)
        corridor = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(corridor, "InField", corridor_distance)
        gmsh.model.mesh.field.setNumber(corridor, "SizeMin", h_crack)
        gmsh.model.mesh.field.setNumber(corridor, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(corridor, "DistMin", 4.0)
        gmsh.model.mesh.field.setNumber(corridor, "DistMax", 15.0)

        tip_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(tip_distance, "PointsList", [p6, p7])
        tip = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tip, "InField", tip_distance)
        gmsh.model.mesh.field.setNumber(tip, "SizeMin", h_crack)
        gmsh.model.mesh.field.setNumber(tip, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(tip, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(tip, "DistMax", 10.0)

        background = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            background, "FieldsList", [corridor, tip]
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(background)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.finalize()


# =============================================================================
# Mesh import and preprocessing
# =============================================================================

def load_mesh(mesh_file: Path):
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("meshio is required: python -m pip install meshio") from exc

    mesh = meshio.read(str(mesh_file))
    nodes = np.asarray(mesh.points[:, :2], dtype=np.float64)
    triangle_blocks = [
        np.asarray(block.data, dtype=np.int64)
        for block in mesh.cells
        if block.type == "triangle"
    ]
    if not triangle_blocks:
        raise RuntimeError("mesh contains no P1 triangles")
    triangles = np.concatenate(triangle_blocks, axis=0)

    boundary_nodes: dict[str, set[int]] = {
        name: set()
        for name in (
            "bottom_symmetry",
            "right",
            "top",
            "left_top",
            "notch",
            "left_impact",
        )
    }
    tag_to_name = {
        PHYSICAL_TAGS[name]: name for name in boundary_nodes
    }
    physical_data = mesh.cell_data.get("gmsh:physical")
    if physical_data is not None:
        for block_index, block in enumerate(mesh.cells):
            if block.type != "line":
                continue
            tags = np.asarray(physical_data[block_index]).reshape(-1)
            for edge, tag in zip(np.asarray(block.data), tags):
                name = tag_to_name.get(int(tag))
                if name is not None:
                    boundary_nodes[name].update(int(value) for value in edge)

    tolerance = 1.0e-8
    if not boundary_nodes["bottom_symmetry"]:
        boundary_nodes["bottom_symmetry"].update(
            np.flatnonzero(np.abs(nodes[:, 1]) <= tolerance).tolist()
        )
    if not boundary_nodes["left_impact"]:
        boundary_nodes["left_impact"].update(
            np.flatnonzero(
                (np.abs(nodes[:, 0]) <= tolerance)
                & (nodes[:, 1] <= NOTCH_Y - NOTCH_HALF_WIDTH + tolerance)
            ).tolist()
        )

    xy = nodes[triangles]
    twice_area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    flip = twice_area < 0.0
    if np.any(flip):
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        xy = nodes[triangles]
        twice_area = (
            (xy[:, 1, 0] - xy[:, 0, 0])
            * (xy[:, 2, 1] - xy[:, 0, 1])
            - (xy[:, 2, 0] - xy[:, 0, 0])
            * (xy[:, 1, 1] - xy[:, 0, 1])
        )
    areas = 0.5 * twice_area
    if np.any(areas <= 0.0):
        raise RuntimeError("mesh contains non-positive triangle areas")

    arrays = {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in boundary_nodes.items()
    }
    if arrays["bottom_symmetry"].size == 0 or arrays["left_impact"].size == 0:
        raise RuntimeError("bottom symmetry or impact boundary is empty")
    return nodes, triangles, areas, arrays


def preprocess_mesh(
    nodes: np.ndarray,
    triangles: np.ndarray,
    areas: np.ndarray,
):
    x = nodes[:, 0]
    y = nodes[:, 1]
    i0, i1, i2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    grad_x = np.stack(
        [y[i1] - y[i2], y[i2] - y[i0], y[i0] - y[i1]], axis=1
    ) / (2.0 * areas[:, None])
    grad_y = np.stack(
        [x[i2] - x[i1], x[i0] - x[i2], x[i1] - x[i0]], axis=1
    ) / (2.0 * areas[:, None])
    gradients = np.stack([grad_x, grad_y], axis=2)

    edge_01 = np.linalg.norm(nodes[i1] - nodes[i0], axis=1)
    edge_12 = np.linalg.norm(nodes[i2] - nodes[i1], axis=1)
    edge_20 = np.linalg.norm(nodes[i0] - nodes[i2], axis=1)
    perimeter = edge_01 + edge_12 + edge_20
    incircle_diameter = 4.0 * areas / perimeter
    h_min = float(np.min(incircle_diameter))
    longest_edge = np.maximum(np.maximum(edge_01, edge_12), edge_20)
    minimum_altitude = 2.0 * areas / longest_edge
    aspect_ratio = longest_edge / minimum_altitude

    nodal_mass = np.zeros(nodes.shape[0], dtype=np.float64)
    np.add.at(
        nodal_mass,
        triangles.reshape(-1),
        np.repeat(RHO * areas / 3.0, 3),
    )
    if np.any(nodal_mass <= 0.0):
        raise RuntimeError("lumped mass contains non-positive entries")
    return (
        gradients,
        nodal_mass,
        h_min,
        incircle_diameter,
        minimum_altitude,
        aspect_ratio,
    )


def mesh_metadata(
    nodes: np.ndarray,
    triangles: np.ndarray,
    areas: np.ndarray,
    incircle_diameter: np.ndarray,
    minimum_altitude: np.ndarray,
    aspect_ratio: np.ndarray,
    boundaries: dict[str, np.ndarray],
    mesh_file: Path,
    h_crack: float,
    h_coarse: float,
) -> dict[str, object]:
    return {
        "mesh_file": str(mesh_file),
        "geometry": {
            "width_mm": W,
            "height_mm": HEIGHT,
            "notch_length_mm": NOTCH_LENGTH,
            "notch_y_mm": NOTCH_Y,
            "notch_total_width_mm": 2.0 * NOTCH_HALF_WIDTH,
            "expected_crack_angle_deg": EXPECTED_CRACK_ANGLE_DEG,
        },
        "mesh_controls": {
            "h_crack_mm": h_crack,
            "h_coarse_mm": h_coarse,
            "corridor_dist_min_mm": 4.0,
            "corridor_dist_max_mm": 15.0,
        },
        "mesh": {
            "nodes": int(nodes.shape[0]),
            "triangles": int(triangles.shape[0]),
            "area_mm2": float(np.sum(areas)),
            "h_incircle_min_mm": float(np.min(incircle_diameter)),
            "h_incircle_p05_mm": float(np.percentile(incircle_diameter, 5.0)),
            "h_incircle_median_mm": float(np.median(incircle_diameter)),
            "h_incircle_p95_mm": float(np.percentile(incircle_diameter, 95.0)),
            "minimum_altitude_mm": float(np.min(minimum_altitude)),
            "aspect_ratio_p95": float(np.percentile(aspect_ratio, 95.0)),
            "aspect_ratio_max": float(np.max(aspect_ratio)),
            "l0_over_h_crack": float(L0 / h_crack),
        },
        "boundary_node_counts": {
            name: int(values.size) for name, values in boundaries.items()
        },
    }


# =============================================================================
# Matrix-free JAX operators
# =============================================================================

def make_solver(
    *,
    nodes_np: np.ndarray,
    triangles_np: np.ndarray,
    areas_np: np.ndarray,
    gradients_np: np.ndarray,
    mass_np: np.ndarray,
    bottom_np: np.ndarray,
    impact_np: np.ndarray,
    dtype,
    dt: float,
    impact_velocity_mm_s: float,
    ramp_time_s: float,
    cg_tol: float,
    cg_atol: float,
    cg_maxiter: int,
):
    triangles = jnp.asarray(triangles_np.astype(np.int32, copy=False))
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    areas = jnp.asarray(areas_np, dtype=dtype)
    gradients = jnp.asarray(gradients_np, dtype=dtype)
    mass = jnp.asarray(mass_np, dtype=dtype)
    bottom = jnp.asarray(bottom_np.astype(np.int32, copy=False))
    impact = jnp.asarray(impact_np.astype(np.int32, copy=False))

    nnode = int(nodes_np.shape[0])
    lambda_j = jnp.asarray(LAMBDA, dtype=dtype)
    mu_j = jnp.asarray(MU, dtype=dtype)
    gc_j = jnp.asarray(GC, dtype=dtype)
    l0_j = jnp.asarray(L0, dtype=dtype)
    eta_j = jnp.asarray(ETA, dtype=dtype)
    dt_j = jnp.asarray(dt, dtype=dtype)
    v0_j = jnp.asarray(impact_velocity_mm_s, dtype=dtype)
    ramp_j = jnp.asarray(ramp_time_s, dtype=dtype)
    tiny = jnp.asarray(1.0e-30, dtype=dtype)
    spectral_eps = jnp.asarray(
        1.0e-14 if dtype == jnp.float64 else 1.0e-7, dtype=dtype
    )

    def scatter_nodes(values):
        return jnp.zeros((nnode, 2), dtype=dtype).at[triangles.reshape(-1)].add(
            values.reshape(-1, 2)
        )

    def scatter_scalar(values):
        return jnp.zeros((nnode,), dtype=dtype).at[triangles.reshape(-1)].add(
            values.reshape(-1)
        )

    def degradation(damage):
        values = jnp.clip(damage[triangles], 0.0, 1.0)
        mean_d = jnp.mean(values, axis=1)
        mean_d2 = (
            jnp.sum(values * values, axis=1)
            + values[:, 0] * values[:, 1]
            + values[:, 1] * values[:, 2]
            + values[:, 2] * values[:, 0]
        ) / 6.0
        return 1.0 - 2.0 * mean_d + mean_d2 + eta_j

    def strain_spectral(displacement):
        local_u = displacement[triangles]
        grad_u = jnp.einsum("eai,eaj->eij", local_u, gradients)
        e11 = grad_u[:, 0, 0]
        e22 = grad_u[:, 1, 1]
        e12 = 0.5 * (grad_u[:, 0, 1] + grad_u[:, 1, 0])
        trace = e11 + e22
        mean = 0.5 * trace
        deviator = 0.5 * (e11 - e22)
        radius = jnp.sqrt(deviator * deviator + e12 * e12)
        principal_1 = mean + radius
        principal_2 = mean - radius
        positive_1 = jnp.maximum(principal_1, 0.0)
        positive_2 = jnp.maximum(principal_2, 0.0)
        positive_mean = 0.5 * (positive_1 + positive_2)
        positive_factor = 0.5 * (positive_1 - positive_2) / jnp.maximum(
            radius, spectral_eps
        )
        ep11 = positive_mean + positive_factor * deviator
        ep22 = positive_mean - positive_factor * deviator
        ep12 = positive_factor * e12
        em11 = e11 - ep11
        em22 = e22 - ep22
        em12 = e12 - ep12

        trace_positive = jnp.maximum(trace, 0.0)
        trace_negative = jnp.minimum(trace, 0.0)
        psi_positive = (
            0.5 * lambda_j * trace_positive * trace_positive
            + mu_j * (positive_1 * positive_1 + positive_2 * positive_2)
        )
        psi_negative = (
            0.5 * lambda_j * trace_negative * trace_negative
            + mu_j
            * (
                em11 * em11
                + em22 * em22
                + 2.0 * em12 * em12
            )
        )
        stress_positive = jnp.stack(
            [
                lambda_j * trace_positive + 2.0 * mu_j * ep11,
                2.0 * mu_j * ep12,
                2.0 * mu_j * ep12,
                lambda_j * trace_positive + 2.0 * mu_j * ep22,
            ],
            axis=1,
        ).reshape(-1, 2, 2)
        stress_negative = jnp.stack(
            [
                lambda_j * trace_negative + 2.0 * mu_j * em11,
                2.0 * mu_j * em12,
                2.0 * mu_j * em12,
                lambda_j * trace_negative + 2.0 * mu_j * em22,
            ],
            axis=1,
        ).reshape(-1, 2, 2)
        return psi_positive, psi_negative, stress_positive, stress_negative

    def mechanics(displacement, damage):
        psi_positive, psi_negative, stress_positive, stress_negative = (
            strain_spectral(displacement)
        )
        g = degradation(damage)
        stress = g[:, None, None] * stress_positive + stress_negative
        local_force = areas[:, None, None] * jnp.einsum(
            "eij,eaj->eai", stress, gradients
        )
        return (
            scatter_nodes(local_force),
            psi_positive,
            psi_negative,
            g,
        )

    def phase_matvec(vector, history):
        local = vector[triangles]
        grad_vector = jnp.einsum("ea,eaj->ej", local, gradients)
        diffusion = (
            gc_j
            * l0_j
            * areas[:, None]
            * jnp.einsum("eaj,ej->ea", gradients, grad_vector)
        )
        coefficient = gc_j / l0_j + 2.0 * history
        consistent_mass_action = (
            areas[:, None]
            / 12.0
            * (jnp.sum(local, axis=1)[:, None] + local)
        )
        reaction = coefficient[:, None] * consistent_mass_action
        return scatter_scalar(diffusion + reaction)

    def phase_diagonal(history):
        coefficient = gc_j / l0_j + 2.0 * history
        local_diagonal = (
            gc_j
            * l0_j
            * areas[:, None]
            * jnp.sum(gradients * gradients, axis=2)
            + coefficient[:, None] * areas[:, None] / 6.0
        )
        return jnp.maximum(scatter_scalar(local_diagonal), tiny)

    def phase_rhs(history):
        local_rhs = (
            2.0
            * history
            * areas
            / 3.0
        )[:, None] * jnp.ones((1, 3), dtype=dtype)
        return scatter_scalar(local_rhs)

    def pcg_damage(history, initial):
        rhs = phase_rhs(history)
        diagonal = phase_diagonal(history)

        def operator(value):
            return phase_matvec(value, history)

        residual_0 = rhs - operator(initial)
        z_0 = residual_0 / diagonal
        rz_0 = jnp.vdot(residual_0, z_0)
        rhs_norm = jnp.linalg.norm(rhs)
        absolute_tolerance = jnp.asarray(cg_atol, dtype=dtype)
        tolerance = jnp.maximum(
            jnp.asarray(cg_tol, dtype=dtype) * rhs_norm,
            absolute_tolerance,
        )

        state_0 = (
            jnp.asarray(0, dtype=jnp.int32),
            initial,
            residual_0,
            z_0,
            rz_0,
            jnp.linalg.norm(residual_0),
        )

        def condition(state):
            iteration, _, _, _, _, residual_norm = state
            return (iteration < cg_maxiter) & (residual_norm > tolerance)

        def body(state):
            iteration, solution, residual, direction, rz_old, _ = state
            action = operator(direction)
            denominator = jnp.vdot(direction, action)
            alpha = rz_old / jnp.where(
                jnp.abs(denominator) > tiny, denominator, tiny
            )
            solution_new = solution + alpha * direction
            residual_new = residual - alpha * action
            z_new = residual_new / diagonal
            rz_new = jnp.vdot(residual_new, z_new)
            beta = rz_new / jnp.where(jnp.abs(rz_old) > tiny, rz_old, tiny)
            direction_new = z_new + beta * direction
            return (
                iteration + 1,
                solution_new,
                residual_new,
                direction_new,
                rz_new,
                jnp.linalg.norm(residual_new),
            )

        final = lax.while_loop(condition, body, state_0)
        iterations, solution, _, _, _, residual_norm = final
        # Normalize by the same mixed relative/absolute scale used for
        # convergence. Values <= cg_tol therefore mean the requested effective
        # tolerance was reached even when the physical RHS is nearly zero.
        residual_scale = jnp.maximum(
            rhs_norm,
            absolute_tolerance / jnp.asarray(cg_tol, dtype=dtype),
        )
        relative_residual = residual_norm / jnp.maximum(residual_scale, tiny)
        return solution, iterations, relative_residual

    def impact_kinematics(time_value):
        ramp_fraction = jnp.clip(time_value / ramp_j, 0.0, 1.0)
        velocity = v0_j * ramp_fraction
        displacement_ramp = 0.5 * v0_j * time_value * time_value / ramp_j
        displacement_hold = v0_j * (time_value - 0.5 * ramp_j)
        displacement = jnp.where(
            time_value < ramp_j, displacement_ramp, displacement_hold
        )
        acceleration = jnp.where(
            time_value < ramp_j, v0_j / ramp_j, jnp.asarray(0.0, dtype=dtype)
        )
        return displacement, velocity, acceleration

    def enforce_displacement(displacement, time_value):
        impact_u, _, _ = impact_kinematics(time_value)
        displacement = displacement.at[bottom, 1].set(0.0)
        return displacement.at[impact, 0].set(impact_u)

    def enforce_velocity(velocity, time_value):
        _, impact_v, _ = impact_kinematics(time_value)
        velocity = velocity.at[bottom, 1].set(0.0)
        return velocity.at[impact, 0].set(impact_v)

    def enforce_acceleration(acceleration, time_value):
        _, _, impact_a = impact_kinematics(time_value)
        acceleration = acceleration.at[bottom, 1].set(0.0)
        return acceleration.at[impact, 0].set(impact_a)

    def advance_impl(state, time_value, solve_damage):
        (
            displacement,
            velocity,
            acceleration,
            damage,
            history,
            boundary_work,
            impact_reaction,
        ) = state
        next_time = time_value + dt_j
        impact_u_old, _, _ = impact_kinematics(time_value)
        impact_u_new, _, _ = impact_kinematics(next_time)
        displacement_new = (
            displacement
            + dt_j * velocity
            + 0.5 * dt_j * dt_j * acceleration
        )
        displacement_new = enforce_displacement(displacement_new, next_time)

        _, psi_positive, _, _ = mechanics(displacement_new, damage)
        history_new = jnp.maximum(history, psi_positive)
        if solve_damage:
            damage_trial, cg_iterations, cg_residual = pcg_damage(
                history_new, damage
            )
            damage_new = jnp.clip(
                jnp.maximum(damage_trial, damage), 0.0, 1.0
            )
        else:
            damage_new = damage
            cg_iterations = jnp.asarray(0, dtype=jnp.int32)
            cg_residual = jnp.asarray(0.0, dtype=dtype)

        internal_force, _, _, _ = mechanics(displacement_new, damage_new)
        acceleration_new = -internal_force / mass[:, None]
        acceleration_new = enforce_acceleration(acceleration_new, next_time)
        velocity_new = velocity + 0.5 * dt_j * (
            acceleration + acceleration_new
        )
        velocity_new = enforce_velocity(velocity_new, next_time)
        # The external nodal reaction required to impose the impact motion is
        # M*a + f_int. Integrate its work with the trapezoidal rule over the
        # common prescribed displacement increment on the impact segment.
        impact_reaction_new = jnp.sum(
            internal_force[impact, 0]
            + mass[impact] * acceleration_new[impact, 0]
        )
        boundary_work_new = boundary_work + 0.5 * (
            impact_reaction + impact_reaction_new
        ) * (impact_u_new - impact_u_old)
        return (
            (
                displacement_new,
                velocity_new,
                acceleration_new,
                damage_new,
                history_new,
                boundary_work_new,
                impact_reaction_new,
            ),
            cg_iterations,
            cg_residual,
        )

    @jax.jit
    def advance_with_damage(state, time_value):
        return advance_impl(state, time_value, True)

    @jax.jit
    def advance_without_damage(state, time_value):
        return advance_impl(state, time_value, False)

    @jax.jit
    def diagnostics(state):
        (
            displacement,
            velocity,
            _,
            damage,
            _,
            boundary_work,
            impact_reaction,
        ) = state
        _, psi_positive, psi_negative, g = mechanics(
            displacement, damage
        )
        elastic = jnp.sum(areas * (g * psi_positive + psi_negative))
        kinetic = 0.5 * jnp.sum(
            mass * jnp.sum(velocity * velocity, axis=1)
        )
        local_damage = damage[triangles]
        damage_squared_integral = (
            areas
            / 6.0
            * (
                jnp.sum(local_damage * local_damage, axis=1)
                + local_damage[:, 0] * local_damage[:, 1]
                + local_damage[:, 1] * local_damage[:, 2]
                + local_damage[:, 2] * local_damage[:, 0]
            )
        )
        grad_damage = jnp.einsum(
            "ea,eaj->ej", local_damage, gradients
        )
        fracture = jnp.sum(
            gc_j / (2.0 * l0_j) * damage_squared_integral
            + 0.5
            * gc_j
            * l0_j
            * areas
            * jnp.sum(grad_damage * grad_damage, axis=1)
        )
        damage_max = jnp.max(damage)

        dx = nodes[:, 0] - NOTCH_LENGTH
        dy = nodes[:, 1] - NOTCH_Y
        distance_squared = dx * dx + dy * dy
        active = (damage >= 0.5) & (dx > 0.2)
        scored_distance = jnp.where(active, distance_squared, -1.0)
        tip_index = jnp.argmax(scored_distance)
        maximum_distance_squared = jnp.max(scored_distance)
        crack_extension = jnp.sqrt(jnp.maximum(maximum_distance_squared, 0.0))
        crack_angle = jnp.where(
            maximum_distance_squared > 0.0,
            jnp.degrees(jnp.arctan2(dy[tip_index], dx[tip_index])),
            jnp.asarray(0.0, dtype=dtype),
        )
        return jnp.asarray(
            [
                elastic,
                kinetic,
                fracture,
                boundary_work,
                elastic + kinetic + fracture - boundary_work,
                impact_reaction,
                damage_max,
                crack_extension,
                crack_angle,
            ],
            dtype=dtype,
        )

    return (
        advance_with_damage,
        advance_without_damage,
        diagnostics,
        enforce_acceleration,
    )


# =============================================================================
# Output and restart helpers
# =============================================================================

HISTORY_FIELDS = (
    "step",
    "physical_time_s",
    "physical_time_us",
    "elastic_energy_Nmm",
    "kinetic_energy_Nmm",
    "fracture_energy_Nmm",
    "total_energy_Nmm",
    "boundary_work_Nmm",
    "energy_balance_residual_Nmm",
    "energy_balance_relative",
    "impact_reaction_N",
    "damage_max",
    "crack_extension_mm",
    "crack_angle_deg",
    "cg_iterations_mean_block",
    "cg_iterations_max_block",
    "cg_residual_max_block",
    "mean_step_wall_s_block",
)

TIMING_FIELDS = (
    "start_step",
    "end_step",
    "steps",
    "wall_s",
    "mean_step_s",
    "damage_solves",
    "cg_iterations_mean",
    "cg_iterations_max",
    "cg_residual_max",
)


def append_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_checkpoint(
    path: Path,
    *,
    step: int,
    physical_time: float,
    physical_solver_time: float,
    state_host: tuple[np.ndarray, ...],
) -> None:
    temporary = path.with_name(path.stem + "_temporary.npz")
    (
        displacement,
        velocity,
        acceleration,
        damage,
        history,
        boundary_work,
        impact_reaction,
    ) = state_host
    np.savez(
        temporary,
        format=np.asarray("kalthoff_dynamic_checkpoint_v2"),
        step=np.asarray(step),
        physical_time=np.asarray(physical_time),
        physical_solver_time=np.asarray(physical_solver_time),
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        damage=damage,
        history=history,
        boundary_work=boundary_work,
        impact_reaction=impact_reaction,
    )
    temporary.replace(path)


def load_checkpoint(path: Path):
    data = np.load(path, allow_pickle=False)
    return (
        int(data["step"]),
        float(data["physical_time"]),
        float(data["physical_solver_time"]),
        (
            np.asarray(data["displacement"]),
            np.asarray(data["velocity"]),
            np.asarray(data["acceleration"]),
            np.asarray(data["damage"]),
            np.asarray(data["history"]),
            np.asarray(data["boundary_work"]),
            np.asarray(data["impact_reaction"]),
        ),
    )


def state_to_host(state) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(jax.device_get(value)) for value in state)


def save_snapshot(
    path: Path,
    *,
    nodes: np.ndarray,
    triangles: np.ndarray,
    state_host: tuple[np.ndarray, ...],
    step: int,
    physical_time: float,
    diagnostics_values: np.ndarray,
) -> None:
    (
        displacement,
        velocity,
        acceleration,
        damage,
        history,
        boundary_work,
        impact_reaction,
    ) = state_host
    np.savez_compressed(
        path,
        format=np.asarray("kalthoff_dynamic_snapshot_v1"),
        nodes=nodes,
        triangles=triangles,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        damage=damage,
        history=history,
        boundary_work=boundary_work,
        impact_reaction=impact_reaction,
        step=np.asarray(step),
        physical_time_s=np.asarray(physical_time),
        diagnostics=np.asarray(diagnostics_values),
    )


def save_vtu(
    path: Path,
    nodes: np.ndarray,
    triangles: np.ndarray,
    state_host: tuple[np.ndarray, ...],
) -> None:
    import meshio

    displacement, velocity, acceleration, damage, history = state_host[:5]
    points = np.column_stack([nodes, np.zeros(nodes.shape[0])])

    def vector3(values):
        return np.column_stack([values, np.zeros(values.shape[0])])

    meshio.write_points_cells(
        str(path),
        points,
        [("triangle", triangles)],
        point_data={
            "damage": damage,
            "displacement": vector3(displacement),
            "velocity": vector3(velocity),
            "acceleration": vector3(acceleration),
        },
        cell_data={"history": [history]},
    )


def save_summary_plot(
    path: Path,
    nodes: np.ndarray,
    triangles: np.ndarray,
    damage: np.ndarray,
    history_rows: list[dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError:
        print("  summary plot skipped: matplotlib is not installed")
        return

    numeric = {
        name: np.asarray([float(row[name]) for row in history_rows])
        for name in HISTORY_FIELDS
        if name in history_rows[0]
    }
    triangulation = mtri.Triangulation(
        nodes[:, 0], nodes[:, 1], triangles=triangles
    )
    fig = plt.figure(figsize=(13.0, 8.0), dpi=180)
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)
    ax_damage = fig.add_subplot(grid[0, :])
    ax_energy = fig.add_subplot(grid[1, 0])
    ax_crack = fig.add_subplot(grid[1, 1])

    image = ax_damage.tripcolor(
        triangulation,
        np.clip(damage, 0.0, 1.0),
        shading="gouraud",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
    )
    ax_damage.set_aspect("equal", adjustable="box")
    ax_damage.set_xlim(0.0, W)
    ax_damage.set_ylim(0.0, HEIGHT)
    ax_damage.set_xlabel("$x$ (mm)")
    ax_damage.set_ylabel("$y$ (mm)")
    ax_damage.set_title("(a) Final damage field", loc="left")
    fig.colorbar(image, ax=ax_damage, label="Damage $d$", shrink=0.82)

    time_us = numeric["physical_time_us"]
    ax_energy.plot(
        time_us, numeric["elastic_energy_Nmm"], label="Elastic", lw=1.8
    )
    ax_energy.plot(
        time_us, numeric["kinetic_energy_Nmm"], label="Kinetic", lw=1.8
    )
    ax_energy.plot(
        time_us, numeric["fracture_energy_Nmm"], label="Fracture", lw=1.8
    )
    ax_energy.plot(
        time_us,
        numeric["boundary_work_Nmm"],
        "--",
        color="black",
        label="Boundary work",
        lw=1.5,
    )
    ax_energy.set_xlabel("Time ($\\mu$s)")
    ax_energy.set_ylabel("Energy per unit thickness (N mm)")
    ax_energy.set_title("(b) Energy histories", loc="left")
    ax_energy.legend(frameon=False)

    ax_crack.plot(
        time_us,
        numeric["crack_extension_mm"],
        color="#2b6f9f",
        lw=1.8,
        label="Crack extension",
    )
    ax_angle = ax_crack.twinx()
    ax_angle.plot(
        time_us,
        numeric["crack_angle_deg"],
        color="#d94b5b",
        lw=1.5,
        label="Crack angle",
    )
    ax_crack.set_xlabel("Time ($\\mu$s)")
    ax_crack.set_ylabel("Crack extension (mm)", color="#2b6f9f")
    ax_angle.set_ylabel("Crack angle (deg)", color="#d94b5b")
    ax_crack.set_title("(c) Crack-tip metrics", loc="left")
    ax_crack.tick_params(axis="y", colors="#2b6f9f")
    ax_angle.tick_params(axis="y", colors="#d94b5b")

    for axis in (ax_damage, ax_energy, ax_crack, ax_angle):
        axis.tick_params(direction="in", top=True, right=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def query_memory_stats(device) -> dict[str, int | float | None]:
    try:
        stats = device.memory_stats() or {}
    except Exception:
        return {}
    result = {}
    for key in (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "largest_free_block_bytes",
    ):
        value = stats.get(key)
        result[key] = int(value) if value is not None else None
    return result


# =============================================================================
# Independent-process mesh-density sweep
# =============================================================================

SWEEP_FIELDS = (
    "h_crack_mm",
    "nodes",
    "triangles",
    "l0_over_h",
    "h_incircle_min_mm",
    "minimum_altitude_mm",
    "aspect_ratio_max",
    "dt_s",
    "steps",
    "first_block_including_jit_s",
    "steady_median_step_s",
    "physical_solver_time_s",
    "initiation_time_us",
    "crack_extension_mm",
    "crack_angle_deg",
    "damage_max",
    "energy_balance_relative",
    "peak_device_bytes",
    "status",
    "case_directory",
)


def parse_sweep_h_values(text: str) -> tuple[float, ...]:
    values = tuple(
        sorted(
            {float(item.strip()) for item in text.split(",") if item.strip()},
            reverse=True,
        )
    )
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--sweep-h-values must contain positive values")
    return values


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sweep_case_complete(case_dir: Path) -> bool:
    summary_path = case_dir / "run_summary.json"
    config_path = case_dir / "run_config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return False
    summary = read_json(summary_path)
    config = read_json(config_path)
    return int(summary.get("completed_steps", -1)) >= int(
        config.get("solver", {}).get("steps", 0)
    )


def collect_sweep_case(case_dir: Path, h_crack: float) -> dict | None:
    mesh_path = case_dir / "mesh_metadata.json"
    if not mesh_path.is_file():
        return None
    mesh_document = read_json(mesh_path)
    mesh = mesh_document["mesh"]
    row = {
        "h_crack_mm": h_crack,
        "nodes": int(mesh["nodes"]),
        "triangles": int(mesh["triangles"]),
        "l0_over_h": float(L0 / h_crack),
        "h_incircle_min_mm": float(mesh["h_incircle_min_mm"]),
        "minimum_altitude_mm": float(mesh["minimum_altitude_mm"]),
        "aspect_ratio_max": float(mesh["aspect_ratio_max"]),
        "dt_s": None,
        "steps": None,
        "first_block_including_jit_s": None,
        "steady_median_step_s": None,
        "physical_solver_time_s": None,
        "initiation_time_us": None,
        "crack_extension_mm": None,
        "crack_angle_deg": None,
        "damage_max": None,
        "energy_balance_relative": None,
        "peak_device_bytes": None,
        "status": "mesh-only",
        "case_directory": str(case_dir),
    }
    config_path = case_dir / "run_config.json"
    summary_path = case_dir / "run_summary.json"
    if config_path.is_file():
        config = read_json(config_path)
        row["dt_s"] = float(config["solver"]["dt_s"])
        row["steps"] = int(config["solver"]["steps"])
    if summary_path.is_file():
        summary = read_json(summary_path)
        final = summary["final"]
        memory = summary.get("device_memory_stats", {})
        row.update(
            {
                "first_block_including_jit_s": summary.get(
                    "first_timed_block_including_jit_s"
                ),
                "steady_median_step_s": summary.get(
                    "steady_block_median_step_s"
                ),
                "physical_solver_time_s": summary.get(
                    "physical_solver_time_s"
                ),
                "initiation_time_us": summary.get(
                    "initiation_time_us_sampled"
                ),
                "crack_extension_mm": final.get("crack_extension_mm"),
                "crack_angle_deg": final.get("crack_angle_deg"),
                "damage_max": final.get("damage_max"),
                "energy_balance_relative": final.get(
                    "energy_balance_relative"
                ),
                "peak_device_bytes": memory.get("peak_bytes_in_use"),
                "status": (
                    "complete"
                    if sweep_case_complete(case_dir)
                    else "partial"
                ),
            }
        )
    return row


def write_sweep_aggregate(sweep_root: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: int(row["nodes"]))
    with (sweep_root / "mesh_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (sweep_root / "mesh_sweep_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "levels": rows,
            },
            stream,
            indent=2,
        )

    complete = [
        row
        for row in rows
        if row["status"] == "complete"
        and row["steady_median_step_s"] is not None
    ]
    if not complete:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    nodes = np.asarray([row["nodes"] for row in complete], dtype=np.float64)

    def numeric_values(key: str) -> np.ndarray:
        return np.asarray(
            [
                np.nan if row[key] is None else float(row[key])
                for row in complete
            ],
            dtype=np.float64,
        )

    step_time = numeric_values("steady_median_step_s")
    total_time = numeric_values("physical_solver_time_s")
    initiation = numeric_values("initiation_time_us")
    angle = numeric_values("crack_angle_deg")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), dpi=180)
    ax_step, ax_total, ax_init, ax_angle = axes.reshape(-1)
    ax_step.loglog(nodes, step_time, "o-", color="#2b6f9f", lw=1.8)
    ax_step.set_xlabel("Nodes")
    ax_step.set_ylabel("Median step time (s)")
    ax_step.set_title("(a) Per-step computational cost", loc="left")
    ax_total.loglog(nodes, total_time, "o-", color="#335c9b", lw=1.8)
    ax_total.set_xlabel("Nodes")
    ax_total.set_ylabel("Physical solver time (s)")
    ax_total.set_title("(b) End-to-end wall time", loc="left")
    ax_init.semilogx(nodes, initiation, "o-", color="#d28b26", lw=1.8)
    ax_init.set_xlabel("Nodes")
    ax_init.set_ylabel("Crack initiation time ($\\mu$s)")
    ax_init.set_title("(c) Initiation-time convergence", loc="left")
    ax_angle.semilogx(nodes, angle, "o-", color="#c84a5a", lw=1.8)
    ax_angle.axhline(70.0, color="black", ls="--", lw=1.0, label="70° reference")
    ax_angle.set_xlabel("Nodes")
    ax_angle.set_ylabel("Final crack angle (deg)")
    ax_angle.set_title("(d) Crack-angle convergence", loc="left")
    ax_angle.legend(frameon=False)
    for axis in axes.reshape(-1):
        axis.grid(True, which="both", alpha=0.2)
        axis.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(
        sweep_root / "mesh_sweep_summary.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def run_mesh_sweep(args: argparse.Namespace) -> None:
    if args.mesh is not None:
        raise ValueError("--mesh cannot be combined with --mesh-sweep")
    h_values = parse_sweep_h_values(args.sweep_h_values)
    sweep_root = args.sweep_root or (
        script_root() / "outputs" / "kalthoff_winkler_mesh_sweep"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    print("=" * 84, flush=True)
    print(
        "Kalthoff-Winkler independent-process mesh-density sweep", flush=True
    )
    print(
        f"  levels: {', '.join(f'h={value:g} mm' for value in h_values)}",
        flush=True,
    )
    print(f"  root:   {sweep_root}", flush=True)
    print(
        "  mode:   "
        + (
            "mesh generation/precheck only"
            if args.sweep_precheck_only
            else f"full {args.total_time_us:g}-us dynamic simulations"
        ),
        flush=True,
    )
    print("=" * 84, flush=True)

    rows: list[dict] = []
    for index, h_crack in enumerate(h_values, 1):
        case_dir = sweep_root / f"h{mesh_token(h_crack)}"
        complete = sweep_case_complete(case_dir)
        if complete and not args.sweep_rerun and not args.sweep_precheck_only:
            print(
                f"[{index}/{len(h_values)}] h={h_crack:g}: "
                "complete result found; skipping",
                flush=True,
            )
        else:
            command = [
                sys.executable,
                str(script_path),
                "--platform",
                args.platform,
                "--dtype",
                args.dtype,
                "--h-crack",
                f"{h_crack:.12g}",
                "--h-coarse",
                f"{args.h_coarse:.12g}",
                "--outdir",
                str(case_dir),
                "--total-time-us",
                f"{args.total_time_us:.12g}",
                "--impact-velocity-m-s",
                f"{args.impact_velocity_m_s:.12g}",
                "--ramp-time-us",
                f"{args.ramp_time_us:.12g}",
                "--dt-safety",
                f"{args.dt_safety:.12g}",
                "--damage-every",
                str(args.damage_every),
                "--cg-tol",
                f"{args.cg_tol:.12g}",
                "--cg-atol",
                f"{args.cg_atol:.12g}",
                "--cg-maxiter",
                str(args.cg_maxiter),
                "--timing-block",
                str(args.timing_block),
                "--history-every",
                str(args.history_every),
                "--checkpoint-every",
                str(args.checkpoint_every),
                "--snapshot-times-us",
                args.snapshot_times_us,
                "--print-every",
                str(args.print_every),
            ]
            if args.sweep_precheck_only:
                command.append("--precheck-only")
            if args.resume and not args.sweep_precheck_only:
                command.append("--resume")
            if args.vtk and not args.sweep_precheck_only:
                command.append("--vtk")
            if not args.plot:
                command.append("--no-plot")
            if args.sweep_rerun:
                command.append("--regenerate-mesh")
            print(
                f"[{index}/{len(h_values)}] h={h_crack:g}: "
                f"launching isolated case in {case_dir}",
                flush=True,
            )
            subprocess.run(command, cwd=script_root(), check=True)

        row = collect_sweep_case(case_dir, h_crack)
        if row is not None:
            rows.append(row)
            write_sweep_aggregate(sweep_root, rows)

    print("-" * 84)
    print(f"mesh sweep finished: {sweep_root}")
    print(f"summary: {sweep_root / 'mesh_sweep_summary.csv'}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()
    if args.h_crack <= 0.0 or args.h_coarse <= args.h_crack:
        raise ValueError("require 0 < h-crack < h-coarse")
    if args.total_time_us <= 0.0 or args.ramp_time_us <= 0.0:
        raise ValueError("physical times must be positive")
    if args.impact_velocity_m_s <= 0.0:
        raise ValueError("impact velocity must be positive")
    if not (0.0 < args.dt_safety <= 1.0):
        raise ValueError("dt-safety must lie in (0, 1]")
    if (
        args.damage_every < 1
        or args.cg_maxiter < 1
        or args.cg_tol <= 0.0
        or args.cg_atol <= 0.0
    ):
        raise ValueError("invalid damage/CG settings")
    if args.timing_block < 1 or args.history_every < 1:
        raise ValueError("timing-block and history-every must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    snapshot_times_us = parse_snapshot_times(args.snapshot_times_us)
    if args.mesh_sweep:
        if args.steps is not None:
            raise ValueError(
                "--steps cannot be combined with --mesh-sweep; "
                "all levels must reach the same physical time"
            )
        run_mesh_sweep(args)
        return

    use_x64 = args.dtype == "float64"
    jax.config.update("jax_enable_x64", use_x64)
    dtype = jnp.float64 if use_x64 else jnp.float32
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    if args.platform == "gpu" and all(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU requested, but JAX devices are {devices}")
    device = devices[0]
    selected_device_slug = device_slug(device)

    if args.outdir is None:
        args.outdir = (
            script_root()
            / "outputs"
            / f"results_{selected_device_slug}"
            / "kalthoff_winkler_dynamic"
            / "jax"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    mesh_file = args.mesh or (
        args.outdir / f"kalthoff_h{mesh_token(args.h_crack)}.msh"
    )
    if args.regenerate_mesh or not mesh_file.is_file():
        print(f"generating self-constructed Kalthoff mesh: {mesh_file}")
        generate_mesh(mesh_file, args.h_crack, args.h_coarse)

    nodes, triangles, areas, boundaries = load_mesh(mesh_file)
    (
        gradients,
        nodal_mass,
        h_min,
        incircle_diameter,
        minimum_altitude,
        aspect_ratio,
    ) = preprocess_mesh(nodes, triangles, areas)
    metadata = mesh_metadata(
        nodes,
        triangles,
        areas,
        incircle_diameter,
        minimum_altitude,
        aspect_ratio,
        boundaries,
        mesh_file,
        args.h_crack,
        args.h_coarse,
    )
    expected_mass = RHO * float(np.sum(areas))  # implicit 1 mm thickness
    relative_mass_error = abs(float(np.sum(nodal_mass)) - expected_mass) / max(
        expected_mass, 1.0e-30
    )
    if relative_mass_error > 1.0e-10:
        raise RuntimeError(
            f"lumped-mass check failed: relative error={relative_mass_error:.3e}"
        )
    if float(np.max(aspect_ratio)) > 10.0:
        raise RuntimeError(
            f"mesh contains a poor triangle: max aspect={np.max(aspect_ratio):.2f}"
        )
    if args.h_crack > L0 / 3.0:
        print(
            "WARNING: nominal crack-corridor resolution is coarser than l0/3 "
            f"(l0/h={L0 / args.h_crack:.2f}); use only for smoke testing."
        )
    with (args.outdir / "mesh_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    if args.precheck_only:
        print(json.dumps(metadata, indent=2))
        print("precheck complete; dynamic solve was not run")
        return

    cp = math.sqrt((LAMBDA + 2.0 * MU) / RHO)
    cs = math.sqrt(MU / RHO)
    rayleigh_speed = cs * (0.862 + 1.14 * NU) / (1.0 + NU)
    dt_cfl = args.dt_safety * h_min / cp
    requested_total_time = args.total_time_us * 1.0e-6
    if args.steps is None:
        number_of_steps = int(math.ceil(requested_total_time / dt_cfl))
        dt = requested_total_time / number_of_steps
    else:
        number_of_steps = args.steps
        dt = dt_cfl
    simulated_total_time = number_of_steps * dt
    impact_velocity_mm_s = args.impact_velocity_m_s * 1.0e3
    ramp_time_s = args.ramp_time_us * 1.0e-6
    snapshot_steps = {
        max(1, min(number_of_steps, int(round(time_us * 1.0e-6 / dt)))): time_us
        for time_us in snapshot_times_us
        if time_us * 1.0e-6 <= simulated_total_time + 0.5 * dt
    }

    (
        advance_with_damage,
        advance_without_damage,
        diagnostics,
        enforce_initial_acceleration,
    ) = make_solver(
        nodes_np=nodes,
        triangles_np=triangles,
        areas_np=areas,
        gradients_np=gradients,
        mass_np=nodal_mass,
        bottom_np=boundaries["bottom_symmetry"],
        impact_np=boundaries["left_impact"],
        dtype=dtype,
        dt=dt,
        impact_velocity_mm_s=impact_velocity_mm_s,
        ramp_time_s=ramp_time_s,
        cg_tol=args.cg_tol,
        cg_atol=args.cg_atol,
        cg_maxiter=args.cg_maxiter,
    )

    checkpoint_file = args.outdir / "checkpoint_latest.npz"
    history_file = args.outdir / "history.csv"
    timing_file = args.outdir / "timing_blocks.csv"
    snapshot_dir = args.outdir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and checkpoint_file.is_file():
        (
            completed_step,
            physical_time,
            physical_solver_time,
            state_np,
        ) = load_checkpoint(checkpoint_file)
        if state_np[0].shape != (nodes.shape[0], 2):
            raise RuntimeError("checkpoint mesh does not match the current mesh")
        state = tuple(jnp.asarray(value, dtype=dtype) for value in state_np)
        history_rows: list[dict] = load_csv_rows(history_file)
        print(f"resuming after step {completed_step:,}: {checkpoint_file}")
    else:
        completed_step = 0
        physical_time = 0.0
        physical_solver_time = 0.0
        displacement = jnp.zeros((nodes.shape[0], 2), dtype=dtype)
        velocity = jnp.zeros_like(displacement)
        acceleration = jnp.zeros_like(displacement)
        acceleration = enforce_initial_acceleration(
            acceleration, jnp.asarray(0.0, dtype=dtype)
        )
        damage = jnp.zeros((nodes.shape[0],), dtype=dtype)
        history = jnp.zeros((triangles.shape[0],), dtype=dtype)
        boundary_work = jnp.asarray(0.0, dtype=dtype)
        initial_impact_reaction = jnp.asarray(
            np.sum(nodal_mass[boundaries["left_impact"]])
            * impact_velocity_mm_s
            / ramp_time_s,
            dtype=dtype,
        )
        state = (
            displacement,
            velocity,
            acceleration,
            damage,
            history,
            boundary_work,
            initial_impact_reaction,
        )
        history_rows = []
        for path in (history_file, timing_file):
            if path.exists():
                path.unlink()

    run_config = {
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "devices": [str(item) for item in devices],
        "device_kinds": [item.device_kind for item in devices],
        "dtype": args.dtype,
        "material": {
            "E_MPa": E,
            "nu": NU,
            "Gc_N_per_mm": GC,
            "l0_mm": L0,
            "rho_tonne_per_mm3": RHO,
            "eta": ETA,
            "model": "AT2",
            "energy_split": "Miehe spectral",
            "plane": "strain",
            "implicit_thickness_mm": 1.0,
            "total_mass_tonne": float(np.sum(nodal_mass)),
        },
        "loading": {
            "impact_velocity_m_s": args.impact_velocity_m_s,
            "ramp_time_us": args.ramp_time_us,
            "requested_total_time_us": args.total_time_us,
            "simulated_total_time_us": simulated_total_time * 1.0e6,
            "final_prescribed_displacement_mm": (
                impact_velocity_mm_s
                * (simulated_total_time - 0.5 * ramp_time_s)
                if simulated_total_time >= ramp_time_s
                else 0.5
                * impact_velocity_mm_s
                * simulated_total_time**2
                / ramp_time_s
            ),
        },
        "solver": {
            "time_integrator": "velocity-Verlet",
            "mass": "lumped",
            "dt_s": dt,
            "dt_cfl_s": dt_cfl,
            "dt_safety": args.dt_safety,
            "steps": number_of_steps,
            "damage_every": args.damage_every,
            "cg_tol": args.cg_tol,
            "cg_atol": args.cg_atol,
            "cg_maxiter": args.cg_maxiter,
            "cp_mm_s": cp,
            "cs_mm_s": cs,
            "rayleigh_mm_s": rayleigh_speed,
        },
        "mesh": metadata,
    }
    with (args.outdir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)

    # Compile diagnostics separately so its JIT time is not mixed into the
    # physical step timing.  The first physical step intentionally includes
    # compilation of the full advance kernel and is reported separately.
    initial_diagnostics = diagnostics(state)
    jax.block_until_ready(initial_diagnostics)

    print("=" * 84)
    print("Standalone JAX Kalthoff-Winkler dynamic phase-field fracture")
    print(f"  JAX {jax.__version__}; Python {platform.python_version()}")
    print(f"  devices: {devices}")
    print(
        f"  mesh: {nodes.shape[0]:,} nodes, {triangles.shape[0]:,} triangles, "
        f"h_min(incircle diameter)={h_min:.5g} mm"
    )
    print(
        f"  l0/h_fine={L0 / args.h_crack:.2f}, "
        f"minimum altitude={np.min(minimum_altitude):.5g} mm, "
        f"max aspect={np.max(aspect_ratio):.2f}"
    )
    print(
        f"  dt={dt:.4e} s, steps={number_of_steps:,}, "
        f"T={simulated_total_time * 1.0e6:.3f} us, "
        f"damage_every={args.damage_every}"
    )
    print(
        f"  cp={cp / 1.0e6:.1f} km/s, cs={cs / 1.0e6:.1f} km/s, "
        f"cR={rayleigh_speed / 1.0e6:.1f} km/s"
    )
    print("=" * 84)

    first_step_time = None
    initiation_time_us = None
    pending_history_rows: list[dict] = []
    pending_timing_rows: list[dict] = []

    while completed_step < number_of_steps:
        chunk_start = completed_step + 1
        # Isolate the first physical increment so its reported wall time is
        # exactly JIT compilation plus one solve, matching the paper protocol.
        if completed_step == 0:
            chunk_end = 1
        else:
            chunk_end = min(
                number_of_steps,
                completed_step + args.timing_block,
            )
        future_boundaries = [
            step
            for step in snapshot_steps
            if completed_step < step < chunk_end
        ]
        if args.checkpoint_every > 0:
            next_checkpoint = (
                (completed_step // args.checkpoint_every) + 1
            ) * args.checkpoint_every
            if completed_step < next_checkpoint < chunk_end:
                future_boundaries.append(next_checkpoint)
        next_history = (
            (completed_step // args.history_every) + 1
        ) * args.history_every
        if completed_step < next_history < chunk_end:
            future_boundaries.append(next_history)
        if future_boundaries:
            chunk_end = min(future_boundaries)

        cg_iterations_device = []
        cg_residuals_device = []
        wall_start = time.perf_counter()
        for step in range(chunk_start, chunk_end + 1):
            step_time_value = jnp.asarray((step - 1) * dt, dtype=dtype)
            if step % args.damage_every == 0:
                state, cg_iterations, cg_residual = advance_with_damage(
                    state, step_time_value
                )
            else:
                state, cg_iterations, cg_residual = advance_without_damage(
                    state, step_time_value
                )
            cg_iterations_device.append(cg_iterations)
            cg_residuals_device.append(cg_residual)

        block_stats = jnp.stack(
            [
                jnp.asarray(cg_iterations_device, dtype=dtype),
                jnp.asarray(cg_residuals_device, dtype=dtype),
            ],
            axis=1,
        )
        jax.block_until_ready(state[3])
        block_stats_host = np.asarray(jax.device_get(block_stats))
        block_wall = time.perf_counter() - wall_start
        block_steps = chunk_end - chunk_start + 1
        physical_solver_time += block_wall
        if first_step_time is None and chunk_start == 1:
            first_step_time = block_wall

        completed_step = chunk_end
        physical_time = completed_step * dt
        solved_mask = block_stats_host[:, 0] > 0
        solved_iterations = block_stats_host[solved_mask, 0]
        solved_residuals = block_stats_host[solved_mask, 1]
        if (
            np.any(~np.isfinite(solved_residuals))
            or np.any(
                (solved_iterations >= args.cg_maxiter)
                & (solved_residuals > 1.05 * args.cg_tol)
            )
        ):
            raise RuntimeError(
                f"damage CG failed in steps {chunk_start}-{chunk_end}: "
                f"max iterations={int(np.max(solved_iterations))}, "
                f"max effective residual={float(np.max(solved_residuals)):.3e}"
            )
        timing_row = {
            "start_step": chunk_start,
            "end_step": chunk_end,
            "steps": block_steps,
            "wall_s": block_wall,
            "mean_step_s": block_wall / block_steps,
            "damage_solves": int(np.count_nonzero(solved_mask)),
            "cg_iterations_mean": (
                float(np.mean(solved_iterations))
                if solved_iterations.size
                else 0.0
            ),
            "cg_iterations_max": (
                int(np.max(solved_iterations))
                if solved_iterations.size
                else 0
            ),
            "cg_residual_max": (
                float(np.max(solved_residuals))
                if solved_residuals.size
                else 0.0
            ),
        }
        pending_timing_rows.append(timing_row)

        need_history = (
            completed_step % args.history_every == 0
            or completed_step in snapshot_steps
            or completed_step == number_of_steps
            or completed_step <= args.timing_block
        )
        diagnostic_values = None
        if need_history:
            diagnostic_values = np.asarray(
                jax.device_get(diagnostics(state)), dtype=np.float64
            )
            (
                elastic_energy,
                kinetic_energy,
                fracture_energy,
                boundary_work,
                energy_balance_residual,
                impact_reaction,
                damage_max,
                crack_extension,
                crack_angle,
            ) = diagnostic_values
            if initiation_time_us is None and crack_extension > 0.2:
                initiation_time_us = physical_time * 1.0e6
            history_row = {
                "step": completed_step,
                "physical_time_s": physical_time,
                "physical_time_us": physical_time * 1.0e6,
                "elastic_energy_Nmm": elastic_energy,
                "kinetic_energy_Nmm": kinetic_energy,
                "fracture_energy_Nmm": fracture_energy,
                "total_energy_Nmm": (
                    elastic_energy + kinetic_energy + fracture_energy
                ),
                "boundary_work_Nmm": boundary_work,
                "energy_balance_residual_Nmm": energy_balance_residual,
                "energy_balance_relative": (
                    energy_balance_residual
                    / max(abs(boundary_work), 1.0e-30)
                ),
                "impact_reaction_N": impact_reaction,
                "damage_max": damage_max,
                "crack_extension_mm": crack_extension,
                "crack_angle_deg": crack_angle,
                "cg_iterations_mean_block": timing_row[
                    "cg_iterations_mean"
                ],
                "cg_iterations_max_block": timing_row[
                    "cg_iterations_max"
                ],
                "cg_residual_max_block": timing_row["cg_residual_max"],
                "mean_step_wall_s_block": timing_row["mean_step_s"],
            }
            pending_history_rows.append(history_row)
            history_rows.append(history_row)

        if completed_step in snapshot_steps:
            if diagnostic_values is None:
                diagnostic_values = np.asarray(
                    jax.device_get(diagnostics(state)), dtype=np.float64
                )
            state_host = state_to_host(state)
            label_us = snapshot_steps[completed_step]
            save_snapshot(
                snapshot_dir / f"state_{label_us:g}us_step{completed_step:07d}.npz",
                nodes=nodes,
                triangles=triangles,
                state_host=state_host,
                step=completed_step,
                physical_time=physical_time,
                diagnostics_values=diagnostic_values,
            )

        checkpoint_due = (
            args.checkpoint_every > 0
            and completed_step % args.checkpoint_every == 0
        )
        if checkpoint_due or completed_step == number_of_steps:
            state_host = state_to_host(state)
            save_checkpoint(
                checkpoint_file,
                step=completed_step,
                physical_time=physical_time,
                physical_solver_time=physical_solver_time,
                state_host=state_host,
            )
            append_rows(history_file, HISTORY_FIELDS, pending_history_rows)
            append_rows(timing_file, TIMING_FIELDS, pending_timing_rows)
            pending_history_rows.clear()
            pending_timing_rows.clear()

        if (
            completed_step <= args.timing_block
            or completed_step % args.print_every == 0
            or completed_step == number_of_steps
            or completed_step in snapshot_steps
        ):
            if diagnostic_values is None:
                diagnostic_values = np.asarray(
                    jax.device_get(diagnostics(state)), dtype=np.float64
                )
            print(
                f"  step={completed_step:7d}/{number_of_steps} "
                f"t={physical_time * 1.0e6:8.3f} us "
                f"dmax={diagnostic_values[6]:.5f} "
                f"da={diagnostic_values[7]:7.3f} mm "
                f"angle={diagnostic_values[8]:6.2f} deg "
                f"CG={timing_row['cg_iterations_mean']:.1f}/"
                f"{timing_row['cg_iterations_max']} "
                f"dt_wall={timing_row['mean_step_s']:.4f} s"
            )

    append_rows(history_file, HISTORY_FIELDS, pending_history_rows)
    append_rows(timing_file, TIMING_FIELDS, pending_timing_rows)
    final_state_host = state_to_host(state)
    final_diagnostics = np.asarray(
        jax.device_get(diagnostics(state)), dtype=np.float64
    )
    save_snapshot(
        snapshot_dir / "state_final.npz",
        nodes=nodes,
        triangles=triangles,
        state_host=final_state_host,
        step=completed_step,
        physical_time=physical_time,
        diagnostics_values=final_diagnostics,
    )
    if args.vtk:
        save_vtu(
            args.outdir / "state_final.vtu",
            nodes,
            triangles,
            final_state_host,
        )
    if args.plot and history_rows:
        save_summary_plot(
            args.outdir / "kalthoff_dynamic_summary.png",
            nodes,
            triangles,
            final_state_host[3],
            history_rows,
        )

    timing_rows = load_csv_rows(timing_file)
    mean_step_values = np.asarray(
        [float(row["mean_step_s"]) for row in timing_rows],
        dtype=np.float64,
    )
    steady_values = mean_step_values[1:] if mean_step_values.size > 1 else mean_step_values
    summary = {
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_steps": completed_step,
        "physical_time_us": physical_time * 1.0e6,
        "physical_solver_time_s": physical_solver_time,
        "first_timed_block_including_jit_s": first_step_time,
        "steady_block_mean_step_s": (
            float(np.mean(steady_values)) if steady_values.size else None
        ),
        "steady_block_median_step_s": (
            float(np.median(steady_values)) if steady_values.size else None
        ),
        "initiation_time_us_sampled": initiation_time_us,
        "final": {
            "elastic_energy_Nmm": float(final_diagnostics[0]),
            "kinetic_energy_Nmm": float(final_diagnostics[1]),
            "fracture_energy_Nmm": float(final_diagnostics[2]),
            "boundary_work_Nmm": float(final_diagnostics[3]),
            "energy_balance_residual_Nmm": float(final_diagnostics[4]),
            "energy_balance_relative": float(
                final_diagnostics[4]
                / max(abs(final_diagnostics[3]), 1.0e-30)
            ),
            "impact_reaction_N": float(final_diagnostics[5]),
            "damage_max": float(final_diagnostics[6]),
            "crack_extension_mm": float(final_diagnostics[7]),
            "crack_angle_deg": float(final_diagnostics[8]),
        },
        "device_memory_stats": query_memory_stats(device),
        "output_directory": str(args.outdir),
    }
    with (args.outdir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print("-" * 84)
    print(
        f"completed {completed_step:,} steps, "
        f"{physical_time * 1.0e6:.3f} us in "
        f"{physical_solver_time:.3f} s physical solver time"
    )
    print(
        f"final crack extension={final_diagnostics[7]:.3f} mm, "
        f"angle={final_diagnostics[8]:.2f} deg, "
        f"dmax={final_diagnostics[6]:.5f}"
    )
    if initiation_time_us is not None:
        print(f"sampled crack initiation time: {initiation_time_us:.3f} us")
    print(f"outputs: {args.outdir}")


if __name__ == "__main__":
    main()
