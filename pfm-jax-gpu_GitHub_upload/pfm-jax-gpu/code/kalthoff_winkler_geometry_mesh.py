#!/usr/bin/env python3
"""
Generate and inspect a self-constructed Kalthoff--Winkler half-plate mesh.

This script performs geometry/mesh validation only.  It does not import JAX
or run the dynamic fracture solver.  The benchmark dimensions follow the
standard half-plate configuration used for the Kalthoff--Winkler impact test:

* 100 x 100 mm half plate (bottom edge is the symmetry plane)
* 50 mm horizontal edge notch centred at y = 25 mm
* impact segment on the left edge below the notch
* broad graded refinement around the expected upper-right crack corridor

Examples
--------
Quick local preview (default, approximately O(10^4) nodes):
  python kalthoff_winkler_geometry_mesh.py

Reference-size preview:
  python kalthoff_winkler_geometry_mesh.py --preset reference

Large mesh generation only (no fracture calculation):
  python kalthoff_winkler_geometry_mesh.py --preset large
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


W = 100.0
H = 100.0
NOTCH_LENGTH = 50.0
NOTCH_Y = 25.0
NOTCH_HALF_WIDTH = 0.05
EXPECTED_CRACK_ANGLE_DEG = 70.0

PRESETS = {
    "preview": {"h_crack": 0.50, "h_coarse": 5.0},
    "reference": {"h_crack": 0.25, "h_coarse": 5.0},
    "large": {"h_crack": 0.10, "h_coarse": 5.0},
}


def script_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code" else script_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and preview a Kalthoff-Winkler half-plate mesh"
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="preview",
        help="mesh density preset (default: preview)",
    )
    parser.add_argument(
        "--h-crack",
        type=float,
        default=None,
        help="override nominal fine element size in the crack corridor (mm)",
    )
    parser.add_argument(
        "--h-coarse",
        type=float,
        default=None,
        help="override far-field element size (mm)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory (default: outputs/mesh_preview/kalthoff_winkler)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the preview window after writing the PNG",
    )
    return parser


def mesh_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def expected_crack_endpoint() -> tuple[float, float]:
    """Intersect the 70-degree guide ray from the notch tip with y = H."""
    angle = math.radians(EXPECTED_CRACK_ANGLE_DEG)
    length = (H - NOTCH_Y) / math.sin(angle)
    return (
        NOTCH_LENGTH + length * math.cos(angle),
        H,
    )


def generate_mesh(mesh_file: Path, h_crack: float, h_coarse: float):
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

        # Boundary points.  The narrow, fixed-width notch is part of the
        # geometry and therefore does not change when the mesh is refined.
        p1 = geo.addPoint(0.0, 0.0, 0.0, h_coarse)
        p2 = geo.addPoint(W, 0.0, 0.0, h_coarse)
        p3 = geo.addPoint(W, H, 0.0, h_coarse)
        p4 = geo.addPoint(0.0, H, 0.0, h_coarse)
        p5 = geo.addPoint(0.0, NOTCH_Y + NOTCH_HALF_WIDTH, 0.0, h_crack)
        p6 = geo.addPoint(
            NOTCH_LENGTH,
            NOTCH_Y + NOTCH_HALF_WIDTH,
            0.0,
            h_crack,
        )
        p7 = geo.addPoint(
            NOTCH_LENGTH,
            NOTCH_Y - NOTCH_HALF_WIDTH,
            0.0,
            h_crack,
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

        # Guide points define a broad refinement corridor without becoming
        # mesh edges or introducing an externally supplied mesh.  The corridor
        # is deliberately wider than the diffuse crack band, so it resolves
        # the expected kink without constraining the crack to one line.
        end_x, end_y = expected_crack_endpoint()
        guide_points = []
        for fraction in np.linspace(0.0, 1.0, 33):
            guide_points.append(
                geo.addPoint(
                    NOTCH_LENGTH + fraction * (end_x - NOTCH_LENGTH),
                    NOTCH_Y + fraction * (end_y - NOTCH_Y),
                    0.0,
                    h_crack,
                )
            )

        geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.model.setPhysicalName(2, 1, "plate")
        boundary_groups = {
            "bottom_symmetry": (11, [bottom]),
            "right": (12, [right]),
            "top": (13, [top]),
            "left_top": (14, [left_top]),
            "notch": (15, [notch_upper, notch_tip, notch_lower]),
            "left_impact": (16, [left_impact]),
        }
        for name, (tag, curves) in boundary_groups.items():
            gmsh.model.addPhysicalGroup(1, curves, tag)
            gmsh.model.setPhysicalName(1, tag, name)

        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "PointsList", guide_points)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)

        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_crack)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 4.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 15.0)

        tip_distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(tip_distance, "PointsList", [p6, p7])
        tip_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tip_threshold, "InField", tip_distance)
        gmsh.model.mesh.field.setNumber(tip_threshold, "SizeMin", h_crack)
        gmsh.model.mesh.field.setNumber(tip_threshold, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(tip_threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(tip_threshold, "DistMax", 10.0)

        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            combined, "FieldsList", [threshold, tip_threshold]
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)

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

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
        node_index = {
            int(tag): index for index, tag in enumerate(np.asarray(node_tags))
        }
        element_types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
        triangles = []
        for element_type, connectivity in zip(element_types, element_nodes):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if nodes_per_element != 3:
                continue
            tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
            triangles.append(
                np.vectorize(node_index.__getitem__, otypes=[np.int64])(tags)
            )
        if not triangles:
            raise RuntimeError("Gmsh generated no P1 triangular elements")
        tri = np.concatenate(triangles, axis=0)
        return points, tri, boundary_groups
    finally:
        gmsh.finalize()


def mesh_statistics(
    points: np.ndarray,
    triangles: np.ndarray,
    h_crack: float,
    h_coarse: float,
    preset: str,
) -> dict[str, object]:
    xy = points[triangles]
    edge_lengths = np.concatenate(
        [
            np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1),
            np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1),
            np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1),
        ]
    )
    twice_area = np.abs(
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])
    )
    area = 0.5 * twice_area
    return {
        "preset": preset,
        "geometry": {
            "width_mm": W,
            "height_mm": H,
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
            "nodes": int(points.shape[0]),
            "triangles": int(triangles.shape[0]),
            "edge_length_min_mm": float(np.min(edge_lengths)),
            "edge_length_p05_mm": float(np.percentile(edge_lengths, 5.0)),
            "edge_length_median_mm": float(np.median(edge_lengths)),
            "edge_length_p95_mm": float(np.percentile(edge_lengths, 95.0)),
            "edge_length_max_mm": float(np.max(edge_lengths)),
            "triangle_area_min_mm2": float(np.min(area)),
            "triangle_area_max_mm2": float(np.max(area)),
        },
        "boundary_groups": [
            "bottom_symmetry",
            "right",
            "top",
            "left_top",
            "notch",
            "left_impact",
        ],
    }


def save_preview(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    metadata: dict[str, object],
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.patches import Rectangle

    triangulation = mtri.Triangulation(
        points[:, 0], points[:, 1], triangles=triangles
    )
    fig = plt.figure(figsize=(15.5, 5.6), dpi=180)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.15, 1.15], wspace=0.18)
    ax_geo = fig.add_subplot(grid[0, 0])
    ax_mesh = fig.add_subplot(grid[0, 1])
    ax_zoom = fig.add_subplot(grid[0, 2])

    # Geometry and boundary-condition schematic.
    ax_geo.add_patch(
        Rectangle((0.0, 0.0), W, H, facecolor="#d7e8f5", edgecolor="black", lw=1.8)
    )
    notch = np.asarray(
        [
            [0.0, NOTCH_Y + NOTCH_HALF_WIDTH],
            [NOTCH_LENGTH, NOTCH_Y + NOTCH_HALF_WIDTH],
            [NOTCH_LENGTH, NOTCH_Y - NOTCH_HALF_WIDTH],
            [0.0, NOTCH_Y - NOTCH_HALF_WIDTH],
        ]
    )
    ax_geo.fill(notch[:, 0], notch[:, 1], color="white", zorder=3)
    ax_geo.plot(
        [0.0, NOTCH_LENGTH, NOTCH_LENGTH, 0.0],
        [
            NOTCH_Y + NOTCH_HALF_WIDTH,
            NOTCH_Y + NOTCH_HALF_WIDTH,
            NOTCH_Y - NOTCH_HALF_WIDTH,
            NOTCH_Y - NOTCH_HALF_WIDTH,
        ],
        color="black",
        lw=1.5,
        zorder=4,
    )
    for y in np.linspace(3.0, NOTCH_Y - 3.0, 6):
        ax_geo.arrow(
            -7.0,
            y,
            5.0,
            0.0,
            width=0.18,
            head_width=1.8,
            head_length=1.8,
            color="#c83e4d",
            length_includes_head=True,
        )
    for x in np.linspace(5.0, 95.0, 12):
        ax_geo.plot([x - 1.5, x, x + 1.5], [-3.0, 0.0, -3.0], color="#2c5f9e", lw=1.0)
    end_x, end_y = expected_crack_endpoint()
    ax_geo.plot(
        [NOTCH_LENGTH, end_x],
        [NOTCH_Y, end_y],
        "--",
        color="#5a5a5a",
        lw=1.2,
        label=r"expected $\sim70^\circ$ corridor",
    )
    ax_geo.text(-7.5, 12.0, r"$v_x(t)$", color="#c83e4d", ha="center", fontsize=11)
    ax_geo.text(50.0, -8.0, r"$u_y=0$ symmetry", color="#2c5f9e", ha="center", fontsize=10)
    ax_geo.text(24.0, 27.8, "50 mm notch", ha="center", fontsize=10)
    ax_geo.set_title("(a) Self-constructed half-plate geometry", loc="left", fontsize=12)
    ax_geo.legend(loc="upper left", frameon=False, fontsize=9)

    # Full mesh and crack-tip zoom.
    ax_mesh.triplot(triangulation, color="#496d89", lw=0.20, alpha=0.72)
    ax_mesh.set_title("(b) Generated P1 triangular mesh", loc="left", fontsize=12)
    ax_zoom.triplot(triangulation, color="#355f7c", lw=0.32, alpha=0.82)
    ax_zoom.plot(
        [NOTCH_LENGTH, end_x],
        [NOTCH_Y, end_y],
        "--",
        color="#c83e4d",
        lw=1.2,
    )
    ax_zoom.set_xlim(42.0, 72.0)
    ax_zoom.set_ylim(12.0, 55.0)
    ax_zoom.set_title("(c) Notch-tip refinement corridor", loc="left", fontsize=12)

    for ax in (ax_geo, ax_mesh):
        ax.set_xlim(-10.0 if ax is ax_geo else -1.0, 102.0)
        ax.set_ylim(-10.0 if ax is ax_geo else -1.0, 102.0)
    for ax in (ax_geo, ax_mesh, ax_zoom):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$ (mm)")
        ax.set_ylabel("$y$ (mm)")
        ax.tick_params(direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    mesh_data = metadata["mesh"]
    fig.suptitle(
        "Kalthoff–Winkler geometry/mesh precheck  |  "
        f"{mesh_data['nodes']:,} nodes, {mesh_data['triangles']:,} triangles  |  "
        f"h_fine={metadata['mesh_controls']['h_crack_mm']:g} mm",
        fontsize=14,
        y=0.995,
    )
    fig.subplots_adjust(top=0.91, bottom=0.11, left=0.04, right=0.99)
    fig.savefig(path, dpi=300, facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    preset_values = PRESETS[args.preset]
    h_crack = (
        args.h_crack if args.h_crack is not None else preset_values["h_crack"]
    )
    h_coarse = (
        args.h_coarse if args.h_coarse is not None else preset_values["h_coarse"]
    )
    outdir = args.outdir or (
        script_root() / "outputs" / "mesh_preview" / "kalthoff_winkler"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    token = mesh_token(h_crack)
    mesh_file = outdir / f"kalthoff_{args.preset}_h{token}.msh"
    metadata_file = outdir / f"kalthoff_{args.preset}_h{token}_metadata.json"
    preview_file = outdir / f"kalthoff_{args.preset}_h{token}_preview.png"

    points, triangles, _ = generate_mesh(mesh_file, h_crack, h_coarse)
    metadata = mesh_statistics(
        points, triangles, h_crack, h_coarse, args.preset
    )
    with metadata_file.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    save_preview(preview_file, points, triangles, metadata, args.show)

    mesh_data = metadata["mesh"]
    print("=" * 72)
    print("Kalthoff-Winkler geometry/mesh precheck completed")
    print(f"  preset:     {args.preset}")
    print(f"  mesh:       {mesh_data['nodes']:,} nodes, {mesh_data['triangles']:,} triangles")
    print(
        "  edge size:  "
        f"min={mesh_data['edge_length_min_mm']:.4g}, "
        f"p05={mesh_data['edge_length_p05_mm']:.4g}, "
        f"median={mesh_data['edge_length_median_mm']:.4g}, "
        f"p95={mesh_data['edge_length_p95_mm']:.4g} mm"
    )
    print(f"  mesh file:  {mesh_file}")
    print(f"  metadata:   {metadata_file}")
    print(f"  preview:    {preview_file}")
    print("  dynamic solve was not run")
    print("=" * 72)


if __name__ == "__main__":
    main()
