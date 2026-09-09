#!/usr/bin/env python3
"""Build a reduced non-convex collision surface from the visual bronchus STL."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_INPUT = PROJECT_ROOT / "meshes" / "part" / "bronchus.stl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "meshes" / "part" / "bronchus_collision_solid_nonconvex.stl"
)


def surface_error_mm(
    reference: o3d.geometry.TriangleMesh,
    candidate: o3d.geometry.TriangleMesh,
    samples: int,
) -> np.ndarray:
    points = reference.sample_points_uniformly(number_of_points=samples)
    query = o3d.core.Tensor(np.asarray(points.points), dtype=o3d.core.Dtype.Float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(candidate))
    return scene.compute_distance(query).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # 20k keeps the sampled p99 surface error below 0.25 mm; the former 8k
    # surface could deviate visibly from the rendered airway at narrow bends.
    parser.add_argument("--triangles", type=int, default=20_000)
    parser.add_argument("--boundary-weight", type=float, default=20.0)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()

    source = o3d.io.read_triangle_mesh(str(args.input.resolve()))
    if source.is_empty():
        raise RuntimeError(f"Could not read bronchial mesh: {args.input}")
    source.remove_duplicated_vertices()
    source.remove_duplicated_triangles()
    source.remove_degenerate_triangles()
    source.remove_unreferenced_vertices()

    reduced = source.simplify_quadric_decimation(
        target_number_of_triangles=args.triangles,
        boundary_weight=args.boundary_weight,
    )
    reduced.remove_degenerate_triangles()
    reduced.remove_duplicated_triangles()
    reduced.remove_unreferenced_vertices()
    reduced.compute_triangle_normals()

    # Check both directions: original-to-reduced detects lost detail, while
    # reduced-to-original detects triangles cutting across airway branches.
    forward = surface_error_mm(source, reduced, args.samples)
    reverse = surface_error_mm(reduced, source, args.samples)
    combined = np.concatenate((forward, reverse))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(
        str(args.output.resolve()), reduced, write_ascii=False
    ):
        raise RuntimeError(f"Could not write reduced mesh: {args.output}")

    print(
        f"Collision mesh: {len(source.triangles)} -> {len(reduced.triangles)} "
        f"triangles, {len(reduced.vertices)} vertices"
    )
    print(
        "Bidirectional sampled surface error (source units are mm): "
        f"median={np.median(combined):.4f}, "
        f"p95={np.percentile(combined, 95):.4f}, "
        f"p99={np.percentile(combined, 99):.4f}, "
        f"max={np.max(combined):.4f}"
    )
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
