import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pyiges


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_INPUT = os.path.join(BASE_DIR, "老模型中心线.igs")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "exports", "terminal_branches")


def read_centerline_curves(path):
    curves = []
    for entity in pyiges.read(path):
        parameters = entity.parameters
        point_count = int(parameters[0][2])
        values = [float(value.strip()) for value in parameters[0][3:]]
        points = np.asarray(values[: point_count * 3], dtype=float).reshape(-1, 3)
        if len(points) >= 2:
            curves.append(points)
    if not curves:
        raise ValueError(f"No centerline curves found in {path}")
    return curves


def endpoint_key(point):
    return tuple(float(value) for value in point)


def build_endpoint_tree(curves):
    adjacency = defaultdict(list)
    for curve_index, curve in enumerate(curves):
        start = endpoint_key(curve[0])
        end = endpoint_key(curve[-1])
        adjacency[start].append((end, curve_index))
        adjacency[end].append((start, curve_index))

    node_count = len(adjacency)
    edge_count = len(curves)
    if edge_count != node_count - 1:
        raise ValueError(
            f"Centerline topology is not a tree: {node_count} nodes, "
            f"{edge_count} edges"
        )

    leaves = [node for node, edges in adjacency.items() if len(edges) == 1]
    if len(leaves) < 2:
        raise ValueError("Centerline tree has fewer than two leaf nodes")

    # The tracheal inlet is the inferior-most leaf in this model coordinate system.
    root = min(leaves, key=lambda node: (node[1], node[0], node[2]))
    parent = {root: (None, None)}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor, curve_index in sorted(
            adjacency[node],
            key=lambda item: item[1],
            reverse=True,
        ):
            if neighbor in parent:
                continue
            parent[neighbor] = (node, curve_index)
            stack.append(neighbor)

    if len(parent) != node_count:
        raise ValueError(
            f"Centerline topology is disconnected: reached {len(parent)} of "
            f"{node_count} nodes"
        )
    terminals = [leaf for leaf in leaves if leaf != root]
    terminals.sort(key=lambda node: adjacency[node][0][1])
    return adjacency, root, terminals, parent


def path_edges_to_terminal(root, terminal, parent):
    edges = []
    node = terminal
    while node != root:
        previous, curve_index = parent[node]
        edges.append((previous, node, curve_index))
        node = previous
    edges.reverse()
    return edges


def orient_curve(curve, start_node):
    if endpoint_key(curve[0]) == start_node:
        return curve
    if endpoint_key(curve[-1]) == start_node:
        return curve[::-1]
    raise ValueError("Curve endpoint does not match its topology node")


def build_trajectory(curves, root, terminal, parent):
    path_edges = path_edges_to_terminal(root, terminal, parent)
    records = []
    cumulative_length = 0.0
    previous_point = None

    for path_edge_index, (start_node, _, curve_index) in enumerate(path_edges):
        oriented = orient_curve(curves[curve_index], start_node)
        first_point = 0 if path_edge_index == 0 else 1
        for curve_point_index in range(first_point, len(oriented)):
            point = oriented[curve_point_index]
            segment_length = (
                0.0
                if previous_point is None
                else float(np.linalg.norm(point - previous_point))
            )
            cumulative_length += segment_length
            records.append(
                {
                    "path_edge_index": path_edge_index,
                    "source_curve_index": curve_index,
                    "curve_point_index": curve_point_index,
                    "point": point.copy(),
                    "segment_length_mm": segment_length,
                    "cumulative_length_mm": cumulative_length,
                    "is_terminal_segment": int(
                        path_edge_index == len(path_edges) - 1
                    ),
                }
            )
            previous_point = point
    return path_edges, records


def export_terminal_trajectories(input_path, output_dir):
    curves = read_centerline_curves(input_path)
    _, root, terminals, parent = build_endpoint_tree(curves)
    os.makedirs(output_dir, exist_ok=True)

    detail_path = os.path.join(output_dir, "terminal_branch_trajectories_65.csv")
    summary_path = os.path.join(output_dir, "terminal_branch_summary_65.csv")
    detail_headers = [
        "BranchID",
        "TerminalOrder",
        "PathPointIndex",
        "PathEdgeIndex",
        "SourceCurveIndex",
        "CurvePointIndex",
        "X_mm",
        "Y_mm",
        "Z_mm",
        "SegmentLength_mm",
        "CumulativeLength_mm",
        "NormalizedProgress",
        "IsTerminalSegment",
        "Terminal_X_mm",
        "Terminal_Y_mm",
        "Terminal_Z_mm",
        "TotalPathLength_mm",
        "PathEdgeCount",
    ]
    summary_headers = [
        "BranchID",
        "TerminalOrder",
        "Terminal_X_mm",
        "Terminal_Y_mm",
        "Terminal_Z_mm",
        "TotalPathLength_mm",
        "PathEdgeCount",
        "PathPointCount",
        "TerminalSourceCurveIndex",
    ]

    summaries = []
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as detail_file:
        writer = csv.DictWriter(detail_file, fieldnames=detail_headers)
        writer.writeheader()
        for terminal_order, terminal in enumerate(terminals, start=1):
            branch_id = f"Branch_{terminal_order:03d}"
            path_edges, records = build_trajectory(curves, root, terminal, parent)
            total_length = records[-1]["cumulative_length_mm"]
            for point_index, record in enumerate(records):
                point = record["point"]
                writer.writerow(
                    {
                        "BranchID": branch_id,
                        "TerminalOrder": terminal_order,
                        "PathPointIndex": point_index,
                        "PathEdgeIndex": record["path_edge_index"],
                        "SourceCurveIndex": record["source_curve_index"],
                        "CurvePointIndex": record["curve_point_index"],
                        "X_mm": f"{point[0]:.9f}",
                        "Y_mm": f"{point[1]:.9f}",
                        "Z_mm": f"{point[2]:.9f}",
                        "SegmentLength_mm": f"{record['segment_length_mm']:.9f}",
                        "CumulativeLength_mm": f"{record['cumulative_length_mm']:.9f}",
                        "NormalizedProgress": (
                            f"{record['cumulative_length_mm'] / total_length:.9f}"
                            if total_length > 0
                            else "0.000000000"
                        ),
                        "IsTerminalSegment": record["is_terminal_segment"],
                        "Terminal_X_mm": f"{terminal[0]:.9f}",
                        "Terminal_Y_mm": f"{terminal[1]:.9f}",
                        "Terminal_Z_mm": f"{terminal[2]:.9f}",
                        "TotalPathLength_mm": f"{total_length:.9f}",
                        "PathEdgeCount": len(path_edges),
                    }
                )
            summaries.append(
                {
                    "BranchID": branch_id,
                    "TerminalOrder": terminal_order,
                    "Terminal_X_mm": f"{terminal[0]:.9f}",
                    "Terminal_Y_mm": f"{terminal[1]:.9f}",
                    "Terminal_Z_mm": f"{terminal[2]:.9f}",
                    "TotalPathLength_mm": f"{total_length:.9f}",
                    "PathEdgeCount": len(path_edges),
                    "PathPointCount": len(records),
                    "TerminalSourceCurveIndex": path_edges[-1][2],
                }
            )

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summaries)

    return {
        "input_path": os.path.abspath(input_path),
        "root": np.asarray(root, dtype=float),
        "terminal_count": len(terminals),
        "detail_path": os.path.abspath(detail_path),
        "summary_path": os.path.abspath(summary_path),
        "detail_row_count": sum(int(item["PathPointCount"]) for item in summaries),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export root-to-terminal trajectories for every airway leaf."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = export_terminal_trajectories(args.input, args.output_dir)
    print(f"Terminal branches: {result['terminal_count']}")
    print(f"Tracheal inlet: {result['root'].tolist()}")
    print(f"Trajectory rows: {result['detail_row_count']}")
    print(f"Detail CSV: {result['detail_path']}")
    print(f"Summary CSV: {result['summary_path']}")


if __name__ == "__main__":
    main()
