"""
Build a triangle mesh from d_tri/d_vert with Barycentric Ridge Graph completion.

This is the final mesh stage on top of Barycentric Ridge Graph (BRG):
  1. Run BRG to get vertex candidates and conservative edge-ridge edges.
  2. Add midpoint-anchor edges from blobs where d_tri is low and d_vert is near 0.25.
  3. Extract triangle faces as 3-cycles in the completed vertex graph.
  4. Write OBJ, PLY, and NPZ outputs.

Only coords, d_tri, and d_vert are used. Offset channels are ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extract_barycentric_ridge_graph import (
    ALGO_SHORT_NAME,
    connected_components,
    default_output_stem,
    default_results_dir,
    extract_brg,
    infer_resolution,
    load_triangle_field,
    threshold_to_raw_max_barycentric,
    voxel_centers,
)


MESH_ALGO_NAME = "Barycentric Ridge Graph Triangle Mesh"


def sorted_edge_tuple(edge: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        raise ValueError("self-edge is not allowed")
    return (a, b) if a < b else (b, a)


def edge_set_to_array(edges: set[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edges), dtype=np.int32)


def midpoint_anchor_edges(
    coords: np.ndarray,
    d_tri: np.ndarray,
    d_vert: np.ndarray,
    vertices: np.ndarray,
    resolution: int,
    *,
    midpoint_dtri_threshold: float = 0.25,
    midpoint_dvert: float = 0.25,
    midpoint_tolerance: float = 0.05,
    midpoint_connectivity: int = 18,
    midpoint_min_component_size: int = 1,
    max_anchor_distance: float | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """
    Add edges from midpoint-like blobs by connecting each blob to its two nearest vertices.

    A true edge midpoint has min(a,b,c)=0 and max(a,b,c)=0.5, which is
    d_tri=0 and d_vert=0.25 in the stored normalized channels.
    """
    d_tri = np.asarray(d_tri, dtype=np.float32).reshape(-1)
    d_vert = np.asarray(d_vert, dtype=np.float32).reshape(-1)
    mask = (
        (d_tri <= float(midpoint_dtri_threshold))
        & (np.abs(d_vert - float(midpoint_dvert)) <= float(midpoint_tolerance))
    )
    components = connected_components(coords, mask, connectivity=midpoint_connectivity)

    pairs: set[tuple[int, int]] = set()
    rejected_small = 0
    rejected_distance = 0
    for component in components:
        if component.size < midpoint_min_component_size:
            rejected_small += 1
            continue
        center = voxel_centers(coords[component], resolution).mean(axis=0)
        dist2 = np.sum((vertices - center[None, :]) ** 2, axis=1)
        nearest = np.argpartition(dist2, 2)[:2]
        if max_anchor_distance is not None and np.sqrt(dist2[nearest].max()) > max_anchor_distance:
            rejected_distance += 1
            continue
        pairs.add(sorted_edge_tuple((int(nearest[0]), int(nearest[1]))))

    stats = {
        "midpoint_mask_voxels": int(mask.sum()),
        "midpoint_components": int(len(components)),
        "midpoint_edges": int(len(pairs)),
        "midpoint_rejected_small": int(rejected_small),
        "midpoint_rejected_distance": int(rejected_distance),
        "midpoint_dtri_threshold": float(midpoint_dtri_threshold),
        "midpoint_dvert": float(midpoint_dvert),
        "midpoint_tolerance": float(midpoint_tolerance),
    }
    return edge_set_to_array(pairs), stats


def adjacency_from_edges(num_vertices: int, edges: np.ndarray) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        if a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def triangle_faces_from_edges(
    vertices: np.ndarray,
    edges: np.ndarray,
    *,
    min_face_area: float = 1e-12,
    orient_outward: bool = True,
) -> np.ndarray:
    adjacency = adjacency_from_edges(len(vertices), edges)
    faces = []
    center = vertices.mean(axis=0) if len(vertices) else np.zeros(3, dtype=np.float32)

    for i, neighbors_i in enumerate(adjacency):
        for j in neighbors_i:
            if j <= i:
                continue
            common = neighbors_i.intersection(adjacency[j])
            for k in common:
                if k <= j:
                    continue
                face = [i, int(j), int(k)]
                p0, p1, p2 = vertices[face]
                normal = np.cross(p1 - p0, p2 - p0)
                area2 = float(np.linalg.norm(normal))
                if area2 * 0.5 < min_face_area:
                    continue
                if orient_outward:
                    face_center = (p0 + p1 + p2) / 3.0
                    if float(np.dot(normal, face_center - center)) < 0.0:
                        face = [i, int(k), int(j)]
                faces.append(face)

    if not faces:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(faces, dtype=np.int32)


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {MESH_ALGO_NAME}\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            f.write(f"f {a} {b} {c}\n")


def write_ply_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment algorithm {MESH_ALGO_NAME}\n")
        f.write(f"comment source_graph {ALGO_SHORT_NAME}\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for edge in edges:
            f.write(f"{int(edge[0])} {int(edge[1])}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1).astype(np.float32)


def write_mesh_npz(
    path: str | Path,
    *,
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    base_edges: np.ndarray,
    midpoint_edges: np.ndarray,
    graph: dict[str, np.ndarray | int | float],
    midpoint_stats: dict[str, int | float],
    resolution: int,
) -> None:
    np.savez_compressed(
        path,
        vertices=vertices.astype(np.float32, copy=False),
        edges=edges.astype(np.int32, copy=False),
        faces=faces.astype(np.int32, copy=False),
        base_brg_edges=base_edges.astype(np.int32, copy=False),
        midpoint_anchor_edges=midpoint_edges.astype(np.int32, copy=False),
        resolution=np.asarray(resolution, dtype=np.int32),
        algorithm=np.asarray(MESH_ALGO_NAME),
        vertex_dvert_threshold=np.asarray(graph["vertex_dvert_threshold"], dtype=np.float32),
        vertex_core_mode=np.asarray(graph["vertex_core_mode"]),
        vertex_core_count=graph["vertex_core_count"].astype(np.int32, copy=False),
        vertex_blob_voxels=np.asarray(graph["num_vertex_blob_voxels"], dtype=np.int32),
        vertex_core_voxels=np.asarray(graph["num_vertex_core_voxels"], dtype=np.int32),
        vertex_clearance_voxels=np.asarray(graph["num_vertex_clearance_voxels"], dtype=np.int32),
        edge_dtri_threshold=np.asarray(graph["edge_dtri_threshold"], dtype=np.float32),
        edge_min_dvert=np.asarray(graph["edge_min_dvert"], dtype=np.float32),
        edge_max_dvert=np.asarray(graph["edge_max_dvert"], dtype=np.float32),
        edge_attach_mode=np.asarray(graph["edge_attach_mode"]),
        edge_evidence_candidate_vertices=np.asarray(graph["edge_evidence_candidate_vertices"], dtype=np.int32),
        edge_evidence_max_segment_distance=np.asarray(graph["edge_evidence_max_segment_distance"], dtype=np.float32),
        edge_evidence_max_pair_distance=np.asarray(graph["edge_evidence_max_pair_distance"], dtype=np.float32),
        edge_evidence_projection_margin=np.asarray(graph["edge_evidence_projection_margin"], dtype=np.float32),
        edge_bridge_mode=np.asarray(graph["edge_bridge_mode"]),
        edge_bridge_max_distance=np.asarray(graph["edge_bridge_max_distance"], dtype=np.float32),
        edge_bridge_enabled=np.asarray(graph["edge_bridge_enabled"], dtype=np.int32),
        edge_bridge_midpoint_dtri_threshold=np.asarray(
            graph["edge_bridge_midpoint_dtri_threshold"],
            dtype=np.float32,
        ),
        edge_bridge_midpoint_dvert=np.asarray(graph["edge_bridge_midpoint_dvert"], dtype=np.float32),
        edge_bridge_midpoint_tolerance=np.asarray(graph["edge_bridge_midpoint_tolerance"], dtype=np.float32),
        edge_bridge_midpoint_min_component_size=np.asarray(
            graph["edge_bridge_midpoint_min_component_size"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_max_component_size=np.asarray(
            graph["edge_bridge_midpoint_max_component_size"],
            dtype=np.int32,
        ),
        edge_components_before_bridge=np.asarray(graph["edge_components_before_bridge"], dtype=np.int32),
        edge_components_after_bridge=np.asarray(graph["edge_components_after_bridge"], dtype=np.int32),
        edge_bridge_midpoint_voxels=np.asarray(graph["edge_bridge_midpoint_voxels"], dtype=np.int32),
        edge_bridge_midpoint_components=np.asarray(graph["edge_bridge_midpoint_components"], dtype=np.int32),
        edge_bridge_midpoint_components_used=np.asarray(
            graph["edge_bridge_midpoint_components_used"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_no_edge=np.asarray(
            graph["edge_bridge_midpoint_components_no_edge"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_single_edge=np.asarray(
            graph["edge_bridge_midpoint_components_single_edge"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_ambiguous=np.asarray(
            graph["edge_bridge_midpoint_components_ambiguous"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_too_small=np.asarray(
            graph["edge_bridge_midpoint_components_too_small"],
            dtype=np.int32,
        ),
        edge_bridge_midpoint_components_too_large=np.asarray(
            graph["edge_bridge_midpoint_components_too_large"],
            dtype=np.int32,
        ),
        edge_bridge_unions=np.asarray(graph["edge_bridge_unions"], dtype=np.int32),
        edge_evidence_components=graph["edge_evidence_components"].astype(np.int32, copy=False),
        evidence_voted_components=np.asarray(graph["num_evidence_voted_components"], dtype=np.int32),
        rejected_bad_evidence_fit_components=np.asarray(
            graph["num_rejected_bad_evidence_fit_components"],
            dtype=np.int32,
        ),
        vertex_clearance=np.asarray(graph["vertex_clearance"], dtype=np.int32),
        attach_radius=np.asarray(graph["attach_radius"], dtype=np.int32),
        brg_edges=np.asarray(graph["edges"].shape[0], dtype=np.int32),
        midpoint_dtri_threshold=np.asarray(midpoint_stats["midpoint_dtri_threshold"], dtype=np.float32),
        midpoint_dvert=np.asarray(midpoint_stats["midpoint_dvert"], dtype=np.float32),
        midpoint_tolerance=np.asarray(midpoint_stats["midpoint_tolerance"], dtype=np.float32),
        midpoint_mask_voxels=np.asarray(midpoint_stats["midpoint_mask_voxels"], dtype=np.int32),
        midpoint_components=np.asarray(midpoint_stats["midpoint_components"], dtype=np.int32),
    )


def build_mesh(
    coords: np.ndarray,
    features: np.ndarray,
    resolution: int,
    args,
) -> tuple[
    dict[str, np.ndarray | int | float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | float],
]:
    d_tri = features[:, 0]
    d_vert = features[:, 1]
    graph = extract_brg(
        coords,
        d_tri,
        d_vert,
        resolution,
        vertex_dvert_threshold=args.vertex_dvert_threshold,
        vertex_connectivity=args.vertex_connectivity,
        vertex_min_component_size=args.vertex_min_component_size,
        vertex_position_mode=args.vertex_position_mode,
        vertex_core_mode=args.vertex_core_mode,
        edge_dtri_threshold=args.edge_dtri_threshold,
        edge_min_dvert=args.edge_min_dvert,
        edge_max_dvert=args.edge_max_dvert,
        edge_connectivity=args.edge_connectivity,
        edge_min_component_size=args.edge_min_component_size,
        edge_attach_mode=args.edge_attach_mode,
        edge_evidence_candidate_vertices=args.edge_evidence_candidate_vertices,
        edge_evidence_max_segment_distance=args.edge_evidence_max_segment_distance,
        edge_evidence_max_pair_distance=args.edge_evidence_max_pair_distance,
        edge_evidence_projection_margin=args.edge_evidence_projection_margin,
        edge_bridge_max_distance=args.edge_bridge_max_distance,
        edge_bridge_midpoint_dtri_threshold=(
            args.edge_bridge_midpoint_dtri_threshold
            if args.edge_bridge_midpoint_dtri_threshold is not None
            else args.midpoint_dtri_threshold
        ),
        edge_bridge_midpoint_tolerance=(
            args.edge_bridge_midpoint_tolerance
            if args.edge_bridge_midpoint_tolerance is not None
            else args.midpoint_tolerance
        ),
        edge_bridge_midpoint_connectivity=args.edge_bridge_midpoint_connectivity,
        edge_bridge_midpoint_min_component_size=args.edge_bridge_midpoint_min_component_size,
        edge_bridge_midpoint_max_component_size=args.edge_bridge_midpoint_max_component_size,
        vertex_clearance=args.vertex_clearance,
        attach_radius=args.attach_radius,
        midpoint_dvert=args.ridge_midpoint_dvert,
        midpoint_tolerance=args.ridge_midpoint_tolerance,
        require_midpoint=not args.no_ridge_midpoint_check,
    )

    vertices = graph["vertices"]
    base_edges = graph["edges"].astype(np.int32, copy=False)
    midpoint_edges, midpoint_stats = midpoint_anchor_edges(
        coords,
        d_tri,
        d_vert,
        vertices,
        resolution,
        midpoint_dtri_threshold=args.midpoint_dtri_threshold,
        midpoint_dvert=args.midpoint_dvert,
        midpoint_tolerance=args.midpoint_tolerance,
        midpoint_connectivity=args.midpoint_connectivity,
        midpoint_min_component_size=args.midpoint_min_component_size,
        max_anchor_distance=args.max_anchor_distance,
    )

    edge_set = {sorted_edge_tuple(edge) for edge in base_edges}
    if not args.no_midpoint_completion:
        edge_set.update(sorted_edge_tuple(edge) for edge in midpoint_edges)
    edges = edge_set_to_array(edge_set)
    faces = triangle_faces_from_edges(
        vertices,
        edges,
        min_face_area=args.min_face_area,
        orient_outward=not args.no_orient_outward,
    )
    return graph, vertices, edges, faces, midpoint_edges, midpoint_stats


def print_summary(
    graph: dict[str, np.ndarray | int | float],
    vertices: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    midpoint_stats: dict[str, int | float],
    resolution: int,
) -> None:
    lengths = edge_lengths(vertices, edges)
    print(f"Algorithm: {MESH_ALGO_NAME}")
    print(f"Resolution: {resolution}")
    print(f"Vertices: {len(vertices)}")
    print(f"Base BRG edges: {graph['edges'].shape[0]}")
    print(f"Midpoint-anchor edges: {midpoint_stats['midpoint_edges']}")
    print(f"Completed unique edges: {len(edges)}")
    print(f"Triangle faces from 3-cycles: {len(faces)}")
    if lengths.size:
        print(
            "Edge length stats: "
            f"min={lengths.min():.6f}, "
            f"mean={lengths.mean():.6f}, "
            f"p95={np.quantile(lengths, 0.95):.6f}, "
            f"max={lengths.max():.6f}"
        )
    print("BRG thresholds:")
    print(
        f"  vertex d_vert >= {graph['vertex_dvert_threshold']} "
        f"(raw max barycentric >= {threshold_to_raw_max_barycentric(graph['vertex_dvert_threshold']):.6f})"
    )
    print(f"  vertex core mode = {graph['vertex_core_mode']}")
    print(f"  vertex blob voxels = {graph['num_vertex_blob_voxels']}")
    print(f"  vertex core voxels = {graph['num_vertex_core_voxels']}")
    print(f"  vertex clearance voxels = {graph['num_vertex_clearance_voxels']}")
    print(f"  ridge edge d_tri <= {graph['edge_dtri_threshold']}")
    print(f"  ridge edge d_vert in [{graph['edge_min_dvert']}, {graph['edge_max_dvert']}]")
    print(f"  edge attach mode = {graph['edge_attach_mode']}")
    print(f"  evidence voted components = {graph['num_evidence_voted_components']}")
    print(f"  rejected bad evidence fit components = {graph['num_rejected_bad_evidence_fit_components']}")
    if graph["edge_attach_mode"] == "evidence":
        print(f"  evidence candidate vertices = {graph['edge_evidence_candidate_vertices']}")
        print(f"  evidence max segment distance = {graph['edge_evidence_max_segment_distance']} voxel(s)")
        print(f"  evidence max pair distance = {graph['edge_evidence_max_pair_distance']} voxel(s)")
        print(f"  evidence projection margin = {graph['edge_evidence_projection_margin']} voxel(s)")
    print(f"  edge bridge mode = {graph['edge_bridge_mode']}")
    print(f"  edge bridge search radius = {graph['edge_bridge_max_distance']} voxel(s)")
    print(f"  edge bridge midpoint d_tri <= {graph['edge_bridge_midpoint_dtri_threshold']}")
    print(
        f"  edge bridge midpoint abs(d_vert - {graph['edge_bridge_midpoint_dvert']}) "
        f"<= {graph['edge_bridge_midpoint_tolerance']}"
    )
    print(f"  edge bridge midpoint voxels = {graph['edge_bridge_midpoint_voxels']}")
    print(f"  edge bridge midpoint components = {graph['edge_bridge_midpoint_components']}")
    print(f"  edge bridge midpoint components used = {graph['edge_bridge_midpoint_components_used']}")
    print(f"  edge bridge midpoint components single-edge = {graph['edge_bridge_midpoint_components_single_edge']}")
    print(f"  edge bridge midpoint components ambiguous = {graph['edge_bridge_midpoint_components_ambiguous']}")
    print(f"  edge components before bridge = {graph['edge_components_before_bridge']}")
    print(f"  edge components after bridge = {graph['edge_components_after_bridge']}")
    print(f"  edge bridge unions = {graph['edge_bridge_unions']}")
    print(f"  vertex clearance radius from core = {graph['vertex_clearance']}")
    print(f"  attach radius = {graph['attach_radius']}")
    print("Midpoint completion thresholds:")
    print(f"  midpoint d_tri <= {midpoint_stats['midpoint_dtri_threshold']}")
    print(
        f"  abs(d_vert - {midpoint_stats['midpoint_dvert']}) "
        f"<= {midpoint_stats['midpoint_tolerance']}"
    )
    print(f"  midpoint mask voxels = {midpoint_stats['midpoint_mask_voxels']}")
    print(f"  midpoint components = {midpoint_stats['midpoint_components']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a triangle OBJ/PLY mesh from d_tri/d_vert using BRG plus midpoint completion."
    )
    parser.add_argument("path", type=str, help="Path to triangle-field .npz or .npz.zst")
    parser.add_argument("--resolution", type=int, default=None)

    parser.add_argument("--vertex_dvert_threshold", type=float, default=0.84)
    parser.add_argument("--vertex_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--vertex_min_component_size", type=int, default=1)
    parser.add_argument("--vertex_position_mode", choices=["peak", "mean", "weighted"], default="weighted")
    parser.add_argument(
        "--vertex_core_mode",
        choices=["closest4", "peak", "position_nearest", "blob", "none"],
        default="closest4",
    )

    parser.add_argument("--edge_dtri_threshold", type=float, default=0.175)
    parser.add_argument("--edge_min_dvert", type=float, default=0.25)
    parser.add_argument("--edge_max_dvert", type=float, default=None)
    parser.add_argument("--edge_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--edge_min_component_size", type=int, default=2)
    parser.add_argument("--edge_attach_mode", choices=["evidence", "endpoints", "contacts"], default="evidence")
    parser.add_argument("--edge_evidence_candidate_vertices", type=int, default=8)
    parser.add_argument("--edge_evidence_max_segment_distance", type=float, default=3.0)
    parser.add_argument("--edge_evidence_max_pair_distance", type=float, default=64.0)
    parser.add_argument("--edge_evidence_projection_margin", type=float, default=4.0)
    parser.add_argument("--edge_bridge_max_distance", type=float, default=4.0)
    parser.add_argument("--edge_bridge_midpoint_dtri_threshold", type=float, default=None)
    parser.add_argument("--edge_bridge_midpoint_tolerance", type=float, default=None)
    parser.add_argument("--edge_bridge_midpoint_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--edge_bridge_midpoint_min_component_size", type=int, default=1)
    parser.add_argument("--edge_bridge_midpoint_max_component_size", type=int, default=128)
    parser.add_argument("--vertex_clearance", type=int, default=1)
    parser.add_argument("--attach_radius", type=int, default=3)
    parser.add_argument("--ridge_midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--ridge_midpoint_tolerance", type=float, default=0.075)
    parser.add_argument("--no_ridge_midpoint_check", action="store_true")

    parser.add_argument("--no_midpoint_completion", action="store_true")
    parser.add_argument("--midpoint_dtri_threshold", type=float, default=0.25)
    parser.add_argument("--midpoint_dvert", type=float, default=0.25)
    parser.add_argument("--midpoint_tolerance", type=float, default=0.05)
    parser.add_argument("--midpoint_connectivity", type=int, default=18, choices=[6, 18, 26])
    parser.add_argument("--midpoint_min_component_size", type=int, default=1)
    parser.add_argument("--max_anchor_distance", type=float, default=None)

    parser.add_argument("--min_face_area", type=float, default=1e-12)
    parser.add_argument("--no_orient_outward", action="store_true")
    parser.add_argument("--out_obj", type=str, default=None)
    parser.add_argument("--out_ply", type=str, default=None)
    parser.add_argument("--out_npz", type=str, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    coords, features = load_triangle_field(path)
    resolution = args.resolution if args.resolution is not None else infer_resolution(coords)
    stem = default_output_stem(path)

    graph, vertices, edges, faces, midpoint_edges, midpoint_stats = build_mesh(coords, features, resolution, args)

    out_dir = default_results_dir(path)
    out_obj = Path(args.out_obj) if args.out_obj else out_dir / f"{stem}_brg_triangle_mesh.obj"
    out_ply = Path(args.out_ply) if args.out_ply else out_dir / f"{stem}_brg_triangle_mesh.ply"
    out_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{stem}_brg_triangle_mesh.npz"

    write_obj(out_obj, vertices, faces)
    write_ply_mesh(out_ply, vertices, faces, edges)
    write_mesh_npz(
        out_npz,
        vertices=vertices,
        edges=edges,
        faces=faces,
        base_edges=graph["edges"],
        midpoint_edges=midpoint_edges,
        graph=graph,
        midpoint_stats=midpoint_stats,
        resolution=resolution,
    )

    print_summary(graph, vertices, edges, faces, midpoint_stats, resolution)
    print(f"Saved {out_obj}")
    print(f"Saved {out_ply}")
    print(f"Saved {out_npz}")


if __name__ == "__main__":
    main()
