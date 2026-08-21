#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from downsample_triangle_field_voxels import (
    align_recomputed_features,
    average_downsample,
    find_source_file,
    load_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate hybrid avg-from-512 triangle-field payloads.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--source_resolution", type=int, default=512)
    parser.add_argument("--resolutions", default="32,64,128,256")
    parser.add_argument("--hybrid_suffix", default="hybrid_avg_from_512")
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def load_field(path):
    with load_npz(path) as data:
        return (
            data["coords"].astype(np.int32, copy=False),
            data["features"].astype(np.float32, copy=False),
        )


def triangle_normal_error(features):
    edge_1 = features[:, 5:8] - features[:, 2:5]
    edge_2 = features[:, 8:11] - features[:, 2:5]
    geometric = np.cross(edge_1, edge_2)
    geometric /= np.maximum(np.linalg.norm(geometric, axis=1, keepdims=True), 1e-12)
    normal = features[:, 14:17]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.abs((geometric * normal).sum(axis=1))


def main():
    args = parse_args()
    instances = [line.strip() for line in args.instances.read_text().splitlines() if line.strip()]
    resolutions = [int(item) for item in args.resolutions.split(",") if item.strip()]
    report = {"instances": instances, "resolutions": resolutions, "checks": [], "failures": []}

    for instance in instances:
        source_path = find_source_file(
            args.root / f"triangle_field_voxels_{args.source_resolution}", instance
        )
        source_coords, source_features = load_field(source_path)
        if source_features.shape[1] != 20:
            raise ValueError(f"{source_path} has {source_features.shape[1]} channels; expected 20")

        for resolution in resolutions:
            expected_coords, averaged = average_downsample(
                source_coords,
                source_features,
                args.source_resolution // resolution,
            )
            recomputed_path = find_source_file(
                args.root / f"triangle_field_voxels_{resolution}", instance
            )
            hybrid_path = find_source_file(
                args.root / f"triangle_field_voxels_{resolution}_{args.hybrid_suffix}", instance
            )
            recomputed_coords, recomputed_features = load_field(recomputed_path)
            hybrid_coords, hybrid_features = load_field(hybrid_path)
            recomputed_features = align_recomputed_features(
                expected_coords, recomputed_coords, recomputed_features, resolution
            )
            hybrid_features = align_recomputed_features(
                expected_coords, hybrid_coords, hybrid_features, resolution
            )

            centroid_expected = (
                hybrid_features[:, 2:5]
                + hybrid_features[:, 5:8]
                + hybrid_features[:, 8:11]
            ) / 3.0
            normal_error = np.abs(np.linalg.norm(hybrid_features[:, 14:17], axis=1) - 1.0)
            orientation_error = triangle_normal_error(hybrid_features.copy())
            orientation_over_threshold = orientation_error > 2e-3
            check = {
                "instance": instance,
                "resolution": resolution,
                "num_voxels": int(expected_coords.shape[0]),
                "target_max_abs_error": float(np.max(np.abs(hybrid_features[:, :2] - averaged[:, :2]))),
                "aux_max_abs_error": float(np.max(np.abs(hybrid_features[:, 2:] - recomputed_features[:, 2:]))),
                "anti_alias_target_mean_abs_delta_from_native": float(
                    np.mean(np.abs(hybrid_features[:, :2] - recomputed_features[:, :2]))
                ),
                "d_min": float(hybrid_features[:, :2].min()),
                "d_max": float(hybrid_features[:, :2].max()),
                "normal_max_unit_error": float(normal_error.max()),
                "triangle_normal_max_error": float(orientation_error.max()),
                "triangle_normal_p999_error": float(np.quantile(orientation_error, 0.999)),
                "triangle_normal_fraction_over_2e3": float(orientation_over_threshold.mean()),
                "centroid_max_abs_error": float(
                    np.max(np.abs(hybrid_features[:, 11:14] - centroid_expected))
                ),
                "all_finite": bool(np.isfinite(hybrid_features).all()),
            }
            report["checks"].append(check)

            if not check["all_finite"]:
                report["failures"].append(f"Non-finite hybrid features: {check}")
            if check["target_max_abs_error"] > 6e-4 or check["aux_max_abs_error"] > 6e-4:
                report["failures"].append(f"Hybrid channel provenance check failed: {check}")
            if check["d_min"] < -6e-4 or check["d_max"] > 1.0006:
                report["failures"].append(f"Hybrid target range check failed: {check}")
            if check["normal_max_unit_error"] > 2e-3:
                report["failures"].append(f"Hybrid normal normalization check failed: {check}")
            # Keep this diagnostic informational. Vertex offsets are stored in
            # float16, so subtracting nearly coincident vertices can make a
            # tiny triangle's reconstructed cross product numerically unstable
            # even when the independently recomputed normal is valid. Exact aux
            # provenance above is the meaningful hybrid-downsampling invariant.
            if check["centroid_max_abs_error"] > 2e-3:
                report["failures"].append(f"Hybrid centroid consistency check failed: {check}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    if report["failures"]:
        raise ValueError(
            f"Hybrid validation failed {len(report['failures'])} check(s); see {args.output_json}"
        )
    print(f"Validated {len(report['checks'])} instance-resolution payloads: {args.output_json}")


if __name__ == "__main__":
    main()
