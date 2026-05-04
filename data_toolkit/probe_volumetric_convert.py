import argparse
import faulthandler
import os
import pickle
import time

import o_voxel

from voxelize_gaussian_distance import (
    normalize_dump,
    sanitize_dump_for_volumetric_convert,
    validate_dump_for_volumetric_convert,
)


faulthandler.enable(all_threads=True)


def log(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[probe {ts}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha256", type=str, required=True, help="Asset sha256 to probe")
    parser.add_argument("--pbr_dump_root", type=str, required=True, help="Root containing pbr_dumps/")
    parser.add_argument("--resolution", type=int, default=256, help="Voxel grid resolution")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to repeat the conversion")
    parser.add_argument("--timing", action="store_true", help="Enable o_voxel timing output")
    args = parser.parse_args()

    dump_path = os.path.join(args.pbr_dump_root, "pbr_dumps", f"{args.sha256}.pickle")
    if not os.path.exists(dump_path):
        raise FileNotFoundError(f"Dump not found: {dump_path}")

    log(f"sha256={args.sha256}")
    log(f"dump_path={dump_path}")
    log(f"resolution={args.resolution}")
    log(f"repeat={args.repeat}")

    log("loading dump")
    with open(dump_path, "rb") as f:
        dump = pickle.load(f)
    log(f"raw objects={len(dump.get('objects', []))}")

    log("normalizing dump")
    dump = normalize_dump(dump)
    log(f"normalized objects={len(dump.get('objects', []))}")

    sanitize_changes = sanitize_dump_for_volumetric_convert(dump)
    for change in sanitize_changes:
        log(f"sanitize: {change}")

    report = validate_dump_for_volumetric_convert(dump)
    log(f"materials={report['num_materials']}")
    for summary in report['summaries']:
        log(summary)
    for warning in report['warnings']:
        log(f"validation warning: {warning}")
    for error in report['errors']:
        log(f"validation error: {error}")
    if report['errors']:
        raise ValueError("validation failed before native convert")

    for run_idx in range(args.repeat):
        log(f"convert run {run_idx + 1}/{args.repeat} start")
        coords, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
            dump,
            grid_size=args.resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            mip_level_offset=0,
            verbose=False,
            timing=args.timing,
        )
        log(f"convert run {run_idx + 1}/{args.repeat} done coords={len(coords)} attrs={list(attr.keys())}")

    log("probe completed")


if __name__ == "__main__":
    main()
