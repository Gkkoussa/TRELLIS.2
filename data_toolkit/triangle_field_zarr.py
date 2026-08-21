import io
import zipfile
from pathlib import Path

import numpy as np
import zarr
import zstandard as zstd
from numcodecs import Blosc


FEATURE_LAYOUT = [
    ["d_tri", 1],
    ["d_vert", 1],
    ["offset_to_v0", 3],
    ["offset_to_v1", 3],
    ["offset_to_v2", 3],
    ["offset_to_centroid", 3],
    ["face_normal", 3],
    ["offset_to_projection", 3],
]


def find_payload(root: Path, instance: str) -> Path:
    for suffix in (".npz.zst", ".npz"):
        path = Path(root) / f"{instance}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No payload for {instance} under {root}")


def load_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.name.endswith(".npz.zst"):
        payload = zstd.ZstdDecompressor().decompress(path.read_bytes())
        data = np.load(io.BytesIO(payload), allow_pickle=False)
    else:
        data = np.load(path, allow_pickle=False)
    with data:
        return (
            data["coords"].astype(np.int32, copy=True),
            data["features"].astype(np.float16, copy=True),
        )


def source_roots(
    dataset_root: Path,
    resolutions: list[int],
    lower_suffix: str,
) -> dict[int, Path]:
    return {
        resolution: Path(dataset_root) / (
            f"triangle_field_voxels_{resolution}"
            if resolution == 512
            else f"triangle_field_voxels_{resolution}_{lower_suffix}"
        )
        for resolution in resolutions
    }


def pack_zipstore(
    output_path: Path,
    roots: dict[int, Path],
    instances: list[str],
    resolutions: list[int],
    chunk_voxels: int,
    compression_level: int,
    attrs: dict | None = None,
) -> dict[int, int]:
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.ZipStore(
        str(output_path),
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    )
    voxel_counts = {}
    try:
        root = zarr.group(store=store, overwrite=True)
        metadata = {
            "format": "triangle_field_zarr_shard",
            "version": 1,
            "instances": len(instances),
            "resolutions": resolutions,
            "feature_layout": FEATURE_LAYOUT,
        }
        if attrs:
            metadata.update(attrs)
        root.attrs.update(metadata)
        root.array(
            "sha256",
            np.asarray(instances, dtype="S64"),
            chunks=(len(instances),),
            compressor=None,
        )

        for resolution in resolutions:
            coords_parts = []
            feature_parts = []
            offsets = [0]
            for instance in instances:
                coords, features = load_payload(find_payload(roots[resolution], instance))
                if features.ndim != 2 or features.shape[1] != 20:
                    raise ValueError(f"{instance} r{resolution} has feature shape {features.shape}")
                if coords.min(initial=0) < 0 or coords.max(initial=0) >= resolution:
                    raise ValueError(f"{instance} r{resolution} has out-of-range coordinates")
                coords_parts.append(coords.astype(np.uint16, copy=False))
                feature_parts.append(features)
                offsets.append(offsets[-1] + len(coords))

            coords = np.concatenate(coords_parts, axis=0)
            features = np.concatenate(feature_parts, axis=0)
            voxel_counts[resolution] = len(coords)
            group = root.create_group(f"r{resolution}")
            group.array(
                "offsets",
                np.asarray(offsets, dtype=np.int64),
                chunks=(len(offsets),),
                compressor=None,
            )
            group.array(
                "coords",
                coords,
                chunks=(min(chunk_voxels, max(len(coords), 1)), 3),
                compressor=compressor,
            )
            group.array(
                "triangle_field",
                features,
                chunks=(min(chunk_voxels, max(len(features), 1)), features.shape[1]),
                compressor=compressor,
            )
        zarr.consolidate_metadata(store)
    finally:
        store.close()
    return voxel_counts


def open_zipstore(path: Path):
    store = zarr.ZipStore(str(path), mode="r")
    return store, zarr.open_consolidated(store=store, mode="r")


def read_zarr_sample(root, resolution: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    group = root[f"r{resolution}"]
    start, end = np.asarray(group["offsets"][index:index + 2], dtype=np.int64)
    return (
        np.asarray(group["coords"][start:end], dtype=np.int32),
        np.asarray(group["triangle_field"][start:end], dtype=np.float16),
    )


def verify_zipstore(
    path: Path,
    roots: dict[int, Path],
    instances: list[str],
    resolutions: list[int],
) -> dict[int, int]:
    store, root = open_zipstore(path)
    voxel_counts = {resolution: 0 for resolution in resolutions}
    try:
        stored_instances = [value.decode("ascii") for value in np.asarray(root["sha256"])]
        if stored_instances != instances:
            raise ValueError("Stored SHA order does not match the shard manifest")
        for index, instance in enumerate(instances):
            for resolution in resolutions:
                expected_coords, expected_features = load_payload(
                    find_payload(roots[resolution], instance)
                )
                coords, features = read_zarr_sample(root, resolution, index)
                if not np.array_equal(coords, expected_coords):
                    raise ValueError(f"Coordinate mismatch for {instance} at {resolution}")
                if not np.array_equal(features, expected_features):
                    raise ValueError(f"Feature mismatch for {instance} at {resolution}")
                voxel_counts[resolution] += len(coords)
    finally:
        store.close()
    return voxel_counts
