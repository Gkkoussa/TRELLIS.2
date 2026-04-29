#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path
from tqdm import tqdm


JSON_CHUNK_TYPE = 0x4E4F534A
BINARY_CHUNK_TYPE = 0x004E4942
TRIANGLES_MODE = 4


def primitive_face_count(primitive: dict, accessors: list[dict]) -> int:
    mode = primitive.get("mode", TRIANGLES_MODE)
    if mode != TRIANGLES_MODE:
        return 0

    if "indices" in primitive:
        return accessors[primitive["indices"]]["count"] // 3

    # Non-indexed primitive: infer triangle count from vertex count.
    position_accessor = primitive["attributes"]["POSITION"]
    return accessors[position_accessor]["count"] // 3


def has_fewer_faces_than(path: Path, max_faces: int) -> bool:
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError("Invalid GLB header")

        magic, version, length = struct.unpack("<III", header)
        if magic != 0x46546C67:
            raise ValueError("Not a GLB file")
        if version != 2:
            raise ValueError(f"Unsupported GLB version: {version}")

        json_chunk = None
        bytes_read = 12
        while bytes_read < length:
            chunk_header = f.read(8)
            if len(chunk_header) != 8:
                raise ValueError("Invalid GLB chunk header")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk_data = f.read(chunk_length)
            if len(chunk_data) != chunk_length:
                raise ValueError("Truncated GLB chunk")
            bytes_read += 8 + chunk_length

            if chunk_type == JSON_CHUNK_TYPE:
                json_chunk = chunk_data
            elif chunk_type == BINARY_CHUNK_TYPE:
                continue

        if json_chunk is None:
            raise ValueError("Missing JSON chunk")

    gltf = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\0"))
    accessors = gltf.get("accessors", [])
    total_faces = 0

    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            total_faces += primitive_face_count(primitive, accessors)
            if total_faces >= max_faces:
                return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root directory to search recursively")
    parser.add_argument("max_faces", type=int, help="Print paths with fewer than this many faces")
    args = parser.parse_args()

    glb_paths = list(args.root.rglob("*.glb"))

    for path in tqdm(glb_paths, total=len(glb_paths), desc="Scanning GLBs", unit="file"):
        try:
            if has_fewer_faces_than(path, args.max_faces):
                print(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")


if __name__ == "__main__":
    main()
