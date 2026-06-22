import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


def _is_top_level_obj_path(value):
    if not value:
        return False
    path = Path(value)
    return not path.is_absolute() and len(path.parts) == 1 and path.suffix.lower() == ".obj"


def _discover_metadata_paths(root, explicit_paths):
    if explicit_paths:
        return [Path(path) for path in explicit_paths]

    paths = [
        root / "metadata.csv",
        root / "metadata_no_triangle_dense.csv",
    ]
    split_root = root / "splits"
    if split_root.exists():
        paths.extend(sorted(split_root.glob("*/metadata.csv")))
    return [path for path in paths if path.exists()]


def _collect_mesh_names(metadata_paths, root):
    names = set()
    for path in metadata_paths:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                local_path = row.get("local_path")
                if _is_top_level_obj_path(local_path):
                    names.add(local_path)

                mesh_path = row.get("triangle_area_mesh_path")
                if mesh_path:
                    mesh_path_obj = Path(mesh_path)
                    if (
                        mesh_path_obj.is_absolute()
                        and mesh_path_obj.parent == root
                        and mesh_path_obj.suffix.lower() == ".obj"
                    ):
                        names.add(mesh_path_obj.name)
    return names


def _rewrite_metadata(path, root, mesh_dir, valid_mesh_names, apply, backup):
    local_path_updates = 0
    triangle_area_mesh_path_updates = 0

    tmp_path = path.with_name(f"{path.name}.tmp")
    writer_file = None
    writer = None

    try:
        with path.open("r", newline="") as reader_file:
            reader = csv.DictReader(reader_file)
            fieldnames = reader.fieldnames
            if not fieldnames:
                return {"path": path, "local_path_updates": 0, "triangle_area_mesh_path_updates": 0}

            if apply:
                writer_file = tmp_path.open("w", newline="")
                writer = csv.DictWriter(writer_file, fieldnames=fieldnames)
                writer.writeheader()

            for row in reader:
                local_path = row.get("local_path")
                if _is_top_level_obj_path(local_path):
                    if local_path in valid_mesh_names:
                        row["local_path"] = str(Path(mesh_dir.name) / local_path)
                        local_path_updates += 1

                mesh_path = row.get("triangle_area_mesh_path")
                if mesh_path:
                    mesh_path_obj = Path(mesh_path)
                    if (
                        mesh_path_obj.is_absolute()
                        and mesh_path_obj.parent == root
                        and mesh_path_obj.suffix.lower() == ".obj"
                        and mesh_path_obj.name in valid_mesh_names
                    ):
                        row["triangle_area_mesh_path"] = str(mesh_dir / mesh_path_obj.name)
                        triangle_area_mesh_path_updates += 1

                if writer is not None:
                    writer.writerow(row)
    finally:
        if writer_file is not None:
            writer_file.close()

    if apply:
        if local_path_updates or triangle_area_mesh_path_updates:
            if backup:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
                shutil.copy2(path, backup_path)
            tmp_path.replace(path)
        elif tmp_path.exists():
            tmp_path.unlink()

    return {
        "path": path,
        "local_path_updates": local_path_updates,
        "triangle_area_mesh_path_updates": triangle_area_mesh_path_updates,
    }


def _plan_moves(root, mesh_dir, mesh_names):
    moves = []
    conflicts = []
    already_moved = []
    missing = []
    for name in sorted(mesh_names):
        src = root / name
        dst = mesh_dir / name
        src_exists = src.exists()
        dst_exists = dst.exists()
        if src_exists and dst_exists:
            conflicts.append((src, dst))
        elif src_exists:
            moves.append((src, dst))
        elif dst_exists:
            already_moved.append(dst)
        else:
            missing.append(name)
    return moves, conflicts, already_moved, missing


def main():
    parser = argparse.ArgumentParser(
        description="Move ObjXL root-level .obj files into a meshes/ directory and update metadata paths."
    )
    parser.add_argument("--root", required=True, help="ObjXL root directory.")
    parser.add_argument("--mesh_dir_name", default="meshes", help="Directory name under root for moved meshes.")
    parser.add_argument("--metadata", action="append", default=None, help="Metadata CSV to update. May be repeated.")
    parser.add_argument("--apply", action="store_true", help="Actually move files and rewrite metadata.")
    parser.add_argument("--no_backup", action="store_true", help="Do not create timestamped CSV backups when applying.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    mesh_dir = root / args.mesh_dir_name
    metadata_paths = _discover_metadata_paths(root, args.metadata)
    mesh_names = _collect_mesh_names(metadata_paths, root)
    moves, conflicts, already_moved, missing = _plan_moves(root, mesh_dir, mesh_names)
    valid_mesh_names = {src.name for src, _ in moves}
    valid_mesh_names.update(path.name for path in already_moved)

    print(f"Root: {root}")
    print(f"Mesh directory: {mesh_dir}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Metadata-referenced top-level .obj names: {len(mesh_names)}")
    print(f"Root-level .obj files to move: {len(moves)}")
    print(f"Already under {mesh_dir.name}/: {len(already_moved)}")
    print(f"Missing metadata-referenced .obj files: {len(missing)}")
    print(f"Destination conflicts: {len(conflicts)}")
    print(f"Metadata CSVs to inspect: {len(metadata_paths)}")

    if conflicts:
        print("\nConflicts:")
        for src, dst in conflicts[:20]:
            print(f"  {src} -> {dst}")
        if len(conflicts) > 20:
            print(f"  ... {len(conflicts) - 20} more")
        raise SystemExit("Refusing to continue because destination files already exist.")

    if missing:
        print("\nMissing examples:")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    metadata_results = []
    for path in metadata_paths:
        metadata_results.append(
            _rewrite_metadata(
                path=path,
                root=root,
                mesh_dir=mesh_dir,
                valid_mesh_names=valid_mesh_names,
                apply=args.apply,
                backup=not args.no_backup,
            )
        )

    print("\nMetadata updates:")
    for result in metadata_results:
        print(
            f"  {result['path']}: "
            f"local_path={result['local_path_updates']}, "
            f"triangle_area_mesh_path={result['triangle_area_mesh_path_updates']}"
        )

    if args.apply:
        mesh_dir.mkdir(exist_ok=True)
        for src, dst in moves:
            shutil.move(str(src), str(dst))
        print(f"\nMoved {len(moves)} .obj files.")
    else:
        print("\nDry run only. Re-run with --apply to move files and rewrite metadata.")


if __name__ == "__main__":
    main()
