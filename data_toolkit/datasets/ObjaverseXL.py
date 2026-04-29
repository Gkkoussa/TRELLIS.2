import os
import argparse
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
import zipfile

JSON_CHUNK_TYPE = 0x4E4F534A
BINARY_CHUNK_TYPE = 0x004E4942
TRIANGLES_MODE = 4



def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of objaverse download worker processes to use')
    parser.add_argument('--max_num_faces', type=int, default=None,
                        help='Keep only meshes with fewer than this many triangle faces')
    parser.add_argument('--delete_oversized', action='store_true',
                        help='Delete downloaded files that exceed --max_num_faces')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Stop after accepting this many downloaded files')
    parser.add_argument('--download_batch_size', type=int, default=256,
                        help='Number of candidate assets to download per batch when filtering')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def _primitive_face_count(primitive: dict, accessors: list[dict]) -> int:
    mode = primitive.get("mode", TRIANGLES_MODE)
    if mode != TRIANGLES_MODE:
        return 0
    if "indices" in primitive:
        return accessors[primitive["indices"]]["count"] // 3
    position_accessor = primitive["attributes"]["POSITION"]
    return accessors[position_accessor]["count"] // 3


def _has_fewer_faces_than(path: str, max_faces: int) -> bool:
    with open(path, "rb") as f:
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
            total_faces += _primitive_face_count(primitive, accessors)
            if total_faces >= max_faces:
                return False
    return True


def _delete_downloaded_path(path: str):
    if os.path.isfile(path):
        os.remove(path)
        return
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path)


def download(
    metadata,
    output_dir,
    processes=1,
    max_num_faces=None,
    delete_oversized=False,
    max_files=None,
    download_batch_size=256,
    **kwargs,
):
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]

    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    annotations = annotations.set_index("fileIdentifier")

    if max_files is not None and max_num_faces is None:
        metadata = metadata.iloc[:max_files]

    file_identifiers = metadata.index.tolist()

    for start in range(0, len(file_identifiers), download_batch_size):
        if max_files is not None and len(downloaded) >= max_files:
            break

        batch_ids = file_identifiers[start:start + download_batch_size]
        batch_annotations = annotations.loc[annotations.index.intersection(batch_ids)].reset_index()
        if len(batch_annotations) == 0:
            continue

        file_paths = oxl.download_objects(
            batch_annotations,
            download_dir=os.path.join(output_dir, "raw"),
            processes=processes,
            save_repo_format="zip",
        )

        batch_kept = 0
        batch_deleted = 0
        batch_rejected = 0
        for file_identifier, local_path in file_paths.items():
            sha256 = metadata.loc[file_identifier, "sha256"]
            keep = True
            if max_num_faces is not None:
                if local_path.endswith(".glb"):
                    try:
                        keep = _has_fewer_faces_than(local_path, max_num_faces)
                    except Exception as e:
                        print(f"Error checking face count for {local_path}: {e}")
                        keep = False
                else:
                    keep = False

                if not keep and delete_oversized:
                    try:
                        _delete_downloaded_path(local_path)
                        batch_deleted += 1
                    except Exception as e:
                        print(f"Error deleting oversized file {local_path}: {e}")

            if keep:
                downloaded[sha256] = os.path.relpath(local_path, output_dir)
                batch_kept += 1
                if max_files is not None and len(downloaded) >= max_files:
                    break
            else:
                batch_rejected += 1

        print(
            f"Batch {start // download_batch_size + 1}: "
            f"requested={len(batch_annotations)}, "
            f"downloaded={len(file_paths)}, "
            f"kept={batch_kept}, "
            f"rejected={batch_rejected}, "
            f"deleted={batch_deleted}, "
            f"accepted_total={len(downloaded)}"
        )

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', no_file=False):
    records = []
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            
            def worker(metadatum):
                try:
                    sha256 = metadatum['sha256']
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum['local_path']
                        if local_path.startswith('raw/github/repos/'):
                            path_parts = local_path.split('/')
                            file_name = os.path.join(*path_parts[5:])
                            zip_file = os.path.join(output_dir, *path_parts[:5])
                            import tempfile, zipfile
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                    zip_ref.extractall(tmp_dir)
                                file = os.path.join(tmp_dir, file_name)
                                record = func(file, metadatum) 
                        else:
                            file = os.path.join(output_dir, local_path)
                            record = func(file, metadatum) 

                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {metadatum.get('sha256', 'unknown')}: {e}")
                    pbar.update()
            
            for metadatum in metadata.to_dict('records'):
                executor.submit(worker, metadatum)
            
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
        
    return pd.DataFrame.from_records(records)
