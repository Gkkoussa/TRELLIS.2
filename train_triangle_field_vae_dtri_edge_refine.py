"""Launcher registration for the selectable R512 d_tri edge architectures."""

import argparse
import json
import os
import sys

from easydict import EasyDict as edict
import torch
import torch.multiprocessing as mp

import train as standard_train
from trellis2 import models, trainers
from trellis2.models.sc_vaes.sparse_unet_vae_hierarchical_vertex_edge_dtri_refine import (
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder,
)
from trellis2.models.sc_vaes.sparse_unet_vae_hierarchical_vertex_edge_dtri_context import (
    SparseUnetVaeHierarchicalVertexEdgeDtriContextDecoder,
    SparseUnetVaeHierarchicalVertexEdgeDtriContextNoVaeFusionDecoder,
)
from trellis2.trainers.vae.triangle_field_vae_dtri_edge_refine import (
    TriangleFieldVaeDtriEdgeRefineTrainer,
)


# Register only in this entry point.  Existing TRELLIS.2 registries and normal
# train.py runs remain unchanged.  This top-level registration is repeated in
# every torch.multiprocessing spawned worker.
setattr(
    models,
    "SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder",
    SparseUnetVaeHierarchicalVertexEdgeDtriRefineDecoder,
)
setattr(
    models,
    "SparseUnetVaeHierarchicalVertexEdgeDtriContextDecoder",
    SparseUnetVaeHierarchicalVertexEdgeDtriContextDecoder,
)
setattr(
    models,
    "SparseUnetVaeHierarchicalVertexEdgeDtriContextNoVaeFusionDecoder",
    SparseUnetVaeHierarchicalVertexEdgeDtriContextNoVaeFusionDecoder,
)
setattr(
    trainers,
    "TriangleFieldVaeDtriEdgeRefineTrainer",
    TriangleFieldVaeDtriEdgeRefineTrainer,
)


def main(local_rank, cfg):
    return standard_train.main(local_rank, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--auto_retry", type=int, default=3)
    parser.add_argument("--tryrun", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=-1)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12345")
    opt = parser.parse_args()

    opt.load_dir = opt.load_dir if opt.load_dir else opt.output_dir
    opt.num_gpus = (
        torch.cuda.device_count() if opt.num_gpus == -1 else opt.num_gpus
    )
    with open(opt.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    cfg = edict()
    cfg.update(opt.__dict__)
    cfg.update(config)
    print("\n\nConfig:")
    print("=" * 80)
    print(json.dumps(cfg.__dict__, indent=4))

    if cfg.node_rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(
            os.path.join(cfg.output_dir, "command.txt"), "w", encoding="utf-8"
        ) as handle:
            print(" ".join(["python"] + sys.argv), file=handle)
        with open(
            os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, indent=4)

    attempts = 1 if cfg.auto_retry == 0 else cfg.auto_retry
    for retry_index in range(attempts):
        try:
            cfg = standard_train.find_ckpt(cfg)
            if cfg.num_gpus > 1:
                mp.spawn(main, args=(cfg,), nprocs=cfg.num_gpus, join=True)
            else:
                main(0, cfg)
            break
        except Exception as error:
            if cfg.auto_retry == 0:
                raise
            print(f"Error: {error}")
            print(f"Retrying ({retry_index + 1}/{cfg.auto_retry})...")
