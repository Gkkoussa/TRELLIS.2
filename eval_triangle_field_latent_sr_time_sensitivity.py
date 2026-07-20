import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trellis2.utils.data_utils import recursive_to_device

from eval_triangle_field_latent_sr_flow import (
    build_dataset,
    build_support_latents,
    build_trainer,
    find_ckpt_step,
    load_config,
    load_encoder_checkpoint,
    predict_z0,
    slice_batch,
)
from eval_metadata_filters import add_eval_metadata_filter_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure latent SR model sensitivity to t with fixed z_t and conditioning."
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--ema_rate", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--low_resolution", type=int, default=64)
    parser.add_argument("--high_resolution", type=int, default=128)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition_mode", choices=("zero", "cond"), default="zero")
    parser.add_argument(
        "--t_values",
        type=str,
        default="1.0,0.8,0.6,0.5,0.4,0.2,0.1,0.05",
        help="Comma-separated t values to test.",
    )
    add_eval_metadata_filter_args(parser)
    return parser.parse_args()


def l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir).resolve()
    root = Path(args.root).resolve()
    cfg = load_config(run_dir)
    ckpt_step = find_ckpt_step(run_dir, args.ckpt)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else run_dir
        / (
            f"eval_filtered_{args.split}_{args.low_resolution}to{args.high_resolution}"
            f"_time_sensitivity_step{ckpt_step:07d}_{args.condition_mode}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_args = argparse.Namespace(
        low_resolution=args.low_resolution,
        high_resolution=args.high_resolution,
        no_latents=True,
        latent_name=None,
        split=args.split,
        render_resolution=None,
        metadata_filter_csv=args.metadata_filter_csv,
        no_train_duplicate_csv=args.no_train_duplicate_csv,
        triangle_filter_csv=args.triangle_filter_csv,
        disable_default_eval_filters=args.disable_default_eval_filters,
    )
    dataset, data_dir = build_dataset(cfg, root, flow_args)
    trainer = build_trainer(cfg, dataset, output_dir)
    ckpt_path = load_encoder_checkpoint(trainer, run_dir, ckpt_step, args.ema_rate)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    iterator = iter(loader)
    data = None
    for _ in range(max(0, args.sample_index) + 1):
        data = next(iterator)
    data = recursive_to_device(slice_batch(data, 1), trainer.device)
    latent_channels = int(cfg["models"]["encoder"]["args"]["latent_channels"])
    z_support, caches = build_support_latents(data["x_0"], latent_channels)
    z_t = z_support.replace(torch.randn_like(z_support.feats))
    cond = data["cond"]
    if args.condition_mode == "zero":
        cond = cond.replace(torch.zeros_like(cond.feats))

    t_values = [float(v.strip()) for v in args.t_values.split(",") if v.strip()]
    preds = []
    decoded = []
    for t in t_values:
        pred = predict_z0(trainer, z_t, cond, caches, cache_paths=None, t=t)
        y = trainer._decode_latents_with_cache(pred, caches=caches)
        preds.append(pred)
        decoded.append(y)

    ref_pred = preds[0]
    ref_decoded = decoded[0]
    rows = []
    for i, (t, pred, y) in enumerate(zip(t_values, preds, decoded)):
        prev_pred = preds[i - 1] if i > 0 else None
        prev_decoded = decoded[i - 1] if i > 0 else None
        rows.append({
            "t": t,
            "pred_z0_abs_mean": float(pred.feats.float().abs().mean().item()),
            "decoded_d_tri_abs_mean": float(y.feats[:, :1].float().abs().mean().item()),
            "pred_z0_l1_vs_t0": l1(pred.feats, ref_pred.feats),
            "decoded_d_tri_l1_vs_t0": l1(y.feats[:, :1], ref_decoded.feats[:, :1]),
            "pred_z0_l1_vs_prev": None if prev_pred is None else l1(pred.feats, prev_pred.feats),
            "decoded_d_tri_l1_vs_prev": None if prev_decoded is None else l1(y.feats[:, :1], prev_decoded.feats[:, :1]),
        })

    csv_path = output_dir / "time_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_step": ckpt_step,
        "checkpoint_path": ckpt_path,
        "root": str(root),
        "split": args.split,
        "data_dir": data_dir,
        "condition_mode": args.condition_mode,
        "low_resolution": args.low_resolution,
        "high_resolution": args.high_resolution,
        "t_values": t_values,
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "rows": rows,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
