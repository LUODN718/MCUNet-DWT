#!/usr/bin/env python3
"""Evaluate a selected checkpoint on the official ImageNet validation set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_val
from src.model import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ImageNet validation directory")
    p.add_argument("--dataset", choices=["imagenet", "tiny-imagenet"], default="imagenet")
    p.add_argument("--variant", required=True, choices=["rgb", "ll", "ll_lh", "ll_hl", "ll_hh", "all"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--config", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.json"))
    p.add_argument("--weights", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.pth"))
    p.add_argument("--mcunet-root", default=str(ROOT.parent / "mcunet"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    os.environ["MCUNET_ROOT"] = args.mcunet_root
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the CUDA Python environment or --device cpu")
    device = torch.device(args.device)
    data_root = Path(args.data_root).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["args"]["variant"] != args.variant or checkpoint["args"].get("dataset", "imagenet") != args.dataset:
        raise ValueError("checkpoint variant/dataset mismatch")
    loader, classes, dataset = build_val(data_root, args.variant, args.batch_size, args.workers,
                                        checkpoint["args"].get("rgb_resolution", 160), args.dataset)
    if classes != checkpoint["classes"]:
        raise ValueError("validation class mapping does not match the checkpoint")
    model = load_model(args.config, args.weights, args.variant, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    output_path = Path(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = top1 = top5 = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_label", "true_name", "pred_top1", "pred_top5", "pred_top5_names"])
        with torch.no_grad():
            batches = loader if args.max_batches is None else islice(loader, args.max_batches)
            for batch_index, (images, target) in enumerate(tqdm(batches, desc="validation", disable=args.no_progress)):
                output = model(images.to(device, non_blocking=True))
                probs = output.softmax(1)
                values, indices = probs.topk(5, dim=1)
                target = target.to(device)
                top1 += indices[:, 0].eq(target).sum().item()
                top5 += indices.eq(target[:, None]).any(1).sum().item()
                total += target.size(0)
                offset = batch_index * loader.batch_size
                for j in range(target.size(0)):
                    sample_path, sample_target = dataset.samples[offset + j]
                    top_ids = indices[j].tolist()
                    writer.writerow([str(Path(sample_path).resolve().relative_to(data_root)), int(sample_target),
                                     classes[sample_target], top_ids[0], " ".join(map(str, top_ids)),
                                     " ".join(classes[k] for k in top_ids)])
    if not total:
        raise ValueError("empty validation loader")
    metrics = {"dataset": args.dataset, "variant": args.variant, "samples": total,
               "available_samples": len(dataset), "limited": total != len(dataset),
               "top1": 100 * top1 / total, "top5": 100 * top5 / total,
               "checkpoint": str(Path(args.checkpoint).resolve()), "selected_epoch": checkpoint["epoch"],
               "scope": "tiny-imagenet preliminary" if args.dataset == "tiny-imagenet" else "imagenet-1k"}
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"top1={100 * top1 / total:.3f} top5={100 * top5 / total:.3f} n={total}")


if __name__ == "__main__":
    main()

