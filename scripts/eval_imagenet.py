#!/usr/bin/env python3
"""Evaluate a selected checkpoint on the official ImageNet validation set."""

from __future__ import annotations

import argparse
import csv
import os
import sys
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
    p.add_argument("--variant", required=True, choices=["rgb", "ll", "ll_lh", "ll_hl", "ll_hh", "all"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--config", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.json"))
    p.add_argument("--weights", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.pth"))
    p.add_argument("--mcunet-root", default=str(ROOT.parent / "mcunet"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["MCUNET_ROOT"] = args.mcunet_root
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    loader, classes, dataset = build_val(args.data_root, args.variant, args.batch_size, args.workers)
    model = load_model(args.config, args.weights, args.variant)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    output_path = Path(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = top1 = top5 = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "true_label", "true_name", "pred_top1", "pred_top5"])
        with torch.no_grad():
            for batch_index, (images, target) in enumerate(tqdm(loader, desc="ImageNet val")):
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
                    writer.writerow([str(Path(sample_path).relative_to(Path(args.data_root))), int(sample_target),
                                     classes[sample_target], top_ids[0], " ".join(map(str, top_ids))])
    print(f"top1={100 * top1 / total:.3f} top5={100 * top5 / total:.3f} n={total}")


if __name__ == "__main__":
    main()

