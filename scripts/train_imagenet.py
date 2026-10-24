#!/usr/bin/env python3
"""Fine-tune one MCUNet-in2 ImageNet input group.

The same script is used for rgb, ll, ll_lh, ll_hl, ll_hh and all. A shared
split manifest makes the development set identical across all six runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_train_dev
from src.model import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ImageNet train directory")
    p.add_argument("--variant", choices=["rgb", "ll", "ll_lh", "ll_hl", "ll_hh", "all"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split-manifest", default=str(ROOT / "splits/imagenet_train_dev_seed2026.json"))
    p.add_argument("--config", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.json"))
    p.add_argument("--weights", default=str(ROOT / "weights/mcunet-256kb-1mb_imagenet.pth"))
    p.add_argument("--mcunet-root", default=str(ROOT.parent / "mcunet"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=4e-5)
    p.add_argument("--dev-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), output.shape[1])
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().item() * 100.0 / target.size(0) for k in topk]


def run_epoch(model, loader, optimizer, device, scaler, train: bool, amp: bool):
    model.train(train)
    total_loss = total_n = total_top1 = total_top5 = 0.0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for images, target in tqdm(loader, leave=False, desc="train" if train else "dev"):
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
            with autocast_ctx:
                output = model(images)
                loss = F.cross_entropy(output, target)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            n = target.size(0)
            top1, top5 = accuracy(output.float(), target)
            total_loss += loss.item() * n
            total_top1 += top1 * n / 100.0
            total_top5 += top5 * n / 100.0
            total_n += n
    return {
        "loss": total_loss / total_n,
        "top1": 100.0 * total_top1 / total_n,
        "top5": 100.0 * total_top5 / total_n,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    Path(args.split_manifest).parent.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["MCUNET_ROOT"] = args.mcunet_root

    train_loader, dev_loader, classes, manifest = build_train_dev(
        args.data_root, args.variant, args.split_manifest, args.dev_fraction, args.seed,
        batch_size=args.batch_size, workers=args.workers,
    )
    model = load_model(args.config, args.weights, args.variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda") if device.type == "cuda" else None
    start_epoch = 0
    best_dev_ce = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_dev_ce = checkpoint.get("best_dev_ce", best_dev_ce)

    run_config = vars(args) | {
        "device_resolved": str(device),
        "num_train": len(train_loader.dataset),
        "num_dev": len(dev_loader.dataset),
        "num_classes": len(classes),
        "classes": classes,
        "normalization": "RGB: (x-0.5)/0.5; LL: (LL/2-0.5)/0.5; signed high bands: band/0.5",
        "subband_order": "LL/RGB, LH/RGB, HL/RGB, HH/RGB",
        "input_rgb_resolution": 160,
        "input_model_resolution": 160 if args.variant == "rgb" else 80,
        "manifest": manifest,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    history_path = run_dir / "history.csv"
    new_history = not history_path.exists() or start_epoch == 0
    history_file = history_path.open("w" if new_history else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(history_file, fieldnames=["epoch", "lr", "train_loss", "train_top1", "train_top5", "dev_loss", "dev_top1", "dev_top5"])
    if new_history:
        writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        train_stats = run_epoch(model, train_loader, optimizer, device, scaler, True, args.amp)
        dev_stats = run_epoch(model, dev_loader, optimizer, device, scaler, False, args.amp)
        lr = optimizer.param_groups[0]["lr"]
        writer.writerow({"epoch": epoch, "lr": lr, "train_loss": train_stats["loss"], "train_top1": train_stats["top1"],
                         "train_top5": train_stats["top5"], "dev_loss": dev_stats["loss"], "dev_top1": dev_stats["top1"], "dev_top5": dev_stats["top5"]})
        history_file.flush()
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_dev_ce": best_dev_ce, "args": vars(args),
                 "dev": dev_stats, "classes": classes}
        torch.save(state, run_dir / "last.pt")
        if dev_stats["loss"] < best_dev_ce:
            best_dev_ce = dev_stats["loss"]
            state["best_dev_ce"] = best_dev_ce
            torch.save(state, run_dir / "best.pt")
        print(f"epoch={epoch:03d} train_loss={train_stats['loss']:.4f} train_top1={train_stats['top1']:.2f} "
              f"dev_loss={dev_stats['loss']:.4f} dev_top1={dev_stats['top1']:.2f} dev_top5={dev_stats['top5']:.2f}")
        scheduler.step()
    history_file.close()


if __name__ == "__main__":
    main()

