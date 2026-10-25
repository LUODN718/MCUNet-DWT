#!/usr/bin/env python3
"""Fine-tune one MCUNet-in2 ImageNet input group.

The same script is used for rgb, ll, ll_lh, ll_hl, ll_hh and all. A shared
split manifest makes the development set identical across all six runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import sys
from itertools import islice
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
    p.add_argument("--dataset", choices=["imagenet", "tiny-imagenet"], default="imagenet")
    p.add_argument("--variant", choices=["rgb", "ll", "ll_lh", "ll_hl", "ll_hh", "all"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split-manifest")
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
    p.add_argument("--rgb-resolution", type=int, default=160)
    p.add_argument("--max-train-batches", type=int)
    p.add_argument("--max-dev-batches", type=int)
    p.add_argument("--stop-after-epochs", type=int, help="Stop cleanly without changing the scheduler budget")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    for name in ("epochs", "batch_size", "rgb_resolution", "max_train_batches", "max_dev_batches", "stop_after_epochs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"{name} must be positive")
    if args.workers < 0 or args.rgb_resolution % 2:
        p.error("workers must be nonnegative and rgb-resolution must be even")
    if args.split_manifest is None:
        prefix = "tiny" if args.dataset == "tiny-imagenet" else "imagenet"
        args.split_manifest = str(ROOT / f"splits/{prefix}_train_dev_seed{args.seed}.json")
    return args


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


def run_epoch(model, loader, optimizer, device, scaler, train: bool, amp: bool,
              max_batches=None, progress=True):
    model.train(train)
    total_loss = total_n = total_top1 = total_top5 = 0.0
    context = torch.enable_grad if train else torch.no_grad
    with context():
        batches = loader if max_batches is None else islice(loader, max_batches)
        total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
        for images, target in tqdm(batches, total=total_batches, disable=not progress,
                                   leave=False, desc="train" if train else "dev"):
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
            with autocast_ctx:
                output = model(images)
                loss = F.cross_entropy(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
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
    if not total_n:
        raise ValueError("empty data loader")
    return {
        "loss": total_loss / total_n,
        "top1": 100.0 * total_top1 / total_n,
        "top5": 100.0 * total_top5 / total_n,
        "samples": int(total_n),
    }


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state(train_loader, dev_loader):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_loader": train_loader.generator.get_state(),
            "dev_loader": dev_loader.generator.get_state()}


def restore_rng(state, train_loader, dev_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    train_loader.generator.set_state(state["train_loader"])
    dev_loader.generator.set_state(state["dev_loader"])


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use a CUDA build or --device cpu")
    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    if (run_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError("run already exists; use --resume or a new --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run_dir / "train.log", mode="a" if args.resume else "w", encoding="utf-8")])
    Path(args.split_manifest).parent.mkdir(parents=True, exist_ok=True)
    os.environ["MCUNET_ROOT"] = args.mcunet_root

    train_loader, dev_loader, classes, manifest = build_train_dev(
        args.data_root, args.variant, args.split_manifest, args.dev_fraction, args.seed,
        rgb_resolution=args.rgb_resolution, batch_size=args.batch_size, workers=args.workers,
        dataset_kind=args.dataset,
    )
    model = load_model(args.config, args.weights, args.variant, len(classes), args.seed).to(device)
    stem = model.first_conv.conv.weight
    initialization = {"ll_or_rgb_sha256": tensor_hash(stem[:, :3]),
                      "extra_nonzero": int(torch.count_nonzero(stem[:, 3:]).item()),
                      "classifier_sha256": tensor_hash(model.classifier.linear.weight),
                      "classifier_reinitialized": len(classes) != 1000}
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = None
    if device.type == "cuda":
        scaler_type = getattr(torch.amp, "GradScaler", None)
        scaler = scaler_type("cuda", enabled=args.amp) if scaler_type else torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 0
    best_dev_ce = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["classes"] != classes:
            raise ValueError("checkpoint class mapping differs from the dataset")
        for key in ("variant", "dataset", "seed", "epochs", "batch_size", "workers", "rgb_resolution",
                    "lr", "weight_decay", "dev_fraction", "amp", "max_train_batches", "max_dev_batches"):
            if checkpoint["args"].get(key) != getattr(args, key):
                raise ValueError(f"resume configuration mismatch: {key}")
        if checkpoint.get("sample_fingerprint") != manifest["sample_fingerprint"]:
            raise ValueError("resume dataset fingerprint mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_dev_ce = checkpoint.get("best_dev_ce", best_dev_ce)
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], train_loader, dev_loader)
        logging.info("Resumed epoch=%d next_lr=%.8g", start_epoch, optimizer.param_groups[0]["lr"])
    else:
        # Stem construction consumes different numbers of random values by variant.
        seed_everything(args.seed)

    run_config = vars(args) | {
        "device_resolved": str(device),
        "num_train": len(train_loader.dataset),
        "num_dev": len(dev_loader.dataset),
        "num_classes": len(classes),
        "classes": classes,
        "normalization": "RGB: (x-0.5)/0.5; LL: (LL/2-0.5)/0.5; signed high bands: band/0.5",
        "subband_order": "LL/RGB, LH/RGB, HL/RGB, HH/RGB",
        "input_rgb_resolution": args.rgb_resolution,
        "input_model_resolution": args.rgb_resolution if args.variant == "rgb" else args.rgb_resolution // 2,
        "manifest": {key: manifest[key] for key in ("seed", "dev_fraction", "num_samples", "sample_fingerprint", "sample_list")},
        "first_conv": model.first_conv.config,
        "initialization": initialization,
        "training_scope": "all parameters and batch normalization",
        "result_scope": "tiny-imagenet preliminary" if args.dataset == "tiny-imagenet" else "imagenet-1k",
        "batch_limited": args.max_train_batches is not None or args.max_dev_batches is not None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "torchvision": __import__("torchvision").__version__, "Pillow": __import__("PIL").__version__},
        "pretrained_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "pretrained_source": "https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-256kb-1mb_imagenet.pth",
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "class_mapping.json").write_text(json.dumps(dict(enumerate(classes)), indent=2), encoding="utf-8")
    logging.info("dataset=%s classes=%d train=%d dev=%d device=%s input=%d", args.dataset, len(classes),
                 len(train_loader.dataset), len(dev_loader.dataset), run_config["device_name"], run_config["input_model_resolution"])
    if run_config["batch_limited"]:
        logging.info("PIPELINE TEST ONLY: truncated train/dev passes; metrics are not benchmark results")
    history_path = run_dir / "history.csv"
    new_history = not history_path.exists() or start_epoch == 0
    history_file = history_path.open("w" if new_history else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(history_file, fieldnames=["epoch", "lr", "train_samples", "dev_samples", "train_loss", "train_top1", "train_top5", "dev_loss", "dev_top1", "dev_top5"])
    if new_history:
        writer.writeheader()

    stop_epoch = args.epochs if args.stop_after_epochs is None else min(args.epochs, start_epoch + args.stop_after_epochs)
    for epoch in range(start_epoch, stop_epoch):
        train_stats = run_epoch(model, train_loader, optimizer, device, scaler, True, args.amp,
                                args.max_train_batches, not args.no_progress)
        dev_stats = run_epoch(model, dev_loader, optimizer, device, scaler, False, args.amp,
                              args.max_dev_batches, not args.no_progress)
        lr = optimizer.param_groups[0]["lr"]
        writer.writerow({"epoch": epoch, "lr": lr, "train_samples": train_stats["samples"], "dev_samples": dev_stats["samples"],
                         "train_loss": train_stats["loss"], "train_top1": train_stats["top1"],
                         "train_top5": train_stats["top5"], "dev_loss": dev_stats["loss"], "dev_top1": dev_stats["top1"], "dev_top5": dev_stats["top5"]})
        history_file.flush()
        is_best = dev_stats["loss"] < best_dev_ce
        if is_best:
            best_dev_ce = dev_stats["loss"]
        # Save the state for the next epoch, including the updated scheduler/best CE.
        scheduler.step()
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler is not None else None,
                 "best_dev_ce": best_dev_ce, "args": vars(args), "rng": rng_state(train_loader, dev_loader),
                 "dev": dev_stats, "train": train_stats, "classes": classes,
                 "sample_fingerprint": manifest["sample_fingerprint"]}
        torch.save(state, run_dir / "last.pt")
        if is_best:
            torch.save(state, run_dir / "best.pt")
        logging.info("epoch=%03d lr=%.8g train_n=%d dev_n=%d train_loss=%.4f train_top1=%.2f dev_loss=%.4f dev_top1=%.2f dev_top5=%.2f",
                     epoch, lr, train_stats["samples"], dev_stats["samples"], train_stats["loss"], train_stats["top1"],
                     dev_stats["loss"], dev_stats["top1"], dev_stats["top5"])
    history_file.close()
    from plot_history import plot_history
    scope = "truncated passes" if run_config["batch_limited"] else "training history"
    plot_history(history_path, run_dir / "curves.png", f"{args.dataset}/{args.variant}: {scope}")


if __name__ == "__main__":
    main()

