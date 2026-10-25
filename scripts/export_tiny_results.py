#!/usr/bin/env python3
"""Publish verified experiment artifacts with portable paths and static plots."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from summarize_tiny_results import VARIANTS, read_csv, summarize

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#333333", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#8B6508")
LABELS = ("RGB", "LL", "LL+LH", "LL+HL", "LL+HH", "All")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def figures(destination):
    directory = destination / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fields = (("train_loss", "Training cross-entropy", "Cross-entropy"),
              ("dev_loss", "Development cross-entropy", "Cross-entropy"),
              ("train_top1", "Training Top-1", "Accuracy (%)"),
              ("dev_top1", "Development Top-1", "Accuracy (%)"))
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        rows = read_csv(destination / variant / "history.csv")
        epochs = [int(row["epoch"]) + 1 for row in rows]
        for axis, (field, title, unit) in zip(axes.flat, fields):
            axis.plot(epochs, [float(row[field]) for row in rows], label=label, color=color,
                      linewidth=1.7, marker="o", markersize=2.8)
            axis.set(title=title, xlabel="Epoch (1-based)", ylabel=unit, xlim=(1, 20))
            axis.set_xticks((1, 5, 10, 15, 20))
            axis.grid(alpha=0.18)
    axes[0, 0].legend(ncol=3, fontsize=9)
    fig.suptitle("Tiny ImageNet / MCUNet-in2: 20 epochs, seed 2026", fontsize=14)
    fig.savefig(directory / "learning_curves.png", dpi=170)
    plt.close(fig)

    results = read_csv(destination / "results.csv")
    top1 = np.array([float(row["tiny_val_top1"]) for row in results])
    top5 = np.array([float(row["tiny_val_top5"]) for row in results])
    positions = np.arange(len(results))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": (1.3, 1)})
    axes[0].bar(positions - 0.2, top1, width=0.38, label="Top-1", color="#0072B2")
    axes[0].bar(positions + 0.2, top5, width=0.38, label="Top-5", color="#009E73")
    for offset, values in ((-0.2, top1), (0.2, top5)):
        for x, value in zip(positions + offset, values):
            axes[0].text(x, value + 1, f"{value:.2f}", ha="center", fontsize=8)
    axes[0].set(xticks=positions, xticklabels=LABELS, ylim=(0, 100), ylabel="Accuracy (%)",
                title="Validation accuracy: 10,000 images")
    axes[0].legend(loc="upper left", ncol=2)
    deltas = top1 - top1[0]
    axes[1].barh(positions, deltas, color=COLORS, height=0.6)
    for y, delta in zip(positions, deltas):
        axes[1].text(delta + (0.009 if delta >= 0 else -0.009), y, f"{delta:+.2f}",
                     va="center", ha="left" if delta >= 0 else "right", fontsize=10)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set(yticks=positions, yticklabels=LABELS, xlim=(-0.21, 0.23),
                xlabel="Top-1 difference vs RGB (percentage points)", title="Small single-seed differences")
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(axis="y" if axis is axes[0] else "x", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle("Tiny ImageNet 200-class preliminary results", fontsize=14)
    fig.savefig(directory / "validation_comparison.png", dpi=170)
    plt.close(fig)


def export(source, destination):
    if destination.exists():
        raise FileExistsError("Choose a new export directory to preserve the existing snapshot")
    summarize(source)
    destination.mkdir(parents=True)
    public = destination.relative_to(ROOT).as_posix()
    config = json.loads((source / "rgb/config.json").read_text())
    data_root = Path(config["data_root"]).parent
    manifest_path = Path(config["split_manifest"])
    upstream = Path(config["mcunet_root"])

    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            replacements = ((source, public), (manifest_path.parent, f"{public}/splits"),
                            (data_root, "../datasets/tiny-imagenet-200"), (upstream, "../mcunet"), (ROOT, "."))
            for local, replacement in replacements:
                if value == str(local) or value.startswith(str(local) + "\\") or value.startswith(str(local) + "/"):
                    return replacement + value[len(str(local)):].replace("\\", "/")
        return value

    for name in ("results.csv", "RESULTS.md"):
        shutil.copyfile(source / name, destination / name)
    split_dir = destination / "splits"
    split_dir.mkdir()
    write_json(split_dir / manifest_path.name, portable(json.loads(manifest_path.read_text())))
    shutil.copyfile(manifest_path.with_suffix(".samples.csv"), split_dir / manifest_path.with_suffix(".samples.csv").name)
    for variant in VARIANTS:
        output = destination / variant
        output.mkdir()
        for name in ("history.csv", "train.log", "curves.png", "val_predictions.csv", "class_mapping.json"):
            shutil.copyfile(source / variant / name, output / name)
        for name in ("config.json", "val_predictions.metrics.json"):
            write_json(output / name, portable(json.loads((source / variant / name).read_text())))
        for name in ("best.pt", "last.pt"):
            checkpoint = torch.load(source / variant / name, map_location="cpu", weights_only=False)
            checkpoint["args"] = portable(checkpoint["args"])
            torch.save(checkpoint, output / name)
    write_json(destination / "environment.json", {
        "recorded_training_versions": config["versions"], "gpu": config["device_name"],
        "recorded_settings": {key: config[key] for key in ("epochs", "batch_size", "workers", "lr", "weight_decay", "seed", "amp")},
        "export_versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "matplotlib", "tqdm")},
        "checkpoint_note": "Only args paths are rewritten; model/optimizer/scheduler/scaler/RNG states are retained. Use torch.load(weights_only=False) for trusted checkpoints.",
    })
    figures(destination)
    report = (destination / "RESULTS.md").read_text()
    report += """
## Visualizations

![Validation results](figures/validation_comparison.png)

![Learning curves](figures/learning_curves.png)

Training uses random crops and flips while development uses deterministic
center crops. Their accuracy/loss gap alone is not evidence of overfitting.
Development CE plateaus late and rises slightly in the final epoch; longer
training and repeated seeds have not been evaluated.

## Files

- `results.csv`: six-group summary; selected epochs are 1-based here.
- `<variant>/history.csv`, `train.log`, `curves.png`: all 20 epochs.
- `<variant>/best.pt`, `last.pt`: portable copies including resume state.
- `<variant>/val_predictions.csv`: all 10,000 image IDs, labels and Top-5 IDs/names.
- `<variant>/config.json`, `class_mapping.json`: settings and common 200-class mapping.
- `splits/`: exact shared split and all 100,000 training-pool image IDs.
- `environment.json`: recorded training versions and export environment versions.
- `SHA256SUMS.txt`: checksums for all exported artifacts except this checksum file.

The checkpoint/history epoch fields are 0-based (18 means the 19th epoch).
Published metadata paths are repository-relative; original local runs are
unchanged. No original images or source pretrained weights are included.

## Reproduce

Install a matching CUDA-enabled PyTorch/torchvision pair and requirements.txt.
Clone the upstream MCUNet repository beside this repository, place Tiny ImageNet
under `../datasets/tiny-imagenet-200`, and fetch the official pretrained weights.
Run from the repository root in PowerShell:

```powershell
python scripts/download_weights.py --out-dir weights
$groups = @('rgb', 'll', 'll_lh', 'll_hl', 'll_hh', 'all')
foreach ($g in $groups) {
    python scripts/train_imagenet.py --dataset tiny-imagenet --data-root ../datasets/tiny-imagenet-200/train --variant $g --run-dir runs/reproduce_tiny/$g --split-manifest results/tiny_imagenet/splits/tiny_train_dev_seed2026.json --mcunet-root ../mcunet --epochs 20 --batch-size 32 --workers 2 --seed 2026 --lr 0.001 --weight-decay 0.00004 --dev-fraction 0.05 --device cuda --amp
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $g" }
}
foreach ($g in $groups) {
    python scripts/eval_imagenet.py --dataset tiny-imagenet --data-root ../datasets/tiny-imagenet-200/val --variant $g --checkpoint runs/reproduce_tiny/$g/best.pt --predictions runs/reproduce_tiny/$g/val_predictions.csv --mcunet-root ../mcunet --batch-size 32 --workers 2 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $g" }
}
python scripts/summarize_tiny_results.py --run-root runs/reproduce_tiny
```

Run `python scripts/summarize_tiny_results.py --run-root results/tiny_imagenet`
to independently recheck this published snapshot and recompute metrics from
predictions. Checkpoints are about 5 MB each and are stored directly in Git.
"""
    (destination / "README.md").write_text(report, encoding="utf-8")
    checksums = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(destination).as_posix()}")
    (destination / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
    print(f"Exported {len(checksums)} artifacts to {public}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/tiny_imagenet")
    parser.add_argument("--destination", type=Path, default=ROOT / "results/tiny_imagenet")
    args = parser.parse_args()
    export(args.source.resolve(), args.destination.resolve())
