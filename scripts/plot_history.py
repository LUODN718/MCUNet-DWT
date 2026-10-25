#!/usr/bin/env python3
"""Render recorded train/development metrics without requiring a display."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history_path, output_path, title="Training history"):
    with Path(history_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("history is empty")
    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for split, color in (("train", "#2166ac"), ("dev", "#b2182b")):
        axes[0].plot(epochs, [float(row[f"{split}_loss"]) for row in rows],
                     marker="o", color=color, label=split)
        axes[1].plot(epochs, [float(row[f"{split}_top1"]) for row in rows],
                     marker="o", color=color, label=split)
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Pass (1-based)")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Training history")
    args = parser.parse_args()
    plot_history(args.history, args.output, args.title)
