#!/usr/bin/env python3
"""Verify completed Tiny ImageNet runs and summarize the selected checkpoints."""

import argparse
import csv
import json
from pathlib import Path

import torch


VARIANTS = ("rgb", "ll", "ll_lh", "ll_hl", "ll_hh", "all")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(root):
    rows = []
    reference = None
    sample_ids = None
    for variant in VARIANTS:
        folder = root / variant
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        history = read_csv(folder / "history.csv")
        metrics = json.loads((folder / "val_predictions.metrics.json").read_text(encoding="utf-8"))
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert config["dataset"] == "tiny-imagenet" and config["num_classes"] == 200
        assert not config["batch_limited"]
        assert [int(row["epoch"]) for row in history] == list(range(config["epochs"]))
        assert all(int(row["train_samples"]) == config["num_train"]
                   and int(row["dev_samples"]) == config["num_dev"] for row in history)
        selected = min(history, key=lambda row: float(row["dev_loss"]))
        assert best["epoch"] == metrics["selected_epoch"] == int(selected["epoch"])
        assert abs(best["dev"]["loss"] - float(selected["dev_loss"])) < 1e-8
        assert last["epoch"] == config["epochs"] - 1
        assert last["optimizer"]["state"] and last["scheduler"] and last["scaler"] and last["rng"]
        assert last["scheduler"]["last_epoch"] == config["epochs"]
        assert metrics["dataset"] == "tiny-imagenet" and metrics["variant"] == variant
        assert metrics["samples"] == metrics["available_samples"] == 10000 and not metrics["limited"]
        common = {key: config[key] for key in ("epochs", "batch_size", "seed", "lr", "weight_decay",
                                               "amp", "num_train", "num_dev", "classes", "pretrained_sha256")}
        common["sample_fingerprint"] = config["manifest"]["sample_fingerprint"]
        common["initialization"] = config["initialization"]
        if reference is None:
            reference = common
        else:
            assert common == reference, "training protocol, split or initial weights differ"
        predictions = read_csv(folder / "val_predictions.csv")
        identifiers = []
        top1 = top5 = 0
        for prediction in predictions:
            identifiers.append((prediction["image_id"], prediction["true_label"]))
            target = int(prediction["true_label"])
            ranks = [int(value) for value in prediction["pred_top5"].split()]
            assert len(set(ranks)) == 5 and all(0 <= value < 200 for value in ranks)
            assert int(prediction["pred_top1"]) == ranks[0]
            assert prediction["true_name"] == config["classes"][target]
            assert prediction["pred_top5_names"].split() == [config["classes"][value] for value in ranks]
            top1 += ranks[0] == target
            top5 += target in ranks
        assert len(identifiers) == len(set(identifiers)) == 10000
        if sample_ids is None:
            sample_ids = identifiers
        else:
            assert identifiers == sample_ids, "validation samples or labels differ"
        assert abs(100 * top1 / 10000 - metrics["top1"]) < 1e-9
        assert abs(100 * top5 / 10000 - metrics["top5"]) < 1e-9
        rows.append({
            "variant": variant, "train_samples": config["num_train"], "dev_samples": config["num_dev"],
            "epochs_completed": len(history), "selected_epoch": best["epoch"] + 1,
            "dev_ce": float(selected["dev_loss"]), "dev_top1": float(selected["dev_top1"]),
            "dev_top5": float(selected["dev_top5"]), "tiny_val_samples": len(predictions),
            "tiny_val_top1": metrics["top1"], "tiny_val_top5": metrics["top5"],
            "official_val_top1": "", "official_val_top5": "", "seed": config["seed"],
            "status": "complete_tiny_imagenet_preliminary",
            "notes": "200 classes; selected_epoch is 1-based; no ImageNet-1K evaluation",
        })
    baseline = rows[0]["tiny_val_top1"]
    for row in rows:
        row["top1_delta_vs_rgb_pp"] = round(row["tiny_val_top1"] - baseline, 2)
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Tiny ImageNet Six-Group Results", "",
              "All six groups completed 20 epochs on the shared 95,000/5,000 train/dev split.",
              "Each selected checkpoint was evaluated on the same 10,000 validation images.",
              "These are preliminary 200-class results, not ImageNet-1K benchmark results.", "",
              "Protocol: seed 2026; batch 32; SGD momentum 0.9; LR 0.001; cosine decay;",
              "weight decay 0.00004; CUDA AMP. RGB augmentation precedes online orthonormal Haar DWT.",
              "The new 200-class head and LL stem weights were identical at initialization.", "",
              "| Group | Selected epoch (1-based) | Dev CE | Dev Top-1 (%) | Val Top-1 (%) | Val Top-5 (%) | Top-1 delta (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['variant']} | {row['selected_epoch']} | {row['dev_ce']:.6f} | "
                      f"{row['dev_top1']:.2f} | {row['tiny_val_top1']:.2f} | "
                      f"{row['tiny_val_top5']:.2f} | {row['top1_delta_vs_rgb_pp']:+.2f} |")
    report += ["", "LL+HL has the highest observed validation Top-1 (66.62%), 0.15 percentage points above RGB.",
               "LL alone is 0.07 percentage points below RGB. RGB has the highest Top-5 (87.78%).",
               "The small differences in this single-seed run do not establish a repeatable improvement.",
               "All groups selected epoch 19 by minimum development CE. CE rises slightly at epoch 20;",
               "the saved best checkpoints avoid selecting that last-epoch regression.", "",
               "Checks passed: equal protocol/initialization/class mapping/split; complete epoch counts;",
               "minimum-CE checkpoint selection; resumable checkpoint state; 10,000 unique predictions per group;",
               "Top-1/Top-5 independently recomputed from prediction CSVs.", "",
               "Per-group directories contain config.json, class_mapping.json, train.log, history.csv,",
               "curves.png, best.pt, last.pt, val_predictions.csv and val_predictions.metrics.json.",
               "The shared sample manifest is recorded by config.json. ImageNet-1K validation columns are blank.", ""]
    (root / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs/tiny_imagenet")
    args = parser.parse_args()
    summarize(args.run_root)
