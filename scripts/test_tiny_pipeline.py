#!/usr/bin/env python3
"""Exercise all six variants, real updates, resume, evaluation and artifacts.

Only a few batches are processed. The output is a pipeline test, not an accuracy
benchmark or a completed epoch on Tiny ImageNet.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_history import plot_history
from src.wavelet import VARIANT_SUBBANDS, input_channels


def execute(command):
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT.parent / "datasets/tiny-imagenet-200"))
    parser.add_argument("--run-dir", default=str(ROOT / "runs/tiny_pipeline_test"))
    parser.add_argument("--mcunet-root", default=str(Path.home() / "Desktop/Mcunet/mcunet"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_root = Path(args.run_dir).resolve()
    if run_root.exists():
        raise FileExistsError("test output exists; choose a new --run-dir")
    run_root.mkdir(parents=True)
    (run_root / "test_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    data_root = Path(args.data_root).resolve()
    manifest = str(run_root / "train_dev_split.json")
    rows, configs = [], []
    for variant in VARIANT_SUBBANDS:
        output = run_root / variant
        command = [sys.executable, str(ROOT / "scripts/train_imagenet.py"),
                   "--dataset", "tiny-imagenet", "--data-root", str(data_root / "train"),
                   "--variant", variant, "--run-dir", str(output), "--split-manifest", manifest,
                   "--mcunet-root", args.mcunet_root, "--device", args.device,
                   "--epochs", "2", "--stop-after-epochs", "1", "--batch-size", str(args.batch_size),
                   "--workers", str(args.workers), "--max-train-batches", "2",
                   "--max-dev-batches", "1", "--no-progress"]
        if args.device.startswith("cuda"):
            command.append("--amp")
        execute(command)
        execute(command + ["--resume", str(output / "last.pt")])
        execute([sys.executable, str(ROOT / "scripts/eval_imagenet.py"),
                 "--dataset", "tiny-imagenet", "--data-root", str(data_root / "val"),
                 "--variant", variant, "--checkpoint", str(output / "best.pt"),
                 "--predictions", str(output / "val_predictions.csv"),
                 "--mcunet-root", args.mcunet_root, "--device", args.device,
                 "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                 "--max-batches", "2", "--no-progress"])
        config = json.loads((output / "config.json").read_text())
        configs.append(config)
        last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        with (output / "history.csv").open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        metrics = json.loads((output / "val_predictions.metrics.json").read_text())
        assert [int(row["epoch"]) for row in history] == [0, 1]
        assert abs(float(history[1]["lr"]) - 0.0005) < 1e-10, "scheduler did not resume correctly"
        assert last["epoch"] == 1 and last["scheduler"]["last_epoch"] == 2
        assert last["optimizer"]["state"] and last["rng"]
        assert not args.device.startswith("cuda") or last["scaler"]
        minimum = min(float(row["dev_loss"]) for row in history)
        assert abs(best["dev"]["loss"] - minimum) < 1e-7
        assert abs(last["best_dev_ce"] - minimum) < 1e-7
        assert len(last["classes"]) == 200
        stem = last["model"]["first_conv.conv.weight"]
        assert stem.shape[1] == input_channels(variant)
        assert config["initialization"]["extra_nonzero"] == 0
        if stem.shape[1] > 3:
            assert torch.count_nonzero(stem[:, 3:]).item() > 0, "high-frequency weights did not train"
        with (output / "val_predictions.csv").open(newline="", encoding="utf-8") as f:
            predictions = list(csv.DictReader(f))
        assert len(predictions) == metrics["samples"] == 2 * args.batch_size
        for prediction in predictions:
            top5 = [int(k) for k in prediction["pred_top5"].split()]
            assert len(set(top5)) == 5 and all(0 <= k < 200 for k in top5)
        plot_history(output / "history.csv", output / "curves.png",
                     f"{variant}: Tiny ImageNet pipeline test (truncated passes)")
        rows.append({"variant": variant, "train_samples_per_pass": int(history[-1]["train_samples"]),
                     "dev_samples_per_pass": int(history[-1]["dev_samples"]), "partial_passes": 2,
                     "selected_epoch": best["epoch"], "dev_ce": best["dev"]["loss"],
                     "dev_top1": best["dev"]["top1"], "tiny_val_samples": metrics["samples"],
                     "tiny_val_top1": metrics["top1"], "tiny_val_top5": metrics["top5"],
                     "official_val_top1": "", "official_val_top5": "",
                     "status": "pipeline_test_passed", "notes": "200 classes; truncated passes; not benchmark accuracy; epoch is 0-based"})
        print(f"PASS {variant}: update, checkpoint, resume, predictions, curves", flush=True)
    for key in ("ll_or_rgb_sha256", "classifier_sha256"):
        assert len({config["initialization"][key] for config in configs}) == 1
    assert len({config["manifest"]["sample_fingerprint"] for config in configs}) == 1
    assert len({tuple(config["classes"]) for config in configs}) == 1
    with (run_root / "pipeline_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "passed", "scope": "Tiny ImageNet 200-class pipeline test only",
              "variants": list(VARIANT_SUBBANDS), "same_ll_initialization": True,
              "same_classifier_initialization": True, "same_class_mapping_and_split": True,
              "resume_checked": True, "full_training_completed": False,
              "offline_wavelet_dataset_created": False}
    (run_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("All six pipeline tests passed:", run_root, flush=True)


if __name__ == "__main__":
    main()
