#!/usr/bin/env python3
"""Train raw/padded-LL1 mixed or single-view controls on fixed 128k/20k."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from padded_input import PaddedMix160, view_for_step


HANDOFF_COMMIT = "432bc72840dbe01820e4eefbcda710a16fb73f0d"
MCUNET_COMMIT = "9c164f8483003a5c6445871d94d30720aecc5918"
PROTOCOL_SHA256 = "0f414ee9a709cbca507b29d36f6cc77cf77b47a0039b8aad1474211d2f8d33b8"
PRETRAINED_SHA256 = "63565b5773a8aa70d75baf5a349f5f384da649804c48a5e7af1f655460158c7d"
CONFIG_SHA256 = "bf1dcb428e0bc21ad102463ee4764eefddaf4151363f8d43898a1e2e52f80485"
TRAIN_SAMPLES = 128_000
DEV_SAMPLES = 20_000
BATCH_SIZE = 32
UPDATES_PER_EPOCH = 4_000
MAX_EPOCHS = 20
CHECKPOINT_INTERVAL_UPDATES = 100
CHECKPOINT_INTERVAL_SECONDS = 60


class RGB160(torch.nn.Module):
    """Keep the pretrained MCUNet stem and backbone unchanged."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.last_logits = None

    def forward(self, x):
        if tuple(x.shape[1:]) != (3, 160, 160):
            raise ValueError("RGB baseline requires 3x160x160 input")
        logits = self.backbone(x * 2.0 - 1.0)
        self.last_logits = logits.detach()
        return logits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("rgb_only", "ll1_pad_only", "rgb_ll1_pad_mix"), required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", type=Path,
                        help="original ILSVRC TRAIN directory containing 1000 WNID folders")
    source.add_argument("--train-tar", type=Path,
                        help="complete uncompressed ILSVRC2012_img_train.tar; no JPEG extraction")
    parser.add_argument("--audit-workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--accept-local-content-sha256",
                        help="explicitly accept this audited local hash; result is not an exact teammate-content control")
    parser.add_argument("--seal-dir", type=Path, required=True,
                        help="new directory for the materialized frozen manifest and content audit")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mcunet-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prefetch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace_with_retry(source, destination, attempts=8):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.1 * (2 ** attempt), 2.0))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def append_event(path, event, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"time": time.time(), "event": event, **details},
                                sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_protocol_modules(handoff_root):
    handoff_root = Path(handoff_root).resolve()
    if not (handoff_root / ".git").exists():
        raise ValueError("handoff-root must be the pinned Git checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(handoff_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != HANDOFF_COMMIT:
        raise ValueError(f"handoff commit mismatch: {commit}")

    sys.path.insert(0, str(handoff_root))
    modules = {
        name: importlib.import_module("experiments.w80_e23." + name)
        for name in ("execution", "data", "model", "runtime")
    }
    membership = importlib.import_module("experiments.w80_e23.handoff.membership")

    root = handoff_root / "experiments" / "w80_e23"
    if sha256(root / "protocol_v1_frozen.json") != PROTOCOL_SHA256:
        raise ValueError("frozen protocol checksum mismatch")
    frozen = read_json(root / "results" / "source_freeze_resource_v2.json")
    for name, checksum in frozen["sources"].items():
        if sha256(root / name) != checksum:
            raise ValueError("frozen execution source mismatch: " + name)
    modules["membership"] = membership
    modules["root"] = root
    modules["commit"] = commit
    modules["source_hashes"] = frozen["sources"]
    return modules


def verify_upstream(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                    text=True).strip()
    if commit != MCUNET_COMMIT or dirty:
        raise ValueError("MCUNet upstream must be the clean pinned commit " + MCUNET_COMMIT)
    return commit


def verify_original_content(train_root, rows, data_module, manifest_path):
    from PIL import Image, features

    manifest_hash = sha256(manifest_path)
    receipt_path = Path(manifest_path).parent / "content_audit.json"
    expected = {
        "schema": 1,
        "train_root": str(train_root),
        "manifest_sha256": manifest_hash,
        "images": len(rows),
        "Pillow": Image.__version__,
        "jpeg": features.version("jpg"),
        "byte_identity": "passed",
        "decoded_rgb_identity": "passed",
    }
    if receipt_path.exists():
        existing = read_json(receipt_path)
        if all(existing.get(key) == value for key, value in expected.items()):
            return existing
        raise ValueError("existing content audit receipt differs; inspect the seal directory")

    for count, row in enumerate(rows, start=1):
        relative = data_module.safe_relative(row["path"])
        path = train_root / relative
        if not path.resolve().is_relative_to(train_root):
            raise ValueError("image escapes original TRAIN root: " + row["path"])
        with path.open("rb") as stream:
            raw = stream.read()
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if hashlib.sha256(raw).hexdigest() != row["sha"]:
            raise ValueError("original image bytes differ from the executed seal: " + row["path"])
        with Image.open(io.BytesIO(raw)) as source:
            rgb = source.convert("RGB")
        rgb_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if rgb_sha != row["rgb"]:
            raise ValueError("decoded RGB differs from the executed seal: " + row["path"])
        if count % 10_000 == 0:
            print(f"content audit: {count}/{len(rows)} original images verified", flush=True)

    receipt = {
        **expected,
        "verified_at_unix": time.time(),
        "order": "frozen manifest row order",
    }
    atomic_json(receipt_path, receipt)
    return receipt


def verify_inputs(args, modules):
    expected_env = read_json(modules["root"] / "protocol_v1_frozen.json")["execution_environment"]
    modules["execution"].setup(seed=2026)
    if args.train_tar is not None:
        return verify_tar_inputs(args, modules, expected_env)

    bundle = modules["root"] / "handoff"
    train_root = Path(args.train_root).resolve()
    if train_root.name != "train" or not train_root.is_dir():
        raise ValueError("train-root must be the original ImageNet TRAIN directory")
    seal_dir = Path(args.seal_dir).resolve()
    manifest_path = seal_dir / "manifest.tsv"
    materialized_now = False
    if not seal_dir.exists():
        materialize_result = modules["membership"].materialize(bundle, train_root, seal_dir)
        if "original bytes and decoded RGB reconstructed" not in materialize_result.get(
                "content_identity", ""):
            raise ValueError("original-image materialization did not verify the frozen content seal")
        materialized_now = True
        print("content audit: all 148000 original images verified", flush=True)
    elif not manifest_path.is_file():
        raise FileNotFoundError("seal-dir exists without manifest.tsv; inspect it and use a new directory")

    weights = Path(args.weights).resolve()
    if sha256(weights) != PRETRAINED_SHA256:
        raise ValueError("pretrained weight checksum mismatch")
    config = modules["root"].parent / "mcunet_dwt" / "weights" / "mcunet-256kb-1mb_imagenet.json"
    if sha256(config) != CONFIG_SHA256:
        raise ValueError("pinned MCUNet config checksum mismatch")
    upstream_commit = verify_upstream(args.mcunet_root)

    membership_result = modules["membership"].validate(bundle, manifest_path)
    expected_classes, member_rows = modules["membership"].read_members(bundle)
    classes = expected_classes
    all_rows = modules["data"].read_manifest(manifest_path)
    if [(r["role"], r["path"], r["label"]) for r in all_rows] != [
            (r["role"], r["path"], r["label"]) for r in member_rows]:
        raise ValueError("manifest member order differs from the frozen cohort")
    train_rows = [row for row in all_rows if row["role"] == "train"]
    dev_rows = [row for row in all_rows if row["role"] == "dev"]
    if (len(train_rows), len(dev_rows)) != (TRAIN_SAMPLES, DEV_SAMPLES):
        raise ValueError("frozen train/dev cardinality mismatch")

    if materialized_now:
        from PIL import Image, features
        content_audit = {
            "schema": 1,
            "train_root": str(train_root),
            "manifest_sha256": sha256(manifest_path),
            "images": len(all_rows),
            "Pillow": Image.__version__,
            "jpeg": features.version("jpg"),
            "byte_identity": "passed",
            "decoded_rgb_identity": "passed",
            "method": "pinned handoff materialize",
            "verified_at_unix": time.time(),
        }
        atomic_json(seal_dir / "content_audit.json", content_audit)
    else:
        content_audit = verify_original_content(
            train_root, all_rows, modules["data"], manifest_path
        )

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.import_module("torchvision").__version__,
    }
    from PIL import Image, features
    versions["Pillow"] = Image.__version__
    versions["jpeg"] = features.version("jpg")
    if versions != expected_env:
        raise RuntimeError(f"environment mismatch: expected {expected_env}, got {versions}")

    return {
        "bundle": bundle,
        "config": config,
        "train_root": train_root,
        "weights": weights,
        "classes": classes,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "manifest": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "membership_validation": membership_result,
        "content_audit": content_audit,
        "upstream_commit": upstream_commit,
        "versions": versions,
        "loader_local": {"train_root": str(train_root)},
        "storage": {"type": "directory"},
    }


def verify_tar_inputs(args, modules, expected_env):
    from PIL import Image, features
    import w80_tar_source

    versions = dict(python=platform.python_version(), torch=torch.__version__,
                    torchvision=importlib.import_module("torchvision").__version__,
                    Pillow=Image.__version__, jpeg=features.version("jpg"))
    if versions != expected_env:
        raise RuntimeError(f"environment mismatch: expected {expected_env}, got {versions}")
    weights = args.weights.resolve()
    if sha256(weights) != PRETRAINED_SHA256:
        raise ValueError("pretrained weight checksum mismatch")
    config = modules["root"].parent / "mcunet_dwt" / "weights" / "mcunet-256kb-1mb_imagenet.json"
    if sha256(config) != CONFIG_SHA256:
        raise ValueError("pinned MCUNet config checksum mismatch")
    upstream_commit = verify_upstream(args.mcunet_root)
    bundle = modules["root"] / "handoff"
    archive, seal_dir = args.train_tar.resolve(), args.seal_dir.resolve()
    receipt, index_receipt = w80_tar_source.prepare_tar_seal(
        archive, bundle, seal_dir, modules["membership"], workers=args.audit_workers,
        accepted_content_sha256=args.accept_local_content_sha256)
    local_accepted = args.accept_local_content_sha256 is not None
    manifest = seal_dir / ("manifest.local_accepted.tsv" if local_accepted else "manifest.tsv")
    membership_result = (w80_tar_source.validate_local_membership(bundle, manifest, modules["membership"])
                         if local_accepted else modules["membership"].validate(bundle, manifest))
    classes, members = modules["membership"].read_members(bundle)
    rows = modules["data"].read_manifest(manifest)
    if [(r["role"], r["path"], r["label"]) for r in rows] != [
            (r["role"], r["path"], r["label"]) for r in members]:
        raise ValueError("tar manifest order differs from frozen cohort")
    train_rows = [r for r in rows if r["role"] == "train"]
    dev_rows = [r for r in rows if r["role"] == "dev"]
    if (len(train_rows), len(dev_rows)) != (TRAIN_SAMPLES, DEV_SAMPLES):
        raise ValueError("tar manifest train/dev cardinality mismatch")
    return dict(bundle=bundle, config=config, train_root=None, weights=weights,
                classes=classes, train_rows=train_rows, dev_rows=dev_rows,
                manifest=manifest, manifest_sha256=sha256(manifest),
                membership_validation=membership_result, content_audit=receipt,
                upstream_commit=upstream_commit, versions=versions,
                dataset_alignment=dict(same_frozen_membership=True,
                                       matches_teammate_content_seal=not local_accepted,
                                       user_accepted_local_content=local_accepted,
                                       local_manifest_sha256=sha256(manifest),
                                       reference_manifest_sha256=modules["membership"].sealed()["manifest_sha256"]),
                loader_local={"archive": str(archive), "index_path": str(seal_dir / "tar_index.tsv")},
                storage={"type": "tar", "archive": index_receipt["archive"],
                         "index_sha256": index_receipt["index_sha256"],
                         "adapter_sha256": sha256(Path(w80_tar_source.__file__))})


def selected_epoch(history):
    """First occurrence of the minimum complete development CE."""
    return min(history, key=lambda row: row["dev_ce"], default={}).get("epoch")


@contextmanager
def run_lock(run_dir):
    """OS-owned lock prevents duplicate training and releases after a crash."""
    lock_path = Path(run_dir).resolve().with_name(Path(run_dir).name + ".training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def initial_progress():
    return {
        "epoch": 0,
        "step": 0,
        "sampler_position": 0,
        "training_exposures": 0,
        "phase": "train",
        "train_loss_sum": 0.0,
        "train_top1": 0,
        "train_top5": 0,
        "train_samples": 0,
        "train_rgb_samples": 0,
        "train_ll1_pad_samples": 0,
        "epoch_lr_start": None,
        "history": [],
        "early_stop": vars(importlib.import_module(
            "experiments.w80_e23.runtime").EarlyStop()),
        "best_ce": None,
        "complete": False,
        "stop_reason": None,
    }


def write_history(path, history, replace):
    fields = ("epoch", "updates", "training_exposures", "lr_start", "lr_end",
              "train_loss", "train_top1", "train_top5", "dev_ce", "dev_top1",
              "dev_top5", "rgb_dev_ce", "rgb_dev_top1", "rgb_dev_top5",
              "train_rgb_samples", "train_ll1_pad_samples", "selected_best")
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        replace_with_retry(temporary, path)
    else:
        temporary.rename(path)


def model_and_optimizers(args, inputs, modules):
    modules["execution"].setup(seed=2026)
    backbone = modules["model"].load_pretrained(
        inputs["config"], inputs["weights"], args.mcunet_root
    )
    if (backbone.classifier.out_features != 1000 or
            backbone.first_conv.conv.in_channels != 3 or
            tuple(backbone.first_conv.conv.stride) != (2, 2)):
        raise ValueError("loaded model is not the unmodified 1000-class RGB160 MCUNet-in2")
    model = PaddedMix160(backbone, args.variant).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                weight_decay=4e-5, dampening=0.0, nesterov=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, modules["runtime"].lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def atomic_checkpoint(checkpoints, state, best):
    for attempt in range(8):
        try:
            return checkpoints.save(state, best=best)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.1 * (2 ** attempt), 2.0))


def evaluate(model, rows, loader, local, args):
    model.eval()
    result = {view: dict(count=0, top1=0, top5=0, loss_sum=0.0) for view in ("ll1_pad", "rgb")}
    batches = loader(local, rows, "dev", workers=args.workers, prefetch=args.prefetch)
    with torch.inference_mode():
        for images, labels in batches:
            if len(labels) != BATCH_SIZE:
                raise ValueError("dev batch must remain32")
            images, target = images.cuda(), labels.cuda()
            for view, totals in result.items():
                model.eval_view = view
                logits = model(images)
                losses = F.cross_entropy(logits, target, reduction="none")
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("nonfinite " + view + " development CE")
                pred = logits.topk(5, dim=1).indices
                totals["loss_sum"] += float(losses.double().sum())
                totals["top1"] += int((pred[:, 0] == target).sum())
                totals["top5"] += int((pred == target[:, None]).any(dim=1).sum())
                totals["count"] += len(labels)
    model.last_logits = None
    metrics = {}
    for view, totals in result.items():
        if totals["count"] != DEV_SAMPLES:
            raise ValueError("incomplete " + view + " development pass")
        metrics[view] = dict(ce=totals["loss_sum"]/DEV_SAMPLES,
                             top1=100.0*totals["top1"]/DEV_SAMPLES,
                             top5=100.0*totals["top5"]/DEV_SAMPLES, images=DEV_SAMPLES)
    return metrics


def train(args, inputs, modules):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in fyp-w80-rgb")
    if inputs["train_root"] is not None and args.run_dir.resolve().is_relative_to(inputs["train_root"]):
        raise ValueError("run-dir must not be inside the source dataset")

    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    pointer = checkpoint_dir / "checkpoints.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError("run-dir is not empty; use --resume or choose a new directory")
    if args.resume and not pointer.exists():
        raise FileNotFoundError("--resume requested but no checkpoint pointer exists")
    if (run_dir / "complete.json").exists():
        raise FileExistsError("run is already complete")
    run_dir.mkdir(parents=True, exist_ok=True)

    local = inputs["loader_local"]
    if inputs["storage"]["type"] == "tar":
        from w80_tar_source import loader
    else:
        loader = modules["execution"].loader
    model, optimizer, scheduler, scaler = model_and_optimizers(args, inputs, modules)
    parts = (model, optimizer, scheduler, scaler)
    contract = {
        "variant": args.variant,
        "input_processing": {
            "wavelet": "orthonormal Haar, level1 LL only, divide coefficient by2 for DC correction",
            "coefficient_size": [80, 80], "canvas_size": [160, 160],
            "placement": "top-left; right and bottom zero in [0,1] domain",
            "normalization_order": "RGB geometric view -> corrected LL -> zero-pad -> 2*x-1; pad becomes -1",
            "upsampling": False, "inference_skip_zero_regions": False,
        },
        "training_mixture": {
            "variant": args.variant, "unit": "whole microbatch32",
            "mixed_counts_per_epoch": {"rgb": 2000, "ll1_pad": 2000},
            "schedule": "sha256 pad-mix-v1:2026:epoch -> local torch Generator randperm4000",
            "extra_updates_or_duplicates": False,
            "schedule_note": "mixture schedule applies only to rgb_ll1_pad_mix; controls use their sole view",
        },
        "primary_development_view": "ll1_pad",
        "secondary_development_view": "rgb",
        "additional_augmentation": None,
        "batch_norm_sharing": "one shared pretrained BN set; no split BN or recalibration",
        "execution_scope": "reference-inspired 160 adaptation with explicit assumptions; not exact224 reproduction or frozen teammate recipe",
        "experiment_sources": {
            name: sha256(Path(__file__).parent / name)
            for name in ("padded_input.py", "w80_tar_source.py", "direct_io.py")
        },
        "resource_watermark_pause": False,
        "official_val_used": False,
        "imagenet_v2_used": False,
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": PROTOCOL_SHA256,
        "handoff_commit": HANDOFF_COMMIT,
        "mcunet_commit": inputs["upstream_commit"],
        "manifest_sha256": inputs["manifest_sha256"],
        "weights_sha256": PRETRAINED_SHA256,
        "config_sha256": CONFIG_SHA256,
        "sources": modules["source_hashes"],
        "batch": BATCH_SIZE,
        "bn_batch": BATCH_SIZE,
        "gradient_accumulation": 1,
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "seed": 2026,
        "loss": "cross_entropy_only",
        "all_parameters_trainable": True,
        "batch_norm_running_stats": "updated in training mode with microbatch 32",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "peak_lr": 0.001,
        "lr_schedule": "linear warmup for 2 epochs, then cosine decay to zero by epoch 20",
        "warmup_epochs": 2,
        "epochs_max": MAX_EPOCHS,
        "minimum_epochs_before_early_stop": 6,
        "early_stop": {
            "minimum_ce_improvement": 0.001,
            "patience": 4,
            "plateau_window": 4,
            "maximum_absolute_ols_slope": 0.001,
            "maximum_ce_range": 0.005,
        },
        "checkpoint_selection": "lowest complete LL1-pad160 development CE; first exact tie; RGB metrics secondary",
        "checkpoint_interval_updates": CHECKPOINT_INTERVAL_UPDATES,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "train_root": str(inputs["train_root"]) if inputs["train_root"] is not None else None,
        "storage": inputs["storage"],
        "dataset_alignment": inputs.get("dataset_alignment", {"matches_teammate_content_seal": True}),
    }
    run_config = {
        **contract,
        "workers": args.workers,
        "prefetch": args.prefetch,
        "classes": len(inputs["classes"]),
        "class_mapping_sha256": sha256(inputs["bundle"] / "classes.json"),
        "train_images": TRAIN_SAMPLES,
        "dev_images": DEV_SAMPLES,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "manifest_iteration": "frozen TSV order for identity audit; per-epoch seeded permutation for training",
        "identity_validation": "original-file SHA256 and decoded RGB SHA256",
        "training_transform": {
            "input": "original RGB",
            "crop": "RandomResizedCrop160",
            "scale": [0.08, 1.0],
            "ratio": [0.75, 1.3333333333333333],
            "interpolation": "bicubic",
            "antialias": True,
            "horizontal_flip_probability": 0.5,
        },
        "dev_transform": {
            "input": "original RGB",
            "resize_short_edge": 182,
            "interpolation": "bicubic",
            "antialias": True,
            "center_crop": 160,
        },
        "normalization": "2*x-1",
        "model": "MCUNet-in2, RGB160 and/or LL1 zero-padded160, original stride2/full backbone/head",
        "conv_linear_macs_per_image": 67287200,
        "mac_scope": "CNN Conv/Linear only; excludes Haar/resize/decode/normalization",
        "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        "batch_norm": "training mode; affine parameters trainable and running statistics updated",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "first_conv": {
            "in_channels": model.backbone.first_conv.conv.in_channels,
            "out_channels": model.backbone.first_conv.conv.out_channels,
            "kernel": list(model.backbone.first_conv.conv.kernel_size),
            "stride": list(model.backbone.first_conv.conv.stride),
        },
        "versions": inputs["versions"],
    }
    config_path = run_dir / "config.json"
    if args.resume:
        existing_config = read_json(config_path)
        if any(existing_config.get(key) != value for key, value in contract.items()):
            raise ValueError("resume configuration differs from the frozen training contract")
    else:
        atomic_json(config_path, run_config)

    checkpoints = modules["runtime"].Checkpoints(checkpoint_dir)
    progress = initial_progress()
    if args.resume:
        progress = modules["runtime"].restore(checkpoints.load(), *parts, contract)
        if progress["phase"] == "dev":
            append_event(run_dir / "events.jsonl", "restart_full_dev",
                         epoch=progress["epoch"] + 1, step=progress["step"])
    else:
        atomic_checkpoint(checkpoints, modules["runtime"].capture(
            *parts, progress, contract), best=False)
        append_event(run_dir / "events.jsonl", "start", step=0)
    write_history(run_dir / "history.csv", progress["history"], replace=True)

    train_rows, dev_rows = inputs["train_rows"], inputs["dev_rows"]
    atomic_json(run_dir / "progress.json", {
        "status": "training", "epoch": progress["epoch"] + 1, "step": progress["step"],
        "sampler_position": progress["sampler_position"], "training_exposures": progress["training_exposures"],
        "train_rgb_samples": progress["train_rgb_samples"],
        "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
    })
    last_save = time.monotonic()
    try:
        while not progress["complete"]:
            epoch = progress["epoch"]
            if epoch >= MAX_EPOCHS:
                progress["complete"] = True
                progress["stop_reason"] = "maximum20epochs"
                break

            if progress["phase"] == "train":
                order = modules["data"].epoch_order(TRAIN_SAMPLES, 2026, epoch)
                order_sha = hashlib.sha256(
                    torch.tensor(order, dtype=torch.int64).numpy().tobytes()
                ).hexdigest()
                if progress["sampler_position"] == 0:
                    progress["epoch_lr_start"] = optimizer.param_groups[0]["lr"]
                batches = loader(
                    local, train_rows, "train", epoch=epoch, order=order,
                    cursor=progress["sampler_position"], workers=args.workers,
                    prefetch=args.prefetch,
                )
                try:
                    for images, labels in batches:
                        if len(labels) != BATCH_SIZE:
                            raise ValueError("training BN/effective batch must remain 32")
                        gpu_free_before_attempt = torch.cuda.mem_get_info()[0]
                        images, labels = images.cuda(), labels.cuda()
                        try:
                            model.set_train_step(progress["step"])
                            batch_view = view_for_step(args.variant, progress["step"])
                            loss, _ = modules["runtime"].transactional_step(
                                model, optimizer, scheduler, scaler, images, labels,
                                progress, contract,
                            )
                            logits = model.last_logits
                            pred = logits.topk(5, dim=1).indices
                            progress["train_loss_sum"] += loss * len(labels)
                            progress["train_top1"] += int((pred[:, 0] == labels).sum())
                            progress["train_top5"] += int((pred == labels[:, None]).any(dim=1).sum())
                            progress["train_samples"] += len(labels)
                            progress["train_" + batch_view + "_samples"] += len(labels)
                            progress["step"] += 1
                            progress["sampler_position"] += len(labels)
                            progress["training_exposures"] += len(labels)
                            model.last_logits = None
                        except BaseException as error:
                            model.last_logits = None
                            append_event(run_dir / "events.jsonl", "batch_failure",
                                         epoch=epoch + 1, committed_step=progress["step"],
                                         sampler_position=progress["sampler_position"], attempted_images=BATCH_SIZE,
                                         gpu_free_before_attempt=gpu_free_before_attempt,
                                         error_type=type(error).__name__, error=str(error))
                            raise
                        if (progress["step"] % CHECKPOINT_INTERVAL_UPDATES == 0 or
                                time.monotonic() - last_save >= CHECKPOINT_INTERVAL_SECONDS):
                            state = modules["runtime"].capture(*parts, progress, contract)
                            atomic_checkpoint(checkpoints, state, best=False)
                            atomic_json(run_dir / "progress.json", {
                                "status": "training", "epoch": epoch + 1,
                                "step": progress["step"],
                                "sampler_position": progress["sampler_position"],
                                "training_exposures": progress["training_exposures"],
                                "train_rgb_samples": progress["train_rgb_samples"],
                                "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
                                "checkpoint_sha256": sha256(checkpoint_dir / json.loads(
                                    pointer.read_text(encoding="utf-8"))["latest"]),
                            })
                            print(f"epoch {epoch + 1}/{MAX_EPOCHS}: step={progress['step']}, "
                                  f"images={progress['sampler_position']}/{TRAIN_SAMPLES}, "
                                  f"train_ce={progress['train_loss_sum']/progress['train_samples']:.6f}", flush=True)
                            last_save = time.monotonic()
                except BaseException:
                    state = modules["runtime"].capture(*parts, progress, contract)
                    atomic_checkpoint(checkpoints, state, best=False)
                    append_event(run_dir / "events.jsonl", "training_error",
                                 epoch=epoch + 1, step=progress["step"])
                    raise

                if (progress["sampler_position"] != TRAIN_SAMPLES or
                        progress["train_samples"] != TRAIN_SAMPLES):
                    raise ValueError("training epoch did not consume all 128000 ordered samples")
                progress["phase"] = "dev"
                state = modules["runtime"].capture(*parts, progress, contract)
                atomic_checkpoint(checkpoints, state, best=False)
                expected_rgb = 64000 if args.variant == "rgb_ll1_pad_mix" else (128000 if args.variant == "rgb_only" else 0)
                if progress["train_rgb_samples"] != expected_rgb or progress["train_ll1_pad_samples"] != 128000-expected_rgb:
                    raise ValueError("complete epoch mixture count differs from declared contract")
                atomic_json(run_dir / "progress.json", {
                    "status": "evaluating_dev", "epoch": epoch + 1, "step": progress["step"],
                    "training_exposures": progress["training_exposures"],
                    "sampler_position": progress["sampler_position"],
                    "train_rgb_samples": progress["train_rgb_samples"],
                    "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
                })
                append_event(run_dir / "events.jsonl", "dev_start",
                             epoch=epoch + 1, step=progress["step"], order_sha256=order_sha)

            if progress["phase"] == "dev":
                metrics = evaluate(model, dev_rows, loader, local, args)
                dev = metrics["ll1_pad"]
                early = modules["runtime"].EarlyStop()
                early.__dict__.update(progress["early_stop"])
                selected_best, early_stop = early.observe(dev["ce"])
                completed_epoch = progress["epoch"] + 1
                row = {
                    "epoch": completed_epoch,
                    "updates": progress["step"],
                    "training_exposures": progress["training_exposures"],
                    "lr_start": progress["epoch_lr_start"],
                    "lr_end": optimizer.param_groups[0]["lr"],
                    "train_loss": progress["train_loss_sum"] / TRAIN_SAMPLES,
                    "train_top1": 100.0 * progress["train_top1"] / TRAIN_SAMPLES,
                    "train_top5": 100.0 * progress["train_top5"] / TRAIN_SAMPLES,
                    "dev_ce": dev["ce"],
                    "dev_top1": dev["top1"],
                    "dev_top5": dev["top5"],
                    "rgb_dev_ce": metrics["rgb"]["ce"],
                    "rgb_dev_top1": metrics["rgb"]["top1"],
                    "rgb_dev_top5": metrics["rgb"]["top5"],
                    "train_rgb_samples": progress["train_rgb_samples"],
                    "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
                    "selected_best": selected_best,
                }
                progress["history"].append(row)
                progress["early_stop"] = vars(early).copy()
                progress["best_ce"] = early.best
                progress["epoch"] += 1
                progress["sampler_position"] = 0
                progress["phase"] = "train"
                progress["train_loss_sum"] = 0.0
                progress["train_top1"] = 0
                progress["train_top5"] = 0
                progress["train_samples"] = 0
                progress["train_rgb_samples"] = 0
                progress["train_ll1_pad_samples"] = 0
                progress["epoch_lr_start"] = None
                if early_stop:
                    progress["complete"] = True
                    progress["stop_reason"] = "development_plateau"
                elif progress["epoch"] == MAX_EPOCHS:
                    progress["complete"] = True
                    progress["stop_reason"] = "maximum20epochs"
                write_history(run_dir / "history.csv", progress["history"], replace=True)
                state = modules["runtime"].capture(*parts, progress, contract)
                atomic_checkpoint(checkpoints, state, best=selected_best)
                append_event(run_dir / "events.jsonl", "dev_complete",
                             epoch=completed_epoch, step=progress["step"],
                             dev_ce=dev["ce"], rgb_dev_ce=metrics["rgb"]["ce"],
                             selected_best=selected_best,
                             early_stop=early_stop)
                atomic_json(run_dir / "progress.json", {
                    "status": "complete" if progress["complete"] else "training",
                    "epoch": progress["epoch"], "step": progress["step"],
                    "training_exposures": progress["training_exposures"],
                    "best_dev_ce": progress["best_ce"],
                    "stop_reason": progress["stop_reason"],
                })
                print(f"completed epoch {completed_epoch}: dev_ce={dev['ce']:.6f}, "
                      f"LL1_pad_top1={dev['top1']:.3f}, "
                      f"RGB_top1={metrics['rgb']['top1']:.3f}, best={selected_best}", flush=True)

    except BaseException:
        append_event(run_dir / "events.jsonl", "run_interrupted",
                     epoch=progress["epoch"], step=progress["step"],
                     sampler_position=progress["sampler_position"])
        raise

    state = modules["runtime"].capture(*parts, progress, contract)
    atomic_checkpoint(checkpoints, state, best=False)
    atomic_json(run_dir / "complete.json", {
        "variant": args.variant,
        "completed_epochs": progress["epoch"],
        "updates": progress["step"],
        "training_exposures": progress["training_exposures"],
        "best_dev_ce": progress["best_ce"],
        "best_epoch": selected_epoch(progress["history"]),
        "stop_reason": progress["stop_reason"],
        "contract": contract,
    })


def main(argv=None):
    args = parse_args(argv)
    modules = load_protocol_modules(args.handoff_root)
    inputs = verify_inputs(args, modules)
    summary = {
        "status": "preflight_passed",
        "train": len(inputs["train_rows"]),
        "dev": len(inputs["dev_rows"]),
        "classes": len(inputs["classes"]),
        "manifest_sha256": inputs["manifest_sha256"],
        "weights_sha256": PRETRAINED_SHA256,
        "versions": inputs["versions"],
        "membership_validation": inputs["membership_validation"],
        "content_audit": inputs["content_audit"],
        "mcunet_commit": inputs["upstream_commit"],
        "storage": inputs["storage"],
        "dataset_alignment": inputs.get("dataset_alignment", {"matches_teammate_content_seal": True}),
    }
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)
    if args.validate_only:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    with run_lock(args.run_dir):
        train(args, inputs, modules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
