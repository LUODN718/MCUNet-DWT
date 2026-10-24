"""ImageNet loaders with a deterministic train/development split."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .wavelet import select_subbands


class HaarTensor:
    """Torchvision transform applying the selected Haar subbands after RGB aug."""

    def __init__(self, variant: str):
        self.variant = variant

    def __call__(self, tensor):
        return select_subbands(tensor.unsqueeze(0), self.variant).squeeze(0)


def build_transform(variant: str, train: bool, rgb_resolution: int = 160):
    if train:
        spatial = transforms.RandomResizedCrop(
            rgb_resolution, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
        )
        aug = [spatial, transforms.RandomHorizontalFlip()]
    else:
        resize = int(rgb_resolution * 256 / 224)
        aug = [transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
               transforms.CenterCrop(rgb_resolution)]

    # RGB baseline follows the official MCUNet eval convention. For wavelet
    # inputs, LL is mapped back to the usual [-1, 1] range and high-frequency
    # signed coefficients remain centered at zero with scale 0.5.
    def normalize_wavelet(tensor):
        if variant == "rgb":
            return transforms.Normalize([0.5] * 3, [0.5] * 3)(tensor)
        bands = select_subbands(tensor.unsqueeze(0), variant).squeeze(0)
        n_bands = bands.shape[0] // 3
        ll = (bands[:3] / 2.0 - 0.5) / 0.5
        if n_bands == 1:
            return ll
        high = bands[3:] / 0.5
        return torch.cat([ll, high], dim=0)

    return transforms.Compose(aug + [transforms.ToTensor(), normalize_wavelet])


def _make_manifest(dataset, manifest_path: Path, dev_fraction: float, seed: int):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    by_class = {}
    for index, (_, target) in enumerate(dataset.samples):
        by_class.setdefault(target, []).append(index)
    rng = random.Random(seed)
    dev_indices = []
    for target in sorted(by_class):
        indices = list(by_class[target])
        rng.shuffle(indices)
        n_dev = max(1, round(len(indices) * dev_fraction))
        dev_indices.extend(sorted(indices[:n_dev]))
    dev_indices = sorted(dev_indices)
    payload = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "num_samples": len(dataset.samples),
        "dev_indices": dev_indices,
        "dev_samples": [
            {"path": dataset.samples[i][0], "target": dataset.samples[i][1]}
            for i in dev_indices
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return set(dev_indices), payload


def _load_manifest(dataset, manifest_path: Path, dev_fraction: float, seed: int):
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("num_samples") != len(dataset.samples):
            raise ValueError("existing manifest does not match the current ImageNet train directory")
        return set(payload["dev_indices"]), payload
    return _make_manifest(dataset, manifest_path, dev_fraction, seed)


def build_train_dev(data_root: str | Path, variant: str, manifest_path: str | Path,
                    dev_fraction: float = 0.05, seed: int = 2026,
                    rgb_resolution: int = 160, batch_size: int = 256,
                    workers: int = 8):
    data_root = Path(data_root)
    raw = datasets.ImageFolder(data_root)
    dev_set, manifest = _load_manifest(raw, Path(manifest_path), dev_fraction, seed)
    train_ds = datasets.ImageFolder(data_root, transform=build_transform(variant, True, rgb_resolution))
    dev_ds = datasets.ImageFolder(data_root, transform=build_transform(variant, False, rgb_resolution))
    train_indices = [i for i in range(len(raw.samples)) if i not in dev_set]
    dev_indices = sorted(dev_set)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0)
    return (DataLoader(Subset(train_ds, train_indices), shuffle=True, drop_last=True, **common),
            DataLoader(Subset(dev_ds, dev_indices), shuffle=False, **common),
            raw.classes, manifest)


def build_val(data_root: str | Path, variant: str, batch_size: int = 256,
              workers: int = 8, rgb_resolution: int = 160):
    ds = datasets.ImageFolder(data_root, transform=build_transform(variant, False, rgb_resolution))
    loader = DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    return loader, ds.classes, ds

