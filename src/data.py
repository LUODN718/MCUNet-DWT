"""ImageNet loaders with a deterministic train/development split."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .wavelet import select_subbands


class NormalizeInput:
    """Pickleable RGB/Haar transform for Windows DataLoader workers."""

    def __init__(self, variant: str):
        self.variant = variant

    def __call__(self, tensor):
        if self.variant == "rgb":
            return (tensor - 0.5) / 0.5
        bands = select_subbands(tensor.unsqueeze(0), self.variant).squeeze(0)
        ll = (bands[:3] / 2.0 - 0.5) / 0.5
        return ll if bands.shape[0] == 3 else torch.cat([ll, bands[3:] / 0.5])


class TinyImageNet(Dataset):
    """Read the original train layout and the separately annotated val layout."""

    def __init__(self, root, transform=None):
        self.root = str(Path(root).resolve())
        root = Path(self.root)
        self.transform = transform
        self.classes = sorted((root.parent / "wnids.txt").read_text().splitlines())
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("wnids.txt must contain unique class IDs")
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        if root.name == "train":
            self.samples = []
            for name in self.classes:
                images = sorted((root / name / "images").glob("*.JPEG"))
                if not images:
                    raise ValueError(f"no training images for {name}")
                self.samples.extend((str(path), self.class_to_idx[name]) for path in images)
        elif root.name == "val":
            self.samples = []
            with (root / "val_annotations.txt").open(encoding="utf-8", newline="") as f:
                for row in csv.reader(f, delimiter="\t"):
                    name, class_id = row[:2]
                    if Path(name).name != name:
                        raise ValueError(f"invalid validation filename: {name}")
                    path = root / "images" / name
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    self.samples.append((str(path), self.class_to_idx[class_id]))
            self.samples.sort()
            names = [path for path, _ in self.samples]
            if len(set(names)) != len(names) or len(names) != len(list((root / "images").glob("*.JPEG"))):
                raise ValueError("validation images and annotations do not match")
        else:
            raise ValueError("Tiny ImageNet root must end in train or val; test has no labels")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return (self.transform(image) if self.transform else image), target


def _dataset(root, dataset_kind, transform=None):
    if dataset_kind == "tiny-imagenet":
        return TinyImageNet(root, transform)
    if dataset_kind != "imagenet":
        raise ValueError(f"unknown dataset: {dataset_kind}")
    return datasets.ImageFolder(root, transform=transform)


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

    return transforms.Compose(aug + [transforms.ToTensor(), NormalizeInput(variant)])


def _fingerprint(dataset):
    digest = hashlib.sha256()
    for path, target in dataset.samples:
        relative = Path(path).relative_to(dataset.root).as_posix()
        digest.update(f"{relative}\t{target}\n".encode())
    return digest.hexdigest()


def _make_manifest(dataset, manifest_path: Path, dev_fraction: float, seed: int):
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    by_class = {}
    for index, (_, target) in enumerate(dataset.samples):
        by_class.setdefault(target, []).append(index)
    rng = random.Random(seed)
    dev_indices = []
    for target in sorted(by_class):
        indices = list(by_class[target])
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError("each class needs at least two training images")
        n_dev = min(len(indices) - 1, max(1, round(len(indices) * dev_fraction)))
        dev_indices.extend(sorted(indices[:n_dev]))
    dev_indices = sorted(dev_indices)
    payload = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "num_samples": len(dataset.samples),
        "classes": dataset.classes,
        "sample_fingerprint": _fingerprint(dataset),
        "dev_indices": dev_indices,
        "dev_samples": [
            {"path": dataset.samples[i][0], "target": dataset.samples[i][1]}
            for i in dev_indices
        ],
    }
    sample_list = manifest_path.with_suffix(".samples.csv")
    dev_set = set(dev_indices)
    with sample_list.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "target", "split"])
        for index, (path, target) in enumerate(dataset.samples):
            writer.writerow([Path(path).relative_to(dataset.root).as_posix(), target,
                             "dev" if index in dev_set else "train"])
    payload["sample_list"] = str(sample_list.resolve())
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return set(dev_indices), payload


def _load_manifest(dataset, manifest_path: Path, dev_fraction: float, seed: int):
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (payload.get("num_samples") != len(dataset.samples)
                or payload.get("sample_fingerprint") != _fingerprint(dataset)
                or payload.get("classes") != dataset.classes
                or payload.get("seed") != seed
                or payload.get("dev_fraction") != dev_fraction):
            raise ValueError("existing manifest differs from this dataset/seed/split; use a new manifest path")
        indices = payload["dev_indices"]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
            raise ValueError("invalid development indices")
        return set(indices), payload
    return _make_manifest(dataset, manifest_path, dev_fraction, seed)


def build_train_dev(data_root: str | Path, variant: str, manifest_path: str | Path,
                    dev_fraction: float = 0.05, seed: int = 2026,
                    rgb_resolution: int = 160, batch_size: int = 256,
                    workers: int = 8, dataset_kind: str = "imagenet"):
    data_root = Path(data_root)
    raw = _dataset(data_root, dataset_kind)
    dev_set, manifest = _load_manifest(raw, Path(manifest_path), dev_fraction, seed)
    train_ds = _dataset(data_root, dataset_kind, build_transform(variant, True, rgb_resolution))
    dev_ds = _dataset(data_root, dataset_kind, build_transform(variant, False, rgb_resolution))
    train_indices = [i for i in range(len(raw.samples)) if i not in dev_set]
    dev_indices = sorted(dev_set)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=seed_worker)
    return (DataLoader(Subset(train_ds, train_indices), shuffle=True,
                       generator=torch.Generator().manual_seed(seed), **common),
            DataLoader(Subset(dev_ds, dev_indices), shuffle=False,
                       generator=torch.Generator().manual_seed(seed + 1), **common),
            raw.classes, manifest)


def build_val(data_root: str | Path, variant: str, batch_size: int = 256,
              workers: int = 8, rgb_resolution: int = 160, dataset_kind: str = "imagenet"):
    ds = _dataset(data_root, dataset_kind, build_transform(variant, False, rgb_resolution))
    loader = DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    return loader, ds.classes, ds


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

