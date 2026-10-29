#!/usr/bin/env python3
"""Download ImageNet-1K 256px Parquet and extract a reproducible balanced subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tqdm import tqdm


REPO = "benjamin-paine/imagenet-1k-256x256"
REVISION = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
API_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data"
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def inventory():
    response = requests.get(API_URL, params={"limit": 100}, timeout=60)
    response.raise_for_status()
    files = {entry["path"]: entry for entry in response.json()}
    paths = [f"data/train-{i:05d}-of-00040.parquet" for i in range(40)]
    paths += [f"data/validation-{i:05d}-of-00002.parquet" for i in range(2)]
    missing = [path for path in paths if path not in files]
    if missing:
        raise RuntimeError(f"dataset inventory is missing files: {missing}")
    return [(path, files[path]["size"]) for path in paths]


def download_one(item, root: Path):
    path, expected_size = item
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == expected_size:
        return 0
    part = target.with_suffix(target.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected_size:
        raise IOError(f"oversized partial file: {part}")
    chunk_size = 16 * 1024 * 1024
    while offset < expected_size:
        end = min(expected_size, offset + chunk_size) - 1
        for attempt in range(6):
            try:
                # The range is also part of the query, preventing stale cached
                # redirects from returning the bytes for an earlier range.
                url = f"{BASE_URL}/{path}?download=true&range={offset}-{end}&attempt={attempt}&t={int(time.time())}"
                with session().get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True,
                                   timeout=(20, 45)) as response:
                    response.raise_for_status()
                    wanted_range = f"bytes {offset}-{end}/{expected_size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != wanted_range:
                        raise IOError(f"incorrect HTTP range: {response.status_code}, {response.headers.get('Content-Range')}")
                    content = response.content
                    if len(content) != end - offset + 1:
                        raise IOError("short range response")
                with part.open("ab") as stream:
                    stream.write(content)
                offset = end + 1
                break
            except Exception as error:
                if attempt == 5:
                    raise
                print(f"Retry {path} offset={offset} attempt={attempt + 1}: {type(error).__name__}", flush=True)
                time.sleep(min(2 ** attempt, 15))
    # Parquet metadata validation catches truncation/format errors before use.
    pq.ParquetFile(part)
    part.replace(target)
    print(f"Downloaded {target.name}: {expected_size / 1e6:.1f} MB", flush=True)
    return expected_size


def download(root: Path, workers: int):
    files = inventory()
    total = sum(size for _, size in files)
    print(f"Pinned dataset revision {REVISION}: {len(files)} Parquet files, {total / 1e9:.2f} GB", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item, root): item[0] for item in files}
        with tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                bar.update(next(size for path, size in files if path == futures[future]))
    return files


def parquet_row_groups(root: Path, paths):
    for relative in paths:
        parquet = pq.ParquetFile(root / relative)
        for row_group in range(parquet.num_row_groups):
            yield relative, row_group, parquet.read_row_group(row_group, columns=["label"])["label"].to_pylist()


def class_names(root: Path, train_paths):
    metadata = pq.ParquetFile(root / train_paths[0]).schema_arrow.metadata or {}
    info = json.loads(metadata.get(b"huggingface", b"{}"))
    names = info.get("info", {}).get("features", {}).get("label", {}).get("names")
    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]
    # The mirror adds a placeholder for the unlabeled test split. It is not
    # a training class; all train/validation indices are checked separately.
    if isinstance(names, list) and len(names) == 1001 and names[-1] == "none":
        names = names[:1000]
    if not isinstance(names, list) or len(names) != 1000:
        raise RuntimeError("missing 1000-class label names in Parquet metadata")
    return names


def image_name(relative, row_group, row_index):
    return f"{Path(relative).stem}_{row_group:04d}_{row_index:06d}.jpg"


def select_training_rows(root, train_paths, per_class, seed, num_classes=1000):
    rng = random.Random(seed)
    populations = [0] * num_classes
    reservoirs: list[list[tuple[str, int, int]]] = [[] for _ in range(num_classes)]
    print("Scanning labels and selecting balanced training images...", flush=True)
    for relative, row_group, labels in tqdm(parquet_row_groups(root, train_paths), desc="scan train"):
        for local_index, raw_label in enumerate(labels):
            label = int(raw_label)
            if not 0 <= label < num_classes:
                raise ValueError(f"invalid class index: {label}")
            count = populations[label]
            item = (relative, row_group, local_index)
            if count < per_class:
                reservoirs[label].append(item)
            else:
                replacement = rng.randrange(count + 1)
                if replacement < per_class:
                    reservoirs[label][replacement] = item
            populations[label] += 1
    if any(count < per_class for count in populations):
        raise RuntimeError(f"some classes have too few examples: {min(populations)}")
    return populations, reservoirs


def extract(root: Path, output: Path, per_class: int, seed: int, files):
    completed = output / "dataset_info.json"
    if completed.exists():
        info = json.loads(completed.read_text(encoding="utf-8"))
        if (info["revision"], info["seed"], info["training_subset_per_class"]) != (REVISION, seed, per_class):
            raise ValueError("output already contains a different subset")
        print("The completed matching subset already exists.", flush=True)
        return
    train_paths = [path for path, _ in files if "/train-" in path]
    val_paths = [path for path, _ in files if "/validation-" in path]
    populations, reservoirs = select_training_rows(root, train_paths, per_class, seed)

    output.mkdir(parents=True, exist_ok=True)
    for label in range(1000):
        (output / "train" / f"{label:04d}").mkdir(parents=True, exist_ok=True)
        (output / "val" / f"{label:04d}").mkdir(parents=True, exist_ok=True)
    names = class_names(root, train_paths)
    mapping = {"class_to_idx": {f"{i:04d}": i for i in range(1000)},
               "label_names": dict(enumerate(names)), "ordering": "dataset Parquet class IDs (0-999)"}
    (output / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    chosen: dict[tuple[str, int], list[tuple[int, int]]] = {}
    manifest_rows = []
    for label, samples in enumerate(reservoirs):
        for sample_index, (path, row_group, row_index) in enumerate(sorted(samples)):
            chosen.setdefault((path, row_group), []).append((row_index, label))
            manifest_rows.append((label, sample_index, path, row_group, row_index))
    del reservoirs
    source_ids = {}
    print(f"Extracting {len(manifest_rows):,} images ({per_class} per class)...", flush=True)
    for (relative, row_group), entries in tqdm(chosen.items(), desc="extract train"):
        parquet = pq.ParquetFile(root / relative)
        rows = parquet.read_row_group(row_group, columns=["image"]).column("image").to_pylist()
        for row_index, label in entries:
            cell = rows[row_index]
            image_bytes = cell["bytes"] if isinstance(cell, dict) else cell.bytes
            source_ids[(relative, row_group, row_index)] = cell.get("path")
            destination = output / "train" / f"{label:04d}" / image_name(relative, row_group, row_index)
            destination.write_bytes(image_bytes)

    split_path = output / "subset_manifest.csv"
    with split_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "target", "source_image_id", "source_shard", "source_row_group", "source_row"])
        for label, sample_index, path, row_group, row_index in sorted(manifest_rows):
            filename = output / "train" / f"{label:04d}" / image_name(path, row_group, row_index)
            writer.writerow([filename.relative_to(output / "train").as_posix(), label,
                             source_ids[(path, row_group, row_index)], path, row_group, row_index])

    print("Extracting labeled 50,000-image validation split...", flush=True)
    val_count = 0
    val_classes = [0] * 1000
    with (output / "validation_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "target", "source_image_id", "source_shard", "source_row_group", "source_row"])
        for relative, row_group, labels in tqdm(parquet_row_groups(root, val_paths), desc="scan val"):
            parquet = pq.ParquetFile(root / relative)
            rows = parquet.read_row_group(row_group, columns=["image"])["image"].to_pylist()
            for row_index, (raw_label, cell) in enumerate(zip(labels, rows)):
                label = int(raw_label)
                if not 0 <= label < 1000:
                    raise ValueError(f"invalid validation class: {label}")
                image_bytes = cell["bytes"] if isinstance(cell, dict) else cell.bytes
                filename = image_name(relative, row_group, row_index)
                (output / "val" / f"{label:04d}" / filename).write_bytes(image_bytes)
                writer.writerow([f"{label:04d}/{filename}", label, cell.get("path"), relative, row_group, row_index])
                val_count += 1
                val_classes[label] += 1
    if val_count != 50_000 or set(val_classes) != {50}:
        raise RuntimeError(f"expected 50,000 validation images, extracted {val_count}")

    metadata = {
        "dataset": REPO,
        "revision": REVISION,
        "source_transform": "center crop to square, then Lanczos resize to 256x256",
        "training_population_images": sum(populations),
        "training_population_min_per_class": min(populations),
        "training_population_max_per_class": max(populations),
        "training_subset_per_class": per_class,
        "training_subset_images": 1000 * per_class,
        "validation_images": val_count,
        "seed": seed,
        "subset_manifest": str(split_path.resolve()),
        "subset_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    (output / "dataset_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.per_class < 2:
        parser.error("--per-class must be at least 2")
    files = download(args.cache_dir, args.download_workers)
    if not args.download_only:
        extract(args.cache_dir, args.output_dir, args.per_class, args.seed, files)


if __name__ == "__main__":
    main()
