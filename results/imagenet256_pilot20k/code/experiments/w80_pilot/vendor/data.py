"""Deterministic original-TRAIN manifests, exclusions and sample-keyed views."""
import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def safe_relative(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or '\\' in path:
        raise ValueError('expected class/image relative TRAIN identity')
    return p.as_posix()


def select_manifest(rows, excluded, classes, seed=2026, train_n=128, dev_n=20):
    """Metadata only. Exclusions must cover all known reserved identities.

    rows: path,label,sha,rgb; exclusions: same keys (label optional).
    A shortage fails before publishing any manifest. No replacement samples.
    """
    if len(classes) != 1000 or sorted(classes) != classes or len(set(classes)) != 1000:
        raise ValueError('fixed sorted 1000-class mapping required')
    if train_n != 128 or dev_n != 20:
        raise ValueError('W80 requires 128/20 per class')
    forbidden = {k: {r[k] for r in excluded if r.get(k)} for k in ('path', 'sha', 'rgb')}
    groups, seen, conflicts = {}, {}, set()
    rows = sorted(rows, key=lambda r: r['path'])
    for r in rows:
        safe_relative(r['path'])
        label = int(r['label'])
        if not 0 <= label < 1000 or r['path'].split('/')[0] != classes[label]:
            raise ValueError('class mapping mismatch')
        for key in ('sha', 'rgb'):
            identity = (key, r[key])
            if identity in seen and seen[identity] != label:
                conflicts.add(identity)
            seen[identity] = label
    used = {k: set() for k in ('path', 'sha', 'rgb')}
    for r in rows:
        if any(r[k] in forbidden[k] or (k, r[k]) in conflicts for k in used):
            continue
        if any(r[k] in used[k] for k in used):
            continue
        for k in used:
            used[k].add(r[k])
        groups.setdefault(int(r['label']), []).append(r)
    shortages = {classes[i]: len(groups.get(i, [])) for i in range(1000)
                 if len(groups.get(i, [])) < train_n+dev_n}
    if shortages:
        raise ValueError('insufficient nonreserved distinct TRAIN: '+json.dumps(shortages))
    selected = []
    for i in range(1000):
        pool = sorted(groups[i], key=lambda r: digest([seed, 'split', r['path']]))
        for j, r in enumerate(pool[:train_n+dev_n]):
            selected.append(dict(role='train' if j < train_n else 'dev',
                                 **{k: r[k] for k in ('path', 'label', 'sha', 'rgb')}))
    training = [r for r in selected if r['role'] == 'train']
    # One per class, then 24 additional identities: exactly 1024 TRAIN images.
    calibration = [next(r['path'] for r in training if int(r['label']) == i) for i in range(1000)]
    calibration_set = set(calibration)
    remaining = sorted((r for r in training if r['path'] not in calibration_set),
                       key=lambda r: digest([seed, 'calibration', r['path']]))
    calibration += [r['path'] for r in remaining[:24]]
    receipt = dict(seed=seed, class_mapping_sha256=digest(classes),
                   counts=dict(Counter(r['role'] for r in selected)),
                   minimum_eligible_class=min(map(len, groups.values())),
                   exclusion_counts={k: len(v) for k, v in forbidden.items()},
                   conflicting_identities=len(conflicts),
                   calibration_sha256=digest(calibration), calibration_count=len(calibration),
                   calibration_role='TRAIN subset shared by all arms and INT8',
                   pretrained_and_historical_TRAIN_exposure=True,
                   independent_test=False, perceptual_near_duplicates='not audited')
    return selected, calibration, receipt


def write_manifest(path, rows):
    p = Path(path)
    if p.exists():
        raise FileExistsError('manifest is immutable; inspect existing file')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['role','path','label','sha','rgb'], delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def read_manifest(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    for r in rows:
        safe_relative(r['path']); r['label'] = int(r['label'])
    return rows


class OriginalImages(Dataset):
    """Views depend only on seed/epoch/identity, never worker or prefetch order.

    Bytes and decoded RGB are checked while reading. No cached 64px surrogate.
    Official validation or V2 paths are not accepted by this training dataset.
    """
    def __init__(self, root, rows, role, seed=2026, epoch=0):
        if role not in ('train', 'dev', 'calibration'):
            raise ValueError('TRAIN-side roles only')
        self.root, self.rows = Path(root).resolve(), rows
        if self.root.name != 'train':
            raise ValueError('explicit original TRAIN directory required')
        self.role, self.seed, self.epoch = role, seed, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        path = self.root / safe_relative(r['path'])
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('image escapes TRAIN root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != r['sha']:
            raise ValueError('original image bytes changed')
        with Image.open(io.BytesIO(raw)) as source:
            im = source.convert('RGB')
        if hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest() != r['rgb']:
            raise ValueError('decoded RGB identity changed')
        if self.role == 'train':
            view_seed = int(digest([self.seed, self.epoch, r['path']])[:16], 16) % 2**63
            # Worker-local fork preserves unrelated RNG and restart determinism.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(view_seed)
                i,j,h,w = T.RandomResizedCrop.get_params(im, (.08,1.), (3/4,4/3))
                im = F.resized_crop(im, i,j,h,w, (160,160), T.InterpolationMode.BICUBIC, antialias=True)
                if torch.rand(()) < .5:
                    im = F.hflip(im)
        else:
            im = F.center_crop(F.resize(im, 182, T.InterpolationMode.BICUBIC, antialias=True), [160,160])
        return F.to_tensor(im), int(r['label'])


def epoch_order(n, seed, epoch):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed+1000003*epoch)).tolist()


def paired_accuracy_ci(candidate, reference, labels, seed=2026, repeats=10000):
    """Cluster bootstrap by image, averaging seeds before resampling images.

    Input predictions: [seeds, images] or [images]. Same image across seeds is
    one sampling unit. This interval does not estimate new-seed uncertainty.
    """
    a, b = np.atleast_2d(candidate), np.atleast_2d(reference)
    y = np.asarray(labels)
    if a.shape != b.shape or a.shape[1] != len(y) or len(y) < 2:
        raise ValueError('paired seeds and image order required')
    d = ((a == y).astype(float) - (b == y).astype(float)).mean(0) * 100
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats)
    for i in range(repeats):
        boot[i] = d[rng.integers(0, len(y), len(y))].mean()
    return dict(delta_pp=float(d.mean()), ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
                images=len(y), seeds=len(a), unit='paired image; seeds averaged within image',
                seed_uncertainty='report seed deltas separately; not covered by this CI')
