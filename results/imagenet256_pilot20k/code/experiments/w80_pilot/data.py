"""Small, explicitly resized-image pilot; not original ImageNet identity admission."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .vendor.data import digest, sha


def prepare(source_root, old_membership, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_membership_sha256'] != sha(old_membership):
            raise ValueError('source membership changed')
        return manifest
    with Path(old_membership).open(encoding='utf-8', newline='') as stream:
        pool = list(csv.DictReader(stream))
    by_class = {}
    for row in pool:
        if row['split'] == 'train':
            by_class.setdefault(int(row['target']), []).append(row['image_id'])
    if sorted(by_class) != list(range(1000)):
        raise ValueError('expected all 1000 classes')
    selected, training = [], []
    byte_seen, rgb_seen = set(), set()
    for label in range(1000):
        paths = sorted(by_class[label], key=lambda path: digest([2026, 'pilot20k', path]))
        if len(paths) < 20:
            raise ValueError('insufficient images for pilot')
        accepted = 0
        for relative in paths:
            raw = (Path(source_root) / relative).read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                image = im.convert('RGB')
                if image.size != (256, 256):
                    raise ValueError('pilot requires the declared 256px mirror')
            byte_sha = hashlib.sha256(raw).hexdigest()
            rgb_sha = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if byte_sha in byte_seen or rgb_sha in rgb_seen:
                continue
            byte_seen.add(byte_sha); rgb_seen.add(rgb_sha)
            row = dict(path=relative, label=label, role='train' if accepted < 16 else 'dev',
                       mirror_byte_sha256=byte_sha, mirror_rgb_sha256=rgb_sha)
            selected.append(row)
            if accepted < 16:
                training.append(row)
            accepted += 1
            if accepted == 20:
                break
        if accepted != 20:
            raise ValueError('insufficient distinct pilot images')
    calibration = [next(r['path'] for r in training if r['label'] == i) for i in range(1000)]
    first = set(calibration)
    rest = sorted((r['path'] for r in training if r['path'] not in first),
                  key=lambda path: digest([2026, 'calibration', path]))
    calibration.extend(rest[:24])
    if len(set(calibration)) != 1024:
        raise ValueError('invalid calibration cohort')
    manifest = dict(scope='ImageNet256 mirror 20k pilot; not original TRAIN W80 protocol',
                    seed=2026, train_per_class=16, dev_per_class=4, classes=1000,
                    train_images=16000, dev_images=4000, source_membership_sha256=sha(old_membership),
                    old_baseline_dev_excluded=True, original_identity_audit=False,
                    original_pretraining_and_historical_exposure=True,
                    rows=selected, calibration=calibration, calibration_sha256=digest(calibration))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (output / 'samples.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(selected)
    return manifest


class MirrorImages(Dataset):
    def __init__(self, root, rows, role, epoch=0):
        self.root, self.rows, self.role, self.epoch = Path(root).resolve(), rows, role, epoch
        if role not in ('train', 'dev', 'calibration'):
            raise ValueError(role)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('image escapes root')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['mirror_byte_sha256']:
            raise ValueError('mirror byte identity changed')
        with Image.open(io.BytesIO(raw)) as source:
            im = source.convert('RGB')
        if hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest() != row['mirror_rgb_sha256']:
            raise ValueError('mirror RGB identity changed')
        if self.role == 'train':
            view_seed = int(digest([2026, self.epoch, row['path']])[:16], 16) % 2**63
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(view_seed)
                i, j, h, w = T.RandomResizedCrop.get_params(im, (.08, 1.), (3/4, 4/3))
                im = F.resized_crop(im, i, j, h, w, (160, 160), T.InterpolationMode.BICUBIC, antialias=True)
                if torch.rand(()) < .5:
                    im = F.hflip(im)
        else:
            im = F.center_crop(F.resize(im, 182, T.InterpolationMode.BICUBIC, antialias=True), [160, 160])
        return F.to_tensor(im), int(row['label'])
