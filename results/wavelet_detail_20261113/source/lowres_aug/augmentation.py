"""Sample-addressed mild augmentation, shared by RGB80 and LL80.

Applied after the original, unchanged crop/flip and before either downsampler.
No extra training images, updates, label mixing or evaluation randomness.
"""
import hashlib
import json

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


SETTINGS = dict(
    application='TRAIN only, after fixed RGB160 crop/flip, before downsampling',
    color_jitter=dict(probability=0.8, brightness=0.15, contrast=0.15,
                      saturation=0.15, hue=0.02),
    random_erasing=dict(probability=0.1, scale=[0.02, 0.08],
                        ratio=[0.3, 3.3], value_rgb=0.5),
    rng='SHA256([lowres-extra-augmentation-v1,seed,epoch,image_path]); local CPU fork',
)
_JITTER = T.ColorJitter(brightness=0.15, contrast=0.15,
                      saturation=0.15, hue=0.02)
_ERASE = T.RandomErasing(p=0.1, scale=(0.02, 0.08),
                        ratio=(0.3, 3.3), value=0.5, inplace=False)


def augment_rgb_view(image, seed, epoch, path, augmentation):
    if augmentation == 'plain':
        return F.to_tensor(image)
    if augmentation != 'light':
        raise ValueError(augmentation)
    identity = json.dumps(['lowres-extra-augmentation-v1', int(seed),
                           int(epoch), str(path)], separators=(',', ':'))
    view_seed = int(hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16], 16) % 2**63
    # Never consume global RNG or depend on loader worker count / resume cursor.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(view_seed)
        if torch.rand(()) < 0.8:
            image = _JITTER(image)
        return _ERASE(F.to_tensor(image))
