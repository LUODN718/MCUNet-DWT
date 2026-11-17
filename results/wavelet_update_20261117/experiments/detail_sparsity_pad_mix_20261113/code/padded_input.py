"""Haar approximation on a fixed black canvas; original full MCUNet is shared."""
import hashlib
from functools import lru_cache

import torch
import torch.nn.functional as F


def corrected_ll(x, level=1):
    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 160, 160):
        raise ValueError('expected raw RGB160 float BCHW in [0,1]')
    if level not in (1, 2, 3):
        raise ValueError('only LL1/2/3 are defined')
    ll = x
    for _ in range(level):
        a, b = ll[..., ::2, ::2], ll[..., ::2, 1::2]
        c, d = ll[..., 1::2, ::2], ll[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
    return ll / (2 ** level)


def padded_view(x, level=1):
    """Top-left coefficient block; zero in [0,1] domain, before normalization."""
    ll = corrected_ll(x, level)
    h, w = ll.shape[-2:]
    return F.pad(ll, (0, 160-w, 0, 160-h), mode='constant', value=0.0)


@lru_cache(maxsize=2)
def epoch_views(epoch):
    if not isinstance(epoch, int) or not 0 <= epoch < 20:
        raise ValueError('epoch must be zero-based, 0..19')
    seed = int(hashlib.sha256(f'pad-mix-v1:2026:{epoch}'.encode()).hexdigest()[:16], 16) % 2**63
    choices = torch.cat((torch.zeros(2000, dtype=torch.int64),
                         torch.ones(2000, dtype=torch.int64)))
    order = torch.randperm(4000, generator=torch.Generator().manual_seed(seed))
    return tuple(int(i) for i in choices[order].tolist())


def view_for_step(variant, step):
    if variant == 'rgb_only':
        return 'rgb'
    if variant == 'll1_pad_only':
        return 'll1_pad'
    if variant != 'rgb_ll1_pad_mix':
        raise ValueError(variant)
    if not isinstance(step, int) or not 0 <= step < 80000:
        raise ValueError('training step must be the committed zero-based cursor')
    epoch, batch_index = divmod(step, 4000)
    return 'll1_pad' if epoch_views(epoch)[batch_index] else 'rgb'


class PaddedMix160(torch.nn.Module):
    def __init__(self, backbone, variant):
        super().__init__()
        if variant not in ('rgb_only', 'll1_pad_only', 'rgb_ll1_pad_mix'):
            raise ValueError(variant)
        self.backbone, self.variant = backbone, variant
        self.committed_step = 0
        self.eval_view = 'll1_pad'
        self.last_logits = None

    def set_train_step(self, step):
        view_for_step(self.variant, step)
        self.committed_step = step

    def forward(self, x):
        view = view_for_step(self.variant, self.committed_step) if self.training else self.eval_view
        if view == 'rgb':
            image = x
        elif view == 'll1_pad':
            image = padded_view(x, level=1)
        else:
            raise ValueError(view)
        logits = self.backbone(image * 2.0 - 1.0)
        self.last_logits = logits.detach()
        return logits
