"""Haar approximation coefficients, amplitude correction, then bilinear resize."""
import torch
import torch.nn.functional as F


def lowpass_view(x, level):
    if level not in (1, 2, 3):
        raise ValueError('level must be 1, 2, or 3')
    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 160, 160):
        raise ValueError('expected a batch of RGB 160x160 views in [0,1]')
    ll = x
    for _ in range(level):
        a, b = ll[..., ::2, ::2], ll[..., ::2, 1::2]
        c, d = ll[..., 1::2, ::2], ll[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
    # Orthonormal Haar LL has DC gain 2 per level. Correct before 2*x-1.
    ll = ll / (2 ** level)
    return F.interpolate(ll, size=(160, 160), mode='bilinear', align_corners=False)


class LL160(torch.nn.Module):
    def __init__(self, backbone, level):
        super().__init__()
        self.backbone = backbone
        self.level = level
        self.last_logits = None

    def forward(self, x):
        logits = self.backbone(lowpass_view(x, self.level) * 2.0 - 1.0)
        self.last_logits = logits.detach()
        return logits
