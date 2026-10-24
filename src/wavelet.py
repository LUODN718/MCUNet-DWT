"""One-level orthonormal 2-D Haar transform used by the ImageNet baseline."""

from __future__ import annotations

from typing import Iterable

import torch


SUBBANDS = ("ll", "lh", "hl", "hh")
VARIANT_SUBBANDS = {
    "rgb": (),
    "ll": ("ll",),
    "ll_lh": ("ll", "lh"),
    "ll_hl": ("ll", "hl"),
    "ll_hh": ("ll", "hh"),
    "all": SUBBANDS,
}


def haar_1level(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return LL/LH/HL/HH for ``x`` with shape ``[B,C,H,W]``.

    The transform is orthonormal: every coefficient uses a factor of 0.5.
    Inputs are expected to have even spatial dimensions; an explicit error is
    preferable to silently changing the 160x160 experiment definition.
    """
    if x.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    h, w = x.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError(f"Haar input must be even, got {h}x{w}")

    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    return {
        "ll": 0.5 * (x00 + x01 + x10 + x11),
        "lh": 0.5 * (x00 + x01 - x10 - x11),
        "hl": 0.5 * (x00 - x01 + x10 - x11),
        "hh": 0.5 * (x00 - x01 - x10 + x11),
    }


def select_subbands(x: torch.Tensor, variant: str) -> torch.Tensor:
    """Apply Haar and concatenate the requested subbands in RGB-channel order."""
    if variant == "rgb":
        return x
    if variant not in VARIANT_SUBBANDS:
        raise ValueError(f"unknown variant {variant!r}")
    bands = haar_1level(x)
    # LL/RGB, LH/RGB, HL/RGB, HH/RGB. Signs are retained.
    return torch.cat([bands[name] for name in VARIANT_SUBBANDS[variant]], dim=1)


def input_channels(variant: str) -> int:
    if variant not in VARIANT_SUBBANDS:
        raise ValueError(f"unknown variant {variant!r}")
    return 3 * len(VARIANT_SUBBANDS[variant]) if variant != "rgb" else 3


def input_resolution(variant: str, rgb_resolution: int = 160) -> int:
    return rgb_resolution if variant == "rgb" else rgb_resolution // 2

