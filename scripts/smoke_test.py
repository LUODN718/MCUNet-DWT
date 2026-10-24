#!/usr/bin/env python3
"""CPU smoke test for all six ImageNet input groups."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import load_model
from src.wavelet import VARIANT_SUBBANDS, input_channels, input_resolution, select_subbands


def main():
    config = ROOT / "weights/mcunet-256kb-1mb_imagenet.json"
    weights = ROOT / "weights/mcunet-256kb-1mb_imagenet.pth"
    for variant in VARIANT_SUBBANDS:
        model = load_model(config, weights, variant).eval()
        rgb = torch.rand(1, 3, 160, 160)
        x = select_subbands(rgb, variant)
        expected = (1, input_channels(variant), input_resolution(variant), input_resolution(variant))
        assert tuple(x.shape) == expected, (variant, x.shape, expected)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 1000), (variant, y.shape)
        print(f"[PASS] {variant:6s} input={tuple(x.shape)} output={tuple(y.shape)}")


if __name__ == "__main__":
    main()

