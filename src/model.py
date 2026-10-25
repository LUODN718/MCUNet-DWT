"""MCUNet-in2 construction and first-layer adaptation for Haar inputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


def _import_mcunet():
    root = Path(os.environ.get("MCUNET_ROOT", Path(__file__).resolve().parents[2] / "mcunet"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mcunet.tinynas.nn.networks import ProxylessNASNets
    from mcunet.tinynas.nn.modules import ConvLayer
    return ProxylessNASNets, ConvLayer


def load_model(config_path: str | Path, weight_path: str | Path, variant: str,
               num_classes: int | None = None, classifier_seed: int = 2026):
    ProxylessNASNets, ConvLayer = _import_mcunet()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = ProxylessNASNets.build_from_config(config)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"unexpected pretrained checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if num_classes is not None and num_classes != model.classifier.out_features:
        classifier_config = dict(model.classifier.config)
        classifier_config.pop("name")
        classifier_config["out_features"] = num_classes
        # Keep the new head identical across variants despite different stem sizes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(classifier_seed)
            model.classifier = type(model.classifier).build_from_config(classifier_config)
    if variant != "rgb":
        subband_counts = {"ll": 1, "ll_lh": 2, "ll_hl": 2, "ll_hh": 2, "all": 4}
        model = adapt_first_conv(model, 3 * subband_counts[variant], ConvLayer)
    return model


def adapt_first_conv(model, in_channels: int, conv_layer_cls):
    old = model.first_conv
    old_conv = old.conv
    new = conv_layer_cls(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=1,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias,
        has_shuffle=old.has_shuffle,
        use_bn=old.use_bn,
        act_func=old.act_func,
        dropout_rate=old.dropout_rate,
        ops_order=old.ops_order,
    )
    with torch.no_grad():
        new.conv.weight.zero_()
        # The first 3 channels always correspond to LL/RGB. Every wavelet
        # variant therefore starts from exactly the same LL initialization.
        new.conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
        if getattr(old_conv, "bias", None) is not None:
            new.conv.bias.zero_()
        if hasattr(old, "bn") and old.bn is not None and hasattr(new, "bn"):
            new.bn.load_state_dict(old.bn.state_dict())
    model.first_conv = new
    return model

