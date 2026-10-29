"""Matched-data comparison of spatial stride2 and channel-compressed W80."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .vendor import model as w80, runtime

ROUTES = ('spatial_s2', 'w80_e23')
VARIANTS = ('ll', 'll_lh', 'll_hl', 'll_hh', 'rgb80')


class RGB80(w80.InputModel):
    def __init__(self, backbone):
        super().__init__(backbone, 'rgb80')

    def forward(self, x):
        if tuple(x.shape[1:]) != (3, 160, 160):
            raise ValueError('expected RGB160 before preprocessing')
        x = F.interpolate(x, size=(80, 80), mode='bilinear', align_corners=False, antialias=True)
        return self.backbone(x * 2 - 1)


def build(base, indices, route, variant):
    if route not in ROUTES or variant not in VARIANTS:
        raise ValueError((route, variant))
    if route == 'w80_e23':
        if variant == 'rgb80':
            raise ValueError('RGB80 belongs only to the spatial route')
        return w80.make_variant(base, indices, variant)
    backbone = copy.deepcopy(base)
    if variant == 'rgb80':
        return RGB80(backbone)
    old = backbone.first_conv.conv
    channels = 3 if variant == 'll' else 6
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, stride=2,
                    padding=old.padding, dilation=old.dilation, groups=1, bias=False)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
    backbone.first_conv.conv = new
    backbone.first_conv.in_channels = channels
    return w80.InputModel(backbone, variant)


def make_parts(base, indices, route, variant):
    model = build(base, indices, route, variant).cuda().train()
    if variant == 'rgb80':
        groups = [{'params': list(model.parameters()), 'lr': .001}]
    else:
        stem = list(model.backbone.first_conv.parameters())
        ids = {id(p) for p in stem}
        groups = [{'params': [p for p in model.parameters() if id(p) not in ids], 'lr': .001},
                  {'params': stem, 'lr': .005}]
    optimizer = torch.optim.SGD(groups, momentum=.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: runtime.lr_factor(step, 500))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, scaler
