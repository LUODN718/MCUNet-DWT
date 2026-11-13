"""Full Haar input and one dense stride2 convolution, with split optimizer weights.

The convolution is a single 12->16 Conv2d. Its weight is concatenated from two
leaf parameters solely to keep inherited LL columns at LR .001 and new HF
columns at LR .005. Export can remove the parametrization without changing the
convolution. No depthwise factorization, channel pruning, or teacher is used.
"""
import copy

import torch
from torch import nn
from torch.nn.utils import parametrize


VARIANT = 'dwt_all_stride2_full'
BAND_ORDER = ('LL_RGB', 'LH_RGB', 'HL_RGB', 'HH_RGB')


def haar_input(x):
    """RGB [0,1] -> [LL-1, 2*LH, 2*HL, 2*HH], in band-major RGB order.

    Orthonormal Haar divides each signed sum by two. LH varies vertically;
    HL varies horizontally, matching the frozen two-band implementations.
    """
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 2:
        raise ValueError('expected nonempty BCHW RGB')
    if any(s % 2 for s in x.shape[-2:]):
        raise ValueError('even spatial dimensions required; no padding')
    a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
    c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) * .5 - 1
    lh = a + b - c - d
    hl = a - b + c - d
    hh = a - b - c + d
    return torch.cat((ll, lh, hl, hh), dim=1)


def inverse_haar(normalized):
    """Undo our normalization and Haar for a structure/information check only."""
    if normalized.ndim != 4 or normalized.shape[1] != 12:
        raise ValueError('expected BCHW with twelve band-major channels')
    ll, lh, hl, hh = normalized.split(3, dim=1)
    ll = (ll + 1) * .5
    a = ll + (lh + hl + hh) * .25
    b = ll + (lh - hl - hh) * .25
    c = ll + (-lh + hl - hh) * .25
    d = ll + (-lh - hl + hh) * .25
    out = normalized.new_empty(normalized.shape[0], 3,
                               normalized.shape[2] * 2, normalized.shape[3] * 2)
    out[..., ::2, ::2], out[..., ::2, 1::2] = a, b
    out[..., 1::2, ::2], out[..., 1::2, 1::2] = c, d
    return out


class BandWeights(nn.Module):
    def forward(self, inherited_ll, new_hf):
        return torch.cat((inherited_ll, new_hf), dim=1)

    def right_inverse(self, dense_weight):
        return dense_weight[:, :3].clone(), dense_weight[:, 3:].clone()


def stem_parameters(model):
    weights = model.backbone.first_conv.conv.parametrizations.weight
    return weights.original0, weights.original1


def build(pretrained, model_api):
    backbone = copy.deepcopy(pretrained)
    old = backbone.first_conv.conv
    if (old.in_channels, old.out_channels, old.kernel_size, old.stride,
            old.groups, old.bias) != (3, 16, (3, 3), (2, 2), 1, None):
        raise ValueError('requires original MCUNet-in2 dense stride2 stem')
    # Every column is overwritten; avoid consuming RNG for discarded initialization.
    with torch.random.fork_rng(devices=[]):
        conv = nn.Conv2d(12, 16, old.kernel_size, stride=old.stride,
                         padding=old.padding, dilation=old.dilation, groups=1,
                         bias=False, padding_mode=old.padding_mode)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[:, :3].copy_(old.weight)
    parametrize.register_parametrization(conv, 'weight', BandWeights())
    backbone.first_conv.conv = conv
    backbone.first_conv.in_channels = 12

    class FourBandInputModel(model_api.InputModel):
        def __init__(self):
            super().__init__(backbone, VARIANT)
            self.last_logits = None

        def forward(self, x):
            if tuple(x.shape[1:]) != (3, 160, 160):
                raise ValueError('original-image RGB crop must be 3x160x160')
            logits = self.backbone(haar_input(x))
            self.last_logits = logits.detach()
            return logits

    return FourBandInputModel().train(pretrained.training)


def check_initialization(model, pretrained):
    actual, reference = model.backbone.state_dict(), pretrained.state_dict()
    for key, value in reference.items():
        if key == 'first_conv.conv.weight':
            continue
        if key not in actual or not torch.equal(actual[key].cpu(), value.cpu()):
            raise AssertionError('inherited tensor changed: ' + key)
    ll, hf = stem_parameters(model)
    if not torch.equal(ll.detach().cpu(), reference['first_conv.conv.weight'].cpu()):
        raise AssertionError('LL convolution columns are not the original pretrained weight')
    if torch.count_nonzero(hf):
        raise AssertionError('new high-frequency columns must start at zero')
    return dict(all_nonstem_tensors_preserved=True, inherited_ll_columns_exact=True,
                new_hf_columns_zero=True, inherited_ll_parameters=ll.numel(),
                new_hf_parameters=hf.numel())


def cost_audit(model, model_api):
    cost = model_api.cost_audit(model)
    # The frozen helper understands at most two bands. Replace ONLY its input
    # preprocessing accounting; Conv/Linear hooks already observe the true model.
    n = 3 * 80 * 80
    cost.update(preprocessing_arithmetic=dict(add_sub=13*n, multiplies=n),
                preprocessing_scope='direct formulas: LL 4 add/sub + 1 multiply; each HF 3 add/sub',
                normalized_stem_input_bytes_fp32=12*80*80*4,
                concatenation_copied_elements=12*80*80,
                band_order=list(BAND_ORDER),
                weight_concatenation_copied_elements_per_forward=16*12*3*3,
                weight_concatenation_scope='optimizer parametrization only; removable for inference')
    return cost
