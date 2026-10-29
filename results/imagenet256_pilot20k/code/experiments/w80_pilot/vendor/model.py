"""Selective Haar, approximate separable stem and physically sliced MCUNet."""
import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

VARIANTS = ('ll', 'll_lh', 'll_hl', 'll_hh', 'rgb_e23')


def haar_input(x, variant):
    """Raw RGB [0,1] -> normalized requested bands; never compute unused bands.

    Orthonormal LL=(a+b+c+d)/2, normalized as LL-1; HF=(signed sum)/2,
    normalized as 2*HF. LH varies vertically; HL varies horizontally.
    LL is exactly normalized 2x2 average pooling (up to roundoff).
    """
    if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 2:
        raise ValueError('expected nonempty BCHW RGB')
    if variant not in VARIANTS:
        raise ValueError(variant)
    if any(s % 2 for s in x.shape[-2:]):
        raise ValueError('even spatial dimensions required; no padding')
    if variant == 'rgb_e23':
        return x * 2 - 1
    a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
    c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) * .5 - 1
    if variant == 'll':
        return ll
    if variant == 'll_lh':
        hf = a + b - c - d
    elif variant == 'll_hl':
        hf = a - b + c - d
    else:
        hf = a - b - c + d
    return torch.cat((ll, hf), 1)


class SeparableStem(nn.Module):
    def __init__(self, old, bands):
        super().__init__()
        conv = old.conv
        if (conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.groups, conv.bias) != (3, 16, (3, 3), (2, 2), 1, None):
            raise ValueError('requires the frozen MCUNet-in2 stem')
        self.dw = nn.Conv2d(3*bands, 3*bands, 3, 1, 1, groups=3*bands, bias=False)
        self.pw = nn.Conv2d(3*bands, 16, 1, bias=False)
        self.bn = copy.deepcopy(old.bn)
        self.act = nn.ReLU6(inplace=False)
        self.rank1_relative_error = []
        with torch.no_grad():
            self.pw.weight.zero_()
            for c in range(3):
                matrix = conv.weight[:, c].reshape(16, 9).double()
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                # Canonical sign to make initialization auditable across SVDs.
                sign = 1 if vh[0, vh[0].abs().argmax()] >= 0 else -1
                filt = vh[0] * sign
                column = u[:, 0] * s[0] * sign
                self.dw.weight[c, 0].copy_(filt.reshape(3, 3))
                self.pw.weight[:, c, 0, 0].copy_(column)
                if bands == 2:
                    self.dw.weight[c+3].copy_(self.dw.weight[c])
                error = (matrix - column[:, None]*filt).norm() / matrix.norm().clamp_min(1e-30)
                self.rank1_relative_error.append(float(error))

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class InputModel(nn.Module):
    def __init__(self, backbone, variant):
        super().__init__()
        self.backbone, self.variant = backbone, variant

    def forward(self, x):
        if tuple(x.shape[1:]) != (3, 160, 160):
            raise ValueError('original-image RGB crop must be 3x160x160')
        return self.backbone(haar_input(x, self.variant))


def load_pretrained(config, weights, mcunet_root):
    sys.path.insert(0, str(Path(mcunet_root).resolve()))
    from mcunet.tinynas.nn.networks import ProxylessNASNets
    cfg = json.loads(Path(config).read_text())
    if cfg['classifier']['out_features'] != 1000 or cfg['resolution'] != 160:
        raise ValueError('1000-class, 160-pixel source required')
    model = ProxylessNASNets.build_from_config(cfg)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('state_dict', state), strict=True)
    return model.eval()


def retained_width(n):
    # nearest multiple of 8 to 2n/3; half-up, not Python banker rounding
    return min(n, max(8, 8 * math.floor(n / 12 + .5)))


def block_audit(model):
    rows = []
    for i, block in enumerate(model.blocks):
        m = block.mobile_inverted_conv
        if not hasattr(m, 'point_linear'):
            raise ValueError(f'unsupported block {i}')
        n = m.point_linear.conv.in_channels
        rows.append(dict(block_zero_based=i, mbconv_one_based=i+1,
                         in_channels=m.in_channels, out_channels=m.out_channels,
                         kernel=m.kernel_size, stride=m.stride, original_expansion=n,
                         retained_expansion=n if i < 2 else retained_width(n),
                         residual=block.shortcut is not None,
                         has_expansion=m.inverted_bottleneck is not None))
    return rows


@torch.no_grad()
def calibration_scores(model, batches, expected=1024):
    """Once on uncompressed pretrained RGB: mean abs pre-projection activation
    times L1 projection-column norm. TRAIN only; caller verifies identities.
    """
    model.eval()
    sums, counts, hooks = {}, {}, []
    for i, block in enumerate(model.blocks):
        if i < 2:
            continue
        def collect(module, args, i=i):
            x = args[0].detach().double()
            sums[i] = sums.get(i, 0) + x.abs().sum((0, 2, 3)).cpu()
            counts[i] = counts.get(i, 0) + x.shape[0]*x.shape[2]*x.shape[3]
        hooks.append(block.mobile_inverted_conv.point_linear.conv.register_forward_pre_hook(collect))
    seen = 0
    try:
        device = next(model.parameters()).device
        for raw in batches:
            x = raw[0] if isinstance(raw, (tuple, list)) else raw
            seen += len(x)
            model(haar_input(x.to(device), 'rgb_e23'))
    finally:
        for hook in hooks:
            hook.remove()
    if seen != expected:
        raise ValueError(f'calibration count {seen}, expected {expected}')
    scores = {}
    for i in sums:
        w = model.blocks[i].mobile_inverted_conv.point_linear.conv.weight
        strength = w.detach().double().abs().sum((0, 2, 3)).cpu()
        scores[str(i)] = (sums[i] / counts[i] * strength).tolist()
    return scores


def select_indices(model, scores):
    result = {}
    for row in block_audit(model):
        i, n, k = row['block_zero_based'], row['original_expansion'], row['retained_expansion']
        if i < 2:
            result[str(i)] = list(range(n))
            continue
        s = scores[str(i)]
        if len(s) != n or not all(math.isfinite(v) for v in s):
            raise ValueError(f'invalid scores block {i}')
        # Tie-break by original channel ID; store ascending original order.
        result[str(i)] = sorted(sorted(range(n), key=lambda j: (-s[j], j))[:k])
    return result


def _slice_bn(old, idx):
    new = nn.BatchNorm2d(len(idx), eps=old.eps, momentum=old.momentum,
                         affine=old.affine, track_running_stats=old.track_running_stats)
    state = old.state_dict()
    new.load_state_dict({k: v.clone() if v.ndim == 0 else v[idx].clone() for k, v in state.items()})
    return new


def compress(model, indices):
    model = copy.deepcopy(model).cpu()
    audit = block_audit(model)
    if set(indices) != {str(r['block_zero_based']) for r in audit}:
        raise ValueError('indices must cover every actual MBConv exactly')
    for r in audit:
        i = r['block_zero_based']
        idx = indices[str(i)]
        n, k = r['original_expansion'], r['retained_expansion']
        if len(idx) != k or idx != sorted(set(idx)) or min(idx) < 0 or max(idx) >= n:
            raise ValueError(f'invalid indices block {i}')
        if i < 2:
            if idx != list(range(n)):
                raise ValueError('first two MBConv must remain intact')
            continue
        m = model.blocks[i].mobile_inverted_conv
        if m.inverted_bottleneck is None or m.use_se:
            raise ValueError('unsupported compressed block')
        exp, dw, proj = m.inverted_bottleneck, m.depth_conv, m.point_linear
        exp_new = nn.Conv2d(m.in_channels, k, 1, bias=False)
        dw_new = nn.Conv2d(k, k, dw.conv.kernel_size, dw.conv.stride,
                           dw.conv.padding, dilation=dw.conv.dilation, groups=k, bias=False)
        proj_new = nn.Conv2d(k, m.out_channels, 1, bias=False)
        with torch.no_grad():
            exp_new.weight.copy_(exp.conv.weight[idx])
            dw_new.weight.copy_(dw.conv.weight[idx])
            proj_new.weight.copy_(proj.conv.weight[:, idx])
        exp.conv, exp.bn = exp_new, _slice_bn(exp.bn, idx)
        dw.conv, dw.bn = dw_new, _slice_bn(dw.bn, idx)
        proj.conv = proj_new
        m.mid_channels = k
    return model.train(model.training)


def make_variant(pretrained, indices, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    model = compress(pretrained, indices)
    if variant != 'rgb_e23':
        model.first_conv = SeparableStem(model.first_conv, 1 if variant == 'll' else 2)
    return InputModel(model, variant).train(pretrained.training)


@torch.no_grad()
def cost_audit(model):
    """Exact Conv/Linear MACs and separate arithmetic; no latency inference."""
    rows, hooks, other = [], [], {'bn_elements_unfused': 0, 'relu6_elements': 0,
                                'residual_adds': 0, 'global_pool_adds': 0,
                                'global_pool_scales': 0, 'bias_adds': 0}
    def capture(name):
        def fn(m, args, y):
            mac = y.numel() * (m.in_channels//m.groups * math.prod(m.kernel_size)
                              if isinstance(m, nn.Conv2d) else m.in_features)
            if m.bias is not None:
                other['bias_adds'] += y.numel()
            rows.append(dict(module=name, input=list(args[0].shape), output=list(y.shape),
                             macs=mac, output_bytes_fp32=y.numel()*4))
        return fn
    def residual(m, args, y):
        if m.shortcut is not None:
            other['residual_adds'] += y.numel()
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(capture(name)))
        if isinstance(m, (nn.BatchNorm2d, nn.ReLU6)):
            key = 'bn_elements_unfused' if isinstance(m, nn.BatchNorm2d) else 'relu6_elements'
            def count(m, args, y, key=key):
                other[key] += y.numel()
            hooks.append(m.register_forward_hook(count))
        if hasattr(m, 'mobile_inverted_conv'):
            hooks.append(m.register_forward_hook(residual))
    backbone = model.backbone if isinstance(model, InputModel) else model
    def gap(m, args, y):
        other['global_pool_adds'] = y.shape[1] * (y.shape[2]*y.shape[3]-1)
        # Upstream executes mean(width), then mean(height), not one fused GAP.
        other['global_pool_scales'] = y.shape[1] * (y.shape[2]+1)
    hooks.append(backbone.blocks[-1].register_forward_hook(gap))
    training = model.training
    try:
        model.eval()(torch.zeros(1, 3, 160, 160))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(training)
    variant = getattr(model, 'variant', 'rgb_e23')
    dual = variant not in ('rgb_e23', 'll')
    elements = 3*80*80
    preprocessing = (dict(add_sub=3*160*160, multiplies=3*160*160)
                     if variant == 'rgb_e23' else
                     dict(add_sub=elements*(4 + 3*int(dual)), multiplies=elements))
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes_fp32=sum(p.numel()*p.element_size() for p in model.parameters()),
                conv_linear_macs=sum(r['macs'] for r in rows), layers=rows,
                preprocessing_arithmetic=preprocessing, extra_backbone_arithmetic=other,
                rgb_input_bytes_fp32=3*160*160*4,
                normalized_stem_input_bytes_fp32=(3*160*160 if variant=='rgb_e23' else elements*(2 if dual else 1))*4,
                concatenation_copied_elements=elements*2 if dual else 0,
                largest_single_conv_linear_output_bytes_fp32=max(r['output_bytes_fp32'] for r in rows),
                peak_live_activation_bytes=None, operator_workspace_bytes=None,
                memory_scope='single tensor sizes are not peak live memory; backend arena/workspace unmeasured',
                resize_decode_cost='backend/input dependent, unmeasured; not included in MACs',
                int8_accuracy=None, int8_memory=None, renode_instructions=None)
