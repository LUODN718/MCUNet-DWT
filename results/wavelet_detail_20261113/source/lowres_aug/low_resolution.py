"""True 80x80 backbone input; no upsampling and no changed backbone stride."""
import torch
import torch.nn.functional as F


def low_resolution_view(x, variant):
    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 160, 160):
        raise ValueError('expected RGB160 float BCHW views in [0,1]')
    if variant == 'll80':
        a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
        c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
        # Haar LL / 2 corrects its DC gain. Following 2*x-1 is exactly
        # (a+b+c+d)*0.5-1, matching the historical direct-LL80 experiment.
        return (a + b + c + d) * 0.25
    if variant == 'rgb80':
        # Tensor bicubic can overshoot; explicitly retain the RGB [0,1] range.
        return F.interpolate(x, size=(80, 80), mode='bicubic',
                             align_corners=False, antialias=True).clamp(0, 1)
    raise ValueError('variant must be rgb80 or ll80')


class LowResolution80(torch.nn.Module):
    def __init__(self, backbone, variant):
        super().__init__()
        if variant not in ('rgb80', 'll80'):
            raise ValueError(variant)
        self.backbone, self.variant = backbone, variant
        self.last_logits = None

    def forward(self, x):
        logits = self.backbone(low_resolution_view(x, self.variant) * 2.0 - 1.0)
        self.last_logits = logits.detach()
        return logits
