"""Deterministic per-batch RGB/center-padded-LL mixtures."""
import hashlib
import torch
import torch.nn.functional as F

RATIOS={
    'rgb50_ll50':(1,2),
    'rgb33_ll67':(1,3),
    'rgb25_ll75':(1,4),
}


def corrected_ll(x):
    if x.ndim!=4 or tuple(x.shape[1:])!=(3,160,160):
        raise ValueError('expected RGB160 BCHW in [0,1]')
    a,b=x[...,::2,::2],x[...,::2,1::2]
    c,d=x[...,1::2,::2],x[...,1::2,1::2]
    return (a+b+c+d)*.25


def padded_center(x):
    return F.pad(corrected_ll(x),(40,40,40,40),value=0.)


def rgb_count(step,variant,batch=32):
    if variant not in RATIOS or batch!=32:
        raise ValueError('unsupported variant or batch')
    numerator,denominator=RATIOS[variant]
    # 1:1 and 1:3 are exact per batch. For1:2, cycle11,11,10 RGB
    # samples to achieve exactly one-third RGB over the full20-epoch run.
    if denominator==3:
        return (11,11,10)[step%3]
    return batch*numerator//denominator


def batch_mask(step,variant,batch=32):
    count=rgb_count(step,variant,batch)
    seed=int(hashlib.sha256(f'center-ratio-v1:2026:{variant}:{step}'.encode()).hexdigest()[:16],16)%2**63
    indices=torch.randperm(batch,generator=torch.Generator().manual_seed(seed))
    mask=torch.zeros(batch,dtype=torch.bool)
    mask[indices[:count]]=True
    return mask


class RatioMix160(torch.nn.Module):
    def __init__(self,backbone,variant):
        super().__init__()
        if variant not in RATIOS:
            raise ValueError(variant)
        self.backbone=backbone
        self.variant=variant
        self.committed_step=0
        self.eval_view='ll1_pad'
        self.last_logits=None

    def set_train_step(self,step):
        if not isinstance(step,int) or not 0<=step<80000:
            raise ValueError('invalid committed update')
        self.committed_step=step

    def forward(self,x):
        if self.training:
            mask=batch_mask(self.committed_step,self.variant,len(x)).to(x.device)
            image=torch.where(mask[:,None,None,None],x,padded_center(x))
        elif self.eval_view=='rgb':
            image=x
        elif self.eval_view=='ll1_pad':
            image=padded_center(x)
        else:
            raise ValueError(self.eval_view)
        logits=self.backbone(image*2-1)
        self.last_logits=logits.detach()
        return logits
