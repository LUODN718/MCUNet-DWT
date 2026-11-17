"""Paired RGB/LL domains within every batch; same pretrained MCUNet and shared BN."""
import hashlib
import torch
import torch.nn.functional as F


def corrected_ll(x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (3,160,160):
        raise ValueError('expected RGB160 BCHW in [0,1]')
    a,b=x[...,::2,::2],x[...,::2,1::2]
    c,d=x[...,1::2,::2],x[...,1::2,1::2]
    return (a+b+c+d)*.25


def padded_view(x):
    return F.pad(corrected_ll(x),(0,80,0,80),value=0.)


def batch_mask(step, batch=32):
    if batch != 32:
        raise ValueError('formal mixture requires exactly32 samples')
    seed=int(hashlib.sha256(f'within-batch-pad-v1:2026:{step}'.encode()).hexdigest()[:16],16)%2**63
    indices=torch.randperm(batch,generator=torch.Generator().manual_seed(seed))
    mask=torch.zeros(batch,dtype=torch.bool)
    mask[indices[:16]]=True
    return mask


class WithinBatch160(torch.nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone=backbone
        self.committed_step=0
        self.eval_view='ll1_pad'
        self.last_logits=None

    def set_train_step(self,step):
        if not isinstance(step,int) or not 0<=step<80000:
            raise ValueError('invalid committed update')
        self.committed_step=step

    def forward(self,x):
        if self.training:
            mask=batch_mask(self.committed_step,len(x)).to(x.device)
            image=torch.where(mask[:,None,None,None],padded_view(x),x)
        elif self.eval_view=='rgb':
            image=x
        elif self.eval_view=='ll1_pad':
            image=padded_view(x)
        else:
            raise ValueError(self.eval_view)
        logits=self.backbone(image*2-1)
        self.last_logits=logits.detach()
        return logits
