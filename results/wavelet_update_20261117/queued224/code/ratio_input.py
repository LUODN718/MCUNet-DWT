"""224px RGB and centered112 LL; batch-level1:2 sample assignment."""
import hashlib
import torch
import torch.nn.functional as F


def corrected_ll(x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (3,224,224):
        raise ValueError('expected RGB224 BCHW in[0,1]')
    a,b=x[...,::2,::2],x[...,::2,1::2]
    c,d=x[...,1::2,::2],x[...,1::2,1::2]
    return (a+b+c+d)*.25


def padded_center(x):
    return F.pad(corrected_ll(x),(56,56,56,56),value=0.)


def rgb_count(step,variant,batch=64):
    if batch not in (32,64):
        raise ValueError('unsupported batch')
    if variant == 'rgb_only':
        return batch
    if variant != 'rgb33_ll67':
        raise ValueError(variant)
    return ((11,11,10) if batch == 32 else (21,21,22))[step%3]


def batch_mask(step,variant,batch=64):
    count=rgb_count(step,variant,batch)
    seed=int(hashlib.sha256(f'center-ratio-v1:2026:{variant}:{step}'.encode()).hexdigest()[:16],16)%2**63
    indices=torch.randperm(batch,generator=torch.Generator().manual_seed(seed))
    mask=torch.zeros(batch,dtype=torch.bool)
    mask[indices[:count]]=True
    return mask


class RatioMix224(torch.nn.Module):
    def __init__(self,backbone,variant,batch=64):
        super().__init__()
        rgb_count(0,variant,batch)
        self.backbone,self.variant,self.batch=backbone,variant,batch
        self.committed_step=0
        self.eval_view='ll1_pad'
        self.last_logits=None

    def set_train_step(self,step):
        if not isinstance(step,int) or not 0<=step<20*(128000//self.batch):
            raise ValueError('invalid update')
        self.committed_step=step

    def forward(self,x):
        if self.training:
            if len(x)!=self.batch:
                raise ValueError('incorrect BN batch')
            if self.variant=='rgb_only':
                image=x
            else:
                mask=batch_mask(self.committed_step,self.variant,self.batch).to(x.device)
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
