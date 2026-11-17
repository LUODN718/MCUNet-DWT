"""Verify modified mixture, actual model cost and exact transactional recovery."""
import copy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
import train_within_batch as base
import w80_tar_source
from within_batch_input import WithinBatch160, batch_mask, corrected_ll, padded_view

ROOT = Path(__file__).resolve().parents[1]


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    start=time.time()
    config=base.read_json(ROOT/'queue_config.json')
    for name,digest in config['sources'].items():
        assert base.sha256(ROOT/name)==digest,name
    args=['--variant','rgb_ll1_center_within_batch']
    for arg in ('handoff_root','train_tar','seal_dir','weights','mcunet_root'):
        args+=['--'+arg.replace('_','-'),config[arg]]
    args+=['--run-dir',str(ROOT/'runs/rgb_ll1_center_within_batch'),
           '--accept-local-content-sha256',config['manifest_sha256']]
    modules=base.load_protocol_modules(config['handoff_root'])
    inputs=base.verify_inputs(base.parse_args(args),modules)
    assert (len(inputs['train_rows']),len(inputs['dev_rows']),len(inputs['classes']))==(128000,20000,1000)
    checks=['fixed128k/20k cohort, accepted local content seal, official pretrained and pinned upstream']
    torch.set_num_threads(2)
    x=torch.rand(32,3,160,160,generator=torch.Generator().manual_seed(2026))
    torch.testing.assert_close(corrected_ll(x),torch.nn.functional.avg_pool2d(x,2),atol=2e-7,rtol=1e-6)
    ll=padded_view(x)
    assert ll.shape==x.shape
    torch.testing.assert_close(ll[...,40:120,40:120],corrected_ll(x))
    outside=torch.ones((160,160),dtype=torch.bool)
    outside[40:120,40:120]=False
    assert torch.count_nonzero(ll[...,outside])==0
    assert torch.all((ll*2-1)[...,outside]==-1)
    assert outside.sum()==19200
    checks.append('Haar LL unchanged; exact center80 block and40pixel zero border; pad normalizes to-1')
    schedule=hashlib.sha256()
    before=torch.get_rng_state().clone()
    for step in range(80000):
        mask=batch_mask(step)
        assert mask.shape==(32,) and int(mask.sum())==16
        schedule.update(mask.numpy().tobytes())
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(batch_mask(79999),batch_mask(79999))
    assert not torch.equal(batch_mask(0),batch_mask(1))
    checks.append('all80000 steps exactly16RGB+16LL with local RNG; resumable deterministic masks')
    # Verify actual decoded geometric views and cursor replay, not just tensors.
    order=modules['data'].epoch_order(128000,2026,0)[:64]
    it=iter(w80_tar_source.loader(inputs['loader_local'],inputs['train_rows'],'train',
                                epoch=0,order=order,workers=0))
    first=next(it)
    second=next(it)
    replay=next(iter(w80_tar_source.loader(inputs['loader_local'],inputs['train_rows'],'train',
                    epoch=0,order=order,cursor=32,workers=0)))
    assert torch.equal(second[0],replay[0]) and torch.equal(second[1],replay[1])
    del it,second,replay
    checks.append('TRAIN JPEG/RGB verification and deterministic crop/flip survive cursor resume')
    backbone=modules['model'].load_pretrained(inputs['config'],inputs['weights'],config['mcunet_root'])
    initial=copy.deepcopy(backbone.state_dict())
    model=WithinBatch160(backbone).eval()
    assert sum(p.numel() for p in model.parameters())==729392
    costs=[]
    for view in ('rgb','ll1_pad'):
        values=[]
        def cost(module,args,output):
            if isinstance(module,torch.nn.Conv2d):
                values.append(output.numel()*module.in_channels//module.groups*math.prod(module.kernel_size))
            else:
                values.append(output.numel()*module.in_features)
        handles=[m.register_forward_hook(cost) for m in model.modules()
                 if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear))]
        model.eval_view=view
        with torch.inference_mode():
            out=model(x[:1])
        assert out.shape==(1,1000) and torch.isfinite(out).all()
        assert sum(values)==67287200
        for h in handles:
            h.remove()
        costs.append(dict(view=view,conv_linear_macs=sum(values)))
    assert equal(initial,backbone.state_dict())
    checks.append('all official pretrained tensors inherited; dual-view eval leaves BN unchanged;67287200MAC each')
    optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=4e-5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,modules['runtime'].lr_factor)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    parts=(model,optimizer,scheduler,scaler)
    progress,contract=dict(step=123),dict(preflight=True,variant='rgb_ll1_center_within_batch')
    model.set_train_step(progress['step'])
    images,labels=first
    seen=[]
    h=backbone.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
    state=modules['runtime'].capture(*parts,progress,contract)
    def fail():
        raise RuntimeError('controlled rollback')
    try:
        modules['runtime'].transactional_step(*parts,images,labels,progress,contract,fail_after_forward=fail)
        raise AssertionError('expected failure')
    except RuntimeError as error:
        assert str(error)=='controlled rollback'
    assert equal(model.state_dict(),state['model'])
    expected=torch.where(batch_mask(123)[:,None,None,None],padded_view(images),images)*2-1
    assert torch.equal(seen[0],expected)
    h.remove()
    loss,_=modules['runtime'].transactional_step(*parts,images,labels,progress,contract)
    assert math.isfinite(loss)
    after=modules['runtime'].capture(*parts,progress,contract)
    assert any(not torch.equal(after['model'][k],v) for k,v in state['model'].items() if k.endswith('weight'))
    restored=modules['runtime'].restore(state,*parts,contract)
    model.set_train_step(restored['step'])
    modules['runtime'].transactional_step(*parts,images,labels,progress,contract)
    replay=modules['runtime'].capture(*parts,progress,contract)
    for field in ('model','optimizer','scheduler','scaler','rng'):
        assert equal(after[field],replay[field]),field
    checks.append('real32-image mixed optimizer update; exact16/16 input routing; failure rollback and exact resume replay')
    base.atomic_json(ROOT/'audit/compute.json',dict(results=costs,parameters=729392,
        scope='Conv/Linear only; excludes preprocessing/BN/activation',sparse_acceleration=False))
    base.atomic_json(ROOT/'preflight.json',dict(status='passed',sources=config['sources'],
        checks=checks,device='CPU',train_images=128000,dev_images=20000,
        disposable_optimizer_updates=2,formal_training_updates=0,
        schedule_sha256=schedule.hexdigest(),finished_at=time.time(),seconds=time.time()-start))
    print('Preflight passed: '+ '; '.join(checks),flush=True)


if __name__=='__main__':
    main()
