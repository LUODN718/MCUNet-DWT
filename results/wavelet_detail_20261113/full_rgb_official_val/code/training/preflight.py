"""Admission checks for the new full-data protocol; never a formal update."""
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
import full_data as data
import train_full as train

def exact(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def main():
    started=time.time(); root=train.ROOT; checks=[]
    seal=train.REPO/'runs/w80_rgb160_original_tar_seal'
    with (seal/'tar_index.tsv').open() as f: indices=list(csv.DictReader(f,delimiter='\t'))
    with (seal/'manifest.local_accepted.tsv').open() as f: rows={r['path']:r for r in csv.DictReader(f,delimiter='\t')}
    reader=data.UncachedReader(train.ARCHIVE)
    with open(train.ARCHIVE,'rb') as cached:
        for item in (indices[0],indices[len(indices)//2],indices[-1]):
            offset,size=int(item['offset']),int(item['size']);raw=reader.read(offset,size)
            cached.seek(offset);assert raw==cached.read(size)
            assert hashlib.sha256(raw).hexdigest()==rows[item['path']]['sha']
            with Image.open(io.BytesIO(raw)) as source: rgb=source.convert('RGB')
            assert hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()==rows[item['path']]['rgb']
    checks.append('aligned uncached reads first/middle/last equal accepted JPEG and RGB seals')
    alignment=reader.alignment;reader.close();del indices,rows;gc.collect()
    modules,config,receipt=train.verify_inputs(); contract=train.contract_for(modules,receipt)
    assert data.N==1_281_167 and data.STEPS==40037 and data.PADDED-data.N==17
    for epoch in (0,19):
        order=data.epoch_order(epoch)
        assert len(order)==data.PADDED and np.array_equal(np.sort(order[:data.N]),np.arange(data.N))
        assert np.array_equal(order[data.N:],order[:17])
    checks.append('first and last epochs consume all1281167 unique identities; deterministic17 repeat padding')
    assert train.lr_factor(0)==1/(2*data.STEPS)
    assert train.lr_factor(2*data.STEPS-1)==1.
    assert train.lr_factor(20*data.STEPS-1)==0. and train.lr_factor(20*data.STEPS)==0.
    checks.append('full-data40037 updates/epoch; two-epoch warmup;800740th update zero LR')
    dataset=data.FullImages(train.ARCHIVE)
    dataset.epoch=0;order=data.epoch_order(0)
    a=dataset[int(order[0])];b=dataset[int(order[0])]
    assert a[1]==b[1] and torch.equal(a[0],b[0])
    x,y=next(iter(data.batches(dataset,order[:32])))
    assert tuple(x.shape)==(32,3,160,160) and y.dtype==torch.int64 and torch.isfinite(x).all()
    checks.append('real full-index image decode and deterministic frozen160 crops; batch32')
    parts=train.make_parts(modules,config);model,opt,sched,scaler=parts
    runtime=modules['runtime'];progress=train.initial_progress();x,y=x.cuda(),y.cuda()
    assert not scaler.is_enabled() and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert all(p.requires_grad for p in model.parameters())
    before=runtime.capture(*parts,progress,contract)
    t=time.monotonic();loss,_=runtime.transactional_step(*parts,x,y,progress,contract)
    torch.cuda.synchronize(); step_seconds=time.monotonic()-t
    after=runtime.capture(*parts,progress,contract);assert loss>0
    del parts,model,opt,sched,scaler;gc.collect();torch.cuda.empty_cache()
    parts=train.make_parts(modules,config)
    restored=runtime.restore(before,*parts,contract)
    loss2,_=runtime.transactional_step(*parts,x,y,restored,contract)
    assert loss==loss2 and exact(after,runtime.capture(*parts,restored,contract))
    checks.append('CUDA batch32 FP32 update exact replay after fresh model optimizer scheduler RNG restore')
    cp=runtime.Checkpoints(root/'preflight/checkpoints')
    train.base.atomic_checkpoint(cp,after,False)
    restored=runtime.restore(cp.load(),*parts,contract)
    assert exact(after,runtime.capture(*parts,restored,contract))
    def fail(): raise RuntimeError('admission injected after-forward failure')
    try: runtime.transactional_step(*parts,x,y,restored,contract,fail_after_forward=fail)
    except RuntimeError as exc:
        if str(exc)!='admission injected after-forward failure': raise
    else: raise AssertionError('injected failure did not occur')
    assert exact(after,runtime.capture(*parts,restored,contract))
    checks.append('durable checkpoint reload exact; injected forward failure rolls back model BN optimizer scheduler RNG')
    # Compare32 deterministic actual updates for a useful throughput estimate.
    elapsed=[]
    for _ in range(4):
        t=time.monotonic();runtime.transactional_step(*parts,x,y,restored,contract);torch.cuda.synchronize();elapsed.append(time.monotonic()-t)
    params=sum(p.numel() for p in parts[0].parameters());dataset.close()
    data.publish(root/'preflight/admission.json',dict(status='passed',sources=train.frozen_sources(),
        contract=contract,checks=checks,parameters=params,alignment=alignment,real_cuda_step_seconds=step_seconds,
        repeated_batch_step_seconds=elapsed,index_sha256=receipt['index_sha256'],completed_at=time.time(),elapsed_seconds=time.time()-started))
    print(json.dumps(dict(status='passed',checks=checks,parameters=params,step_seconds=elapsed)),flush=True)

if __name__=='__main__': main()
