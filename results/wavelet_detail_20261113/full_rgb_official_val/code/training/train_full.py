"""Full ImageNet TRAIN RGB160 fine-tuning; fixed epoch20, no dev/test selection."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import full_data as data
import baseline_helpers as base
from w80_local_guard import Guard

ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT.parents[2]
REPO = ROOT.parents[1]
ARCHIVE = FYP / 'ILSVRC2012_img_train/ILSVRC2012_img_train/ILSVRC2012_img_train.tar'
HANDOFF = FYP / 'tmp/team-w80-432bc728-lf'
UPSTREAM = Path(os.environ['MCUNET_ROOT'])
WEIGHTS = REPO / 'weights/mcunet-256kb-1mb_imagenet.pth'

def frozen_sources():
    receipt = json.loads((ROOT / 'source_freeze.json').read_text())
    for rel, digest in receipt['sources'].items():
        if data.sha(ROOT / rel) != digest: raise ValueError('isolated source changed: '+rel)
    return receipt['sources']

def verify_inputs():
    modules = base.load_protocol_modules(HANDOFF)
    modules['execution'].setup(seed=2026)
    base.verify_upstream(UPSTREAM)
    config = modules['root'].parent / 'mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json'
    if data.sha(WEIGHTS) != base.PRETRAINED_SHA256 or data.sha(config) != base.CONFIG_SHA256:
        raise ValueError('official pretrained/config checksum mismatch')
    receipt = data.prepare(ARCHIVE,modules['root'] / 'handoff/classes.json')
    return modules, config, receipt

def lr_factor(update):
    warm, total = 2 * data.STEPS, 20 * data.STEPS
    if update < warm: return (update + 1) / warm
    if update >= total: return 0.0
    return .5 * (1 + math.cos(math.pi * (update-warm)/(total-warm-1)))

def make_parts(modules, config):
    modules['execution'].setup(seed=2026)
    backbone = modules['model'].load_pretrained(config, WEIGHTS, UPSTREAM)
    if backbone.first_conv.conv.in_channels != 3 or tuple(backbone.first_conv.conv.stride) != (2,2) or backbone.classifier.out_features != 1000:
        raise ValueError('unmodified RGB MCUNet-in2 required')
    model = base.RGB160(backbone).cuda().train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
            if m.momentum != .1 or m.eps != 1e-5: raise ValueError('BN settings mismatch')
    if not all(p.requires_grad for p in model.parameters()): raise ValueError('all parameters must be trainable')
    optimizer = torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=4e-5,dampening=0.,nesterov=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    return model,optimizer,scheduler,scaler

def contract_for(modules, receipt):
    return dict(schema=1,variant='full_imagenet_rgb160_fixed20',sources=frozen_sources(),
        inherited_protocol_sha256=base.PROTOCOL_SHA256,protocol_scope='new full-data protocol; not the frozen 128k/20k experiment',
        upstream_commit=base.MCUNET_COMMIT,handoff_commit=base.HANDOFF_COMMIT,
        pretrained_sha256=base.PRETRAINED_SHA256,config_sha256=base.CONFIG_SHA256,
        archive=receipt['archive'],index_sha256=receipt['index_sha256'],classes_sha256=receipt['class_mapping_sha256'],
        train_images=data.N,dev_images=0,official_val_used=False,imagenet_v2_used=False,
        historical_exclusions=False,former_development_images_included=True,
        train_shape=[3,160,160],input_normalization='2*x-1',
        train_transform=dict(crop='RandomResizedCrop160',scale=[.08,1.],ratio=[.75,4/3],interpolation='bicubic',antialias=True,hflip=.5),
        seed=2026,order='randperm(seed2026+1000003*epoch), then repeat first17 to preserve batch32',
        per_epoch_padding=data.PADDED-data.N,samples_per_epoch=data.PADDED,updates_per_epoch=data.STEPS,
        maximum_epochs=20,total_updates=20*data.STEPS,early_stop=False,checkpoint_selection='fixed final epoch20 only; no best-dev selection',
        batch=32,bn_batch=32,gradient_accumulation=1,workers=0,prefetch=1,
        optimizer='SGD',peak_lr=.001,momentum=.9,weight_decay=4e-5,weight_decay_includes_bn_and_bias=True,
        warmup_epochs=2,schedule='two epoch linear warmup, cosine to zero at final update of epoch20',
        precision='fp32',amp=False,tf32=False,all_parameters_trainable=True,bn_update=True,bn_momentum=.1,bn_eps=1e-5,
        read_mode='Windows FILE_FLAG_NO_BUFFERING aligned reads; no JPEG data cache',
        integrity='archive size/mtime and tar index SHA checked on launch; image raw and decoded-RGB hashes established on first use then checked each read',
        resource_pause_host_gib=2,resource_resume_host_gib=4,resource_resume_healthy_seconds=120,
        checkpoint_interval_updates=100,checkpoint_interval_seconds=60)

def initial_progress():
    return dict(epoch=0,step=0,sampler_position=0,training_exposures=0,train_loss_sum=0.,train_top1=0,train_top5=0,
        train_samples=0,epoch_lr_start=None,history=[],complete=False,stop_reason=None)

def write_history(history):
    path = ROOT / 'run/history.csv'; tmp = path.with_suffix('.tmp')
    fields = ['epoch','updates','training_exposures','unique_images_per_epoch','padding_repeats','lr_start','lr_end',
        'train_loss','train_top1','train_top5','wall_seconds']
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); writer.writeheader(); writer.writerows(history)
        f.flush(); os.fsync(f.fileno())
    base.replace_with_retry(tmp,path)

def run(resume=False):
    run_dir = ROOT / 'run'; run_dir.mkdir(exist_ok=True)
    with base.run_lock(run_dir):
        if (run_dir / 'complete.json').exists(): return
        modules,config,receipt = verify_inputs()
        contract = contract_for(modules,receipt)
        parts = make_parts(modules,config); model,optimizer,scheduler,scaler = parts
        cp = modules['runtime'].Checkpoints(run_dir / 'checkpoints')
        config_path = run_dir / 'config.json'
        if resume:
            if json.loads(config_path.read_text())['contract'] != contract: raise ValueError('resume contract mismatch')
            progress = modules['runtime'].restore(cp.load(),*parts,contract)
        else:
            if (run_dir / 'checkpoints/checkpoints.json').exists() or config_path.exists(): raise FileExistsError('use resume for existing formal run')
            progress = initial_progress()
            data.publish(config_path,dict(contract=contract,parameter_count=sum(p.numel() for p in model.parameters()),
                conv_linear_macs_per_image=67287200,final_test='only after epoch20 model/protocol are frozen; no validation tuning'))
        dataset = data.FullImages(ARCHIVE,epoch=progress['epoch'])
        guard = Guard(run_dir / 'resources',modules['runtime'].ResourcePolicy)
        last_save = 0.
        def save(status='training'):
            nonlocal last_save
            dataset.flush()
            name = base.atomic_checkpoint(cp,modules['runtime'].capture(*parts,progress,contract),False)
            data.publish(run_dir / 'progress.json',dict(status=status,epoch=min(progress['epoch']+1,20),
                completed_epochs=progress['epoch'],step=progress['step'],total_steps=20*data.STEPS,
                sampler_position=progress['sampler_position'],samples_per_epoch=data.PADDED,
                training_exposures=progress['training_exposures'],checkpoint=name,updated_at=time.time(),
                train_ce_so_far=progress['train_loss_sum']/progress['train_samples'] if progress['train_samples'] else None))
            last_save = time.monotonic()
        try:
            save()
            base.append_event(run_dir/'events.jsonl','resume' if resume else 'start',step=progress['step'])
            write_history(progress['history'])
            while not progress['complete']:
                epoch = progress['epoch']; dataset.epoch=epoch; started=time.monotonic()
                order=data.epoch_order(epoch)
                order_sha=hashlib.sha256(order.tobytes()).hexdigest()
                if progress['sampler_position']==0:
                    progress['epoch_lr_start']=optimizer.param_groups[0]['lr']
                    base.append_event(run_dir/'events.jsonl','epoch_start',epoch=epoch+1,order_sha256=order_sha)
                iterator=iter(data.batches(dataset,order,progress['sampler_position']))
                while progress['sampler_position']<data.PADDED:
                    guard.wait(on_pause=lambda:save('resource_paused'))
                    images,labels=next(iterator)
                    if len(labels)!=32: raise ValueError('formal microbatch/BN batch must remain32')
                    images,labels=images.cuda(),labels.cuda()
                    loss,_=modules['runtime'].transactional_step(*parts,images,labels,progress,contract)
                    pred=model.last_logits.topk(5,dim=1).indices
                    progress['train_loss_sum']+=loss*32
                    progress['train_top1']+=int((pred[:,0]==labels).sum())
                    progress['train_top5']+=int((pred==labels[:,None]).any(dim=1).sum())
                    progress['train_samples']+=32;progress['step']+=1;progress['sampler_position']+=32;progress['training_exposures']+=32
                    model.last_logits=None
                    del images,labels,pred
                    if progress['step']%100==0 or time.monotonic()-last_save>=60:
                        save()
                        print(f"epoch {epoch+1}/20 step {progress['step']}/{20*data.STEPS} images {progress['sampler_position']}/{data.PADDED} CE {progress['train_loss_sum']/progress['train_samples']:.6f}",flush=True)
                if progress['train_samples']!=data.PADDED: raise ValueError('incomplete full epoch')
                hashes=int(np.count_nonzero(dataset.hashes['valid']))
                if hashes!=data.N: raise ValueError('not all original images consumed/hashed')
                progress['history'].append(dict(epoch=epoch+1,updates=progress['step'],training_exposures=progress['training_exposures'],
                    unique_images_per_epoch=data.N,padding_repeats=data.PADDED-data.N,lr_start=progress['epoch_lr_start'],lr_end=optimizer.param_groups[0]['lr'],
                    train_loss=progress['train_loss_sum']/data.PADDED,train_top1=100*progress['train_top1']/data.PADDED,train_top5=100*progress['train_top5']/data.PADDED,wall_seconds=time.monotonic()-started))
                progress['epoch']+=1; progress['sampler_position']=0
                progress['train_loss_sum']=0.;progress['train_top1']=0;progress['train_top5']=0;progress['train_samples']=0;progress['epoch_lr_start']=None
                if progress['epoch']==20:progress['complete']=True;progress['stop_reason']='fixed20epochs_completed'
                write_history(progress['history']);save('complete' if progress['complete'] else 'training')
                base.append_event(run_dir/'events.jsonl','epoch_complete',epoch=progress['epoch'],step=progress['step'])
                print('completed epoch '+str(progress['epoch']),flush=True)
            pointer=json.loads((run_dir/'checkpoints/checkpoints.json').read_text())
            data.publish(run_dir/'complete.json',dict(status='complete',contract=contract,completed_epochs=20,updates=progress['step'],
                training_exposures=progress['training_exposures'],selected_epoch=20,selected_checkpoint=pointer['latest'],
                official_val_evaluated=False,imagenet_v2_evaluated=False,completed_at=time.time()))
        except BaseException as exc:
            save('interrupted')
            base.append_event(run_dir/'events.jsonl','interrupted',step=progress['step'],error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            guard.close();dataset.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    run(args.resume)
