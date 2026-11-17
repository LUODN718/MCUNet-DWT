"""Validate224 data geometry,64/32 schedules, actual update and exact resume."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import torch
import torch.nn.functional as F
import train_ratio as api
from ratio_input import batch_mask,corrected_ll,padded_center,rgb_count
import w80_tar_source


def macs(backbone,size):
    total=[0];hooks=[]
    def count(module,inputs,output):
        if isinstance(module,torch.nn.Conv2d):
            total[0]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        else:total[0]+=output.numel()*module.in_features
    try:
        for module in backbone.modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
                hooks.append(module.register_forward_hook(count))
        with torch.inference_mode():backbone(torch.zeros(1,3,size,size,device='cuda'))
    finally:
        for hook in hooks:hook.remove()
    return total[0]


def main():
    config=api.read_json(ROOT/'study_config.json')
    modules=api.load_protocol_modules(config['handoff_root'])
    argslist=['--variant','rgb33_ll67','--batch','64','--run-dir',str(ROOT/'runs/preflight_unused'),
        '--accept-local-content-sha256',config['manifest_sha256']]
    for key in ('handoff_root','train_tar','seal_dir','weights','mcunet_root'):
        argslist+=['--'+key.replace('_','-'),config[key]]
    args=api.parse_args(argslist)
    inputs=api.verify_inputs(args,modules)
    api.BATCH_SIZE=64;api.UPDATES_PER_EPOCH=2000
    model,optimizer,scheduler,scaler=api.model_and_optimizers(args,inputs,modules)
    checks=[]
    schedules={}
    for batch in (32,64):
        steps=128000//batch
        count=sum(rgb_count(step,'rgb33_ll67',batch) for step in range(steps*20))
        assert abs(count/(steps*20*batch)-1/3)<1/(steps*20*batch)
        assert sum(rgb_count(step,'rgb33_ll67',batch) for step in range(3))==batch
        assert int(batch_mask(17,'rgb33_ll67',batch).sum())==rgb_count(17,'rgb33_ll67',batch)
        assert torch.equal(batch_mask(17,'rgb33_ll67',batch),batch_mask(17,'rgb33_ll67',batch))
        assert sum(rgb_count(step,'rgb_only',batch) for step in range(steps))==128000
        assert modules['runtime'].lr_factor(2*steps-1,steps)==1.
        assert modules['runtime'].lr_factor(20*steps-1,steps)==0.
        schedules[str(batch)]=dict(steps_per_epoch=steps,maximum_updates=steps*20,rgb_samples=count,ll_samples=2560000-count)
    checks.append('32/64 exact epoch cardinality; near1:2 global mixtures; deterministic masks; epoch-scaled warmup/cosine')
    local={**inputs['loader_local'],'batch':64}
    train_images,labels=next(iter(w80_tar_source.loader(local,inputs['train_rows'][:64],'train')))
    dev_images,_=next(iter(w80_tar_source.loader(local,inputs['dev_rows'][:64],'dev')))
    assert train_images.shape==dev_images.shape==(64,3,224,224)
    assert torch.equal(corrected_ll(train_images),F.avg_pool2d(train_images,2))
    padded=padded_center(train_images)
    assert torch.equal(padded[...,56:168,56:168],corrected_ll(train_images))
    border=torch.ones(224,224,dtype=torch.bool);border[56:168,56:168]=False
    assert torch.all((padded*2-1)[...,border]==-1)
    checks.append('real TRAIN/dev byte and RGB identity checks;224 crop;112LL; centered56px padding before normalization')
    image,target=train_images.cuda(),labels.cuda()
    model.set_train_step(0)
    loss,_=modules['runtime'].transactional_step(model,optimizer,scheduler,scaler,image,target,{}, {})
    state=modules['runtime'].capture(model,optimizer,scheduler,scaler,{}, {})
    model.set_train_step(1)
    loss2,_=modules['runtime'].transactional_step(model,optimizer,scheduler,scaler,image,target,{}, {})
    expected={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    expected_lr=optimizer.param_groups[0]['lr']
    modules['runtime'].restore(state,model,optimizer,scheduler,scaler,{})
    model.set_train_step(1)
    resumed_loss,_=modules['runtime'].transactional_step(model,optimizer,scheduler,scaler,image,target,{}, {})
    assert loss2==resumed_loss
    assert expected_lr==optimizer.param_groups[0]['lr']
    assert all(torch.equal(value,model.state_dict()[key].cpu()) for key,value in expected.items())
    checks.append('FP32 batch64 real-data SGD steps finite; resumed next step exactly reproduces weights,BN,loss and LR')
    model.eval()
    with torch.inference_mode():
        for view in ('rgb','ll1_pad'):
            model.eval_view=view
            # Full evaluation has20,000 images; final batch32 must work with batch64 model.
            logits=model(dev_images[:32].cuda())
            assert logits.shape==(32,1000) and torch.isfinite(logits).all()
    checks.append('both eval views support final32-image remainder at batch64')
    cost224=macs(model.backbone,224);cost112=macs(model.backbone,112)
    result=dict(status='passed',formal_training_updates=0,disposable_preflight_updates=3,
        checks=checks,schedules=schedules,conv_linear_macs_224=cost224,
        conv_linear_macs_native112=cost112,preprocessing_cost_included=False,
        train_images=128000,dev_images=20000,classes=1000,finished_at=time.time())
    api.atomic_json(ROOT/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)
    paths=list((ROOT/'code').glob('*.py'))+[ROOT/'run_queue.py',ROOT/'prepare.py',ROOT/'preflight.py',ROOT/'batch_probe.py']
    config['sources']={str(path.relative_to(ROOT)).replace('\\','/'):api.sha256(path) for path in paths}
    api.atomic_json(ROOT/'study_config.json',config)


if __name__=='__main__':main()
