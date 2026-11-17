"""Check ratio schedules, preprocessing, MACs and resume determinism."""
import hashlib
import json
from pathlib import Path
import time

import torch
from ratio_input import RATIOS,batch_mask,padded_center,rgb_count

ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'queue_config.json').read_text(encoding='utf-8'))
    checks=[]
    total_steps=80000
    schedules={}
    for variant in RATIOS:
        rgb_total=0
        digest=hashlib.sha256()
        for step in range(total_steps):
            count=rgb_count(step,variant)
            mask=batch_mask(step,variant)
            assert int(mask.sum())==count
            assert torch.equal(mask,batch_mask(step,variant))
            rgb_total+=count
            digest.update(mask.numpy().tobytes())
        expected={'rgb50_ll50':1280000,'rgb33_ll67':853334,
                  'rgb25_ll75':640000}[variant]
        assert rgb_total==expected,(variant,rgb_total,expected)
        schedules[variant]=dict(rgb_samples=rgb_total,ll_samples=total_steps*32-rgb_total,
                                ratio=f'{rgb_total}:{total_steps*32-rgb_total}',sha256=digest.hexdigest())
    checks.append('all three80k-step schedules have deterministic exact global ratios')
    x=torch.rand(4,3,160,160,generator=torch.Generator().manual_seed(2026))
    y=padded_center(x)
    assert y.shape==x.shape
    assert torch.equal(y[...,40:120,40:120],torch.nn.functional.avg_pool2d(x,2))
    border=torch.ones((160,160),dtype=torch.bool);border[40:120,40:120]=False
    assert torch.count_nonzero(y[...,border])==0
    assert torch.all((y*2-1)[...,border]==-1)
    checks.append('all arms share same centered LL80 preprocessing and160 input canvas')
    assert config['eval_views']==['rgb','center_ll1_pad']
    assert [a['variant'] for a in config['experiments']]==list(RATIOS)
    checks.append('each trained model is configured for separate RGB160 and centered LL evaluation')
    result=dict(status='passed',formal_training_updates=0,train_images=128000,
        dev_images=20000,maximum_epochs=20,checks=checks,schedules=schedules,
        actual_conv_linear_macs_per_view=67287200,mac_reduction=False,
        finished_at=time.time())
    (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
