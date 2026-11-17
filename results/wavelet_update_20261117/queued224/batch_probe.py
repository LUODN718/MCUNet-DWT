"""Disposable synthetic 224px FP32 throughput probes; never writes train weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'pad_mix_ratio_study_20261115'


def worker(batch):
    sys.path.insert(0, str(BASE / 'code'))
    import torch
    import torch.nn.functional as F
    import train_ratio as api
    config = api.read_json(BASE / 'queue_config.json')
    modules = api.load_protocol_modules(config['handoff_root'])
    modules['execution'].setup(seed=2026)
    # Probe only: fail cleanly before using shared GPU system memory.
    torch.cuda.set_per_process_memory_fraction(.80)
    backbone = modules['model'].load_pretrained(
        modules['root'].parent / 'mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json',
        config['weights'], config['mcunet_root']).cuda().train()
    optimizer = torch.optim.SGD(backbone.parameters(), lr=.001, momentum=.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    x = torch.rand(batch, 3, 224, 224, device='cuda')
    target = torch.arange(batch, device='cuda') % 1000
    result = dict(batch=batch, input=224, precision='FP32; AMP/TF32 off', synthetic=True)
    try:
        torch.cuda.reset_peak_memory_stats()
        for step in range(10):
            torch.cuda.synchronize()
            if step == 2:
                started = time.perf_counter()
            n = (batch//3, batch//3, batch-2*(batch//3))[step%3]
            low = F.pad(F.avg_pool2d(x, 2), (56, 56, 56, 56))
            image = torch.cat((x[:n], low[n:])) * 2 - 1
            loss, before = modules['runtime'].transactional_step(
                backbone, optimizer, scheduler, scaler, image, target, {}, {})
            del before, image, low
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        result.update(status='passed', samples_per_second=batch*8/elapsed,
            seconds_per_update=elapsed/8, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), free_gpu_bytes=torch.cuda.mem_get_info()[0])
    except torch.OutOfMemoryError as error:
        result.update(status='oom', error=str(error).splitlines()[0])
    print('PROBE_JSON=' + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    if args.batch:
        worker(args.batch)
        return
    results = []
    for batch in (32, 64, 128):
        proc = subprocess.run([sys.executable, '-B', '-u', __file__, '--batch', str(batch)],
                              capture_output=True, text=True)
        (ROOT / f'probe_batch{batch}.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
        lines = [line for line in proc.stdout.splitlines() if line.startswith('PROBE_JSON=')]
        if proc.returncode or not lines:
            raise RuntimeError(f'batch{batch} probe failed; inspect saved log')
        result = json.loads(lines[-1].split('=', 1)[1])
        results.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / 'batch_probe.json').write_text(json.dumps(dict(results=results), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
