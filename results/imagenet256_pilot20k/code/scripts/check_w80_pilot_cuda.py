#!/usr/bin/env python3
"""Real CUDA batch32 checks for the local resized-image W80 pilot only."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.w80_pilot.data import MirrorImages
from experiments.w80_pilot.vendor import model as w80, runtime
from experiments.w80_pilot.vendor.data import sha, digest


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def make_parts(base, indices, variant):
    model = w80.make_variant(base, indices, variant).cuda().train()
    if variant == 'rgb_e23':
        groups = [{'params': list(model.parameters()), 'lr': .001}]
    else:
        stem = list(model.backbone.first_conv.parameters())
        stem_ids = {id(p) for p in stem}
        groups = [{'params': [p for p in model.parameters() if id(p) not in stem_ids], 'lr': .001},
                  {'params': stem, 'lr': .005}]
    optimizer = torch.optim.SGD(groups, momentum=.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: runtime.lr_factor(step, 500))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    return model, optimizer, scheduler, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/w80_e23_pilot20k')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'datasets/imagenet-1k-256-subset200/train')
    parser.add_argument('--mcunet-root', type=Path, default=Path.home() / 'Desktop/Mcunet/mcunet')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    manifest_path = args.run_root / 'data/manifest.json'
    selection_path = args.run_root / 'channel_selection.json'
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection['contract']['manifest_sha256'] != sha(manifest_path):
        raise ValueError('calibration manifest mismatch')
    if selection['indices_sha256'] != digest(selection['indices']):
        raise ValueError('channel index checksum mismatch')
    vendor = ROOT / 'experiments/w80_pilot/vendor'
    source = json.loads((vendor / 'SOURCE.json').read_text())
    for filename, checksum in source['files_sha256'].items():
        if sha(vendor / filename) != checksum:
            raise ValueError('vendored teammate component changed: ' + filename)
    by_path = {r['path']: r for r in manifest['rows'] if r['role'] == 'train'}
    rows = [by_path[p] for p in manifest['calibration'][:32]]
    loader = DataLoader(MirrorImages(args.data_root, rows, 'train', epoch=0), batch_size=32, num_workers=0)
    x, y = next(iter(loader)); x = x.cuda(); y = y.cuda()
    base = w80.load_pretrained(ROOT / 'weights/mcunet-256kb-1mb_imagenet.json',
                               ROOT / 'weights/mcunet-256kb-1mb_imagenet.pth', args.mcunet_root)
    contract = dict(manifest_sha256=sha(manifest_path), selection_sha256=sha(selection_path),
                    precision='fp32', tf32=False, batch=32, steps_per_epoch=500, total_epochs=20)
    results = []
    reference_logits = None
    for variant in w80.VARIANTS:
        parts = make_parts(base, selection['indices'], variant)
        model, optimizer, scheduler, scaler = parts
        model.eval()
        with torch.no_grad():
            logits = model(x)
        if logits.shape != (32, 1000) or logits.dtype != torch.float32:
            raise RuntimeError('wrong output shape or precision')
        if variant == 'll':
            reference_logits = logits.detach().clone()
        elif variant != 'rgb_e23':
            torch.testing.assert_close(logits, reference_logits, rtol=3e-5, atol=3e-6)
        progress = dict(epoch=0, batch_cursor=0, successful_updates=0, training_exposures=0)
        torch.cuda.reset_peak_memory_stats()
        loss, _ = runtime.transactional_step(*parts, x, y, progress, contract)
        progress.update(batch_cursor=1, successful_updates=1, training_exposures=32)
        if variant != 'rgb_e23' and variant != 'll':
            if model.backbone.first_conv.pw.weight[:, 3:].count_nonzero().item() == 0:
                raise RuntimeError('high-frequency PW columns did not learn')
        saved = runtime.capture(*parts, progress, contract)
        with tempfile.TemporaryDirectory(dir=args.run_root) as temporary:
            checkpoints = runtime.Checkpoints(temporary)
            checkpoints.save(saved, best=True)
            runtime.transactional_step(*parts, x, y, progress, contract)
            expected = runtime.capture(*parts, progress, contract)
            fresh = make_parts(base, selection['indices'], variant)
            restored = runtime.restore(checkpoints.load(), *fresh, contract)
            runtime.transactional_step(*fresh, x, y, restored, contract)
            actual = runtime.capture(*fresh, restored, contract)
            if not equal_state(expected, actual):
                raise RuntimeError('interrupted batch32 update did not reproduce full state: ' + variant)
            del fresh, expected, actual
        # Inject after a real CUDA forward/BN update; do not exhaust device RAM.
        before = runtime.capture(*parts, progress, contract)
        def injected_oom():
            raise torch.cuda.OutOfMemoryError('controlled pilot failure after CUDA BN')
        try:
            runtime.transactional_step(*parts, x, y, progress, contract, injected_oom)
        except torch.cuda.OutOfMemoryError:
            pass
        else:
            raise RuntimeError('controlled OOM not raised')
        if not equal_state(before, runtime.capture(*parts, progress, contract)):
            raise RuntimeError('OOM failed to roll back BN/optimizer/RNG')
        original_step = optimizer.step
        def partially_failed_step(*a, **kw):
            original_step(*a, **kw)
            raise RuntimeError('controlled failure after optimizer mutation')
        optimizer.step = partially_failed_step
        try:
            runtime.transactional_step(*parts, x, y, progress, contract)
        except RuntimeError as error:
            if 'controlled failure' not in str(error):
                raise
        else:
            raise RuntimeError('controlled optimizer failure not raised')
        finally:
            optimizer.step = original_step
        if not equal_state(before, runtime.capture(*parts, progress, contract)):
            raise RuntimeError('partial optimizer failure did not roll back all state')
        result = dict(variant=variant, real_cuda_batch=32, output_classes=1000, fp32=True,
                      finite_ce=bool(np.isfinite(loss)), disk_resume_exact=True,
                      controlled_after_bn_oom_rollback=True, partial_optimizer_rollback=True,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2))
        print(json.dumps(result), flush=True); results.append(result)
        del parts, model, optimizer, scheduler, scaler, before, saved, logits
        torch.cuda.empty_cache()
    receipt = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), contract=contract, variants=results,
                   passed=True, scope='resized-image pilot CUDA integration; not teammate original-TRAIN production admission',
                   source_commit=source['commit'], physical_device_oom_induced=False,
                   watchdog_integration_verified=False, official_val_or_v2_used=False)
    output = args.run_root / 'cuda_checks.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('Saved CUDA checks:', output, flush=True)


if __name__ == '__main__':
    main()
