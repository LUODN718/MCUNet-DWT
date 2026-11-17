"""CPU-only input, schedule, metrics and exact recovery admission before handoff."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

import train_pad_mix as base
import w80_tar_source
from padded_input import PaddedMix160, corrected_ll, padded_view, epoch_views, view_for_step

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    started = time.time()
    config = base.read_json(ROOT / 'queue_config.json')
    for relative, checksum in config['sources'].items():
        assert base.sha256(ROOT / relative) == checksum, relative
    command = ['--variant', 'rgb_ll1_pad_mix']
    for name in ('handoff_root', 'train_tar', 'seal_dir', 'weights', 'mcunet_root'):
        command += ['--' + name.replace('_', '-'), config[name]]
    command += ['--run-dir', str(ROOT / 'runs/p2_rgb_ll1_pad_mix'),
                '--accept-local-content-sha256', config['manifest_sha256']]
    modules = base.load_protocol_modules(config['handoff_root'])
    inputs = base.verify_inputs(base.parse_args(command), modules)
    assert (len(inputs['train_rows']), len(inputs['dev_rows']), len(inputs['classes'])) == (128000, 20000, 1000)
    checks = ['frozen cohort/local byte-RGB seal, official pretrained, upstream and decoder environment']
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2026)
    x = torch.rand(2, 3, 160, 160, generator=generator)
    for level in (1, 2, 3):
        ll, padded = corrected_ll(x, level), padded_view(x, level)
        size = 160 // (2**level)
        torch.testing.assert_close(ll, torch.nn.functional.avg_pool2d(x, 2**level), atol=3e-7, rtol=1e-6)
        assert tuple(padded.shape) == (2, 3, 160, 160)
        assert torch.equal(padded[..., :size, :size], ll)
        assert torch.count_nonzero(padded[..., size:, :]) == 0
        assert torch.count_nonzero(padded[..., :, size:]) == 0
        normalized = padded*2-1
        assert torch.all(normalized[..., size:, :] == -1)
        assert torch.all(normalized[..., :, size:] == -1)
        constant = corrected_ll(torch.full_like(x, .37), level)
        torch.testing.assert_close(constant, torch.full_like(constant, .37), atol=1e-7, rtol=0)
    # The active region matches historical LL amplitude exactly; padding does
    # not accidentally include a level-dependent brightness increase.
    torch.testing.assert_close(padded_view(x)[..., :80, :80]*2-1,
                               modules['model'].haar_input(x, 'll'), atol=1e-7, rtol=1e-6)
    checks.append('LL1/2/3 shapes/DC correction; top-left no-resize block; zero before normalization -> -1 pad')

    schedules = []
    for epoch in range(20):
        before = torch.get_rng_state().clone()
        views = epoch_views(epoch)
        assert torch.equal(before, torch.get_rng_state())
        assert len(views) == 4000 and sum(views) == 2000
        assert sum(view_for_step('rgb_ll1_pad_mix', epoch*4000+i) == 'rgb' for i in range(4000)) == 2000
        assert view_for_step('rgb_only', epoch*4000+700) == 'rgb'
        assert view_for_step('ll1_pad_only', epoch*4000+700) == 'll1_pad'
        schedules.append(dict(epoch=epoch+1, rgb_batches=2000, ll1_pad_batches=2000,
                              schedule_sha256=hashlib.sha256(bytes(views)).hexdigest()))
    assert schedules[0]['schedule_sha256'] != schedules[1]['schedule_sha256']
    base.atomic_json(ROOT / 'audit/mixture_schedules.json', dict(seed=2026, rows=schedules,
        unit='whole batch32; each epoch original128000 samples exactly once; no extra presentations'))
    checks.append('20 deterministic local-RNG schedules, each exactly2000 RGB +2000 LL batches; controls unchanged')

    # Actual direct tar decoding and deterministic geometry survive cursor resume.
    import importlib.util
    old_path = ROOT.parent / 'detail_sparsity_ll_up160_20261110/code/w80_tar_source.py'
    spec = importlib.util.spec_from_file_location('old_tar_adapter', old_path)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    current = w80_tar_source.TarOriginalImages(config['train_tar'], ROOT / 'data/seal/tar_index.tsv',
                                                inputs['train_rows'], 'train', epoch=3)
    old = historical.TarOriginalImages(config['train_tar'], ROOT / 'data/seal/tar_index.tsv',
                                       inputs['train_rows'], 'train', epoch=3)
    for i in range(4):
        actual, reference = current[i], old[i]
        assert torch.equal(actual[0], reference[0]) and actual[1] == reference[1]
    current.close()
    old.close()
    order = modules['data'].epoch_order(128000, 2026, 3)[:64]
    batches = iter(w80_tar_source.loader(inputs['loader_local'], inputs['train_rows'], 'train',
                                        epoch=3, order=order, workers=0))
    next(batches)
    expected = next(batches)
    resumed = next(iter(w80_tar_source.loader(inputs['loader_local'], inputs['train_rows'], 'train',
                       epoch=3, order=order, cursor=32, workers=0)))
    assert torch.equal(expected[0], resumed[0]) and torch.equal(expected[1], resumed[1])
    del batches, expected, resumed
    checks.append('sample-read byte/RGB hashes and exactly inherited crop/flip; resumed64-image loader replay')

    backbone = modules['model'].load_pretrained(inputs['config'], inputs['weights'], config['mcunet_root'])
    original_state = copy.deepcopy(backbone.state_dict())
    model = PaddedMix160(backbone, 'rgb_ll1_pad_mix').eval()
    assert sum(p.numel() for p in model.parameters()) == 729392
    assert tuple(backbone.first_conv.conv.stride) == (2, 2)
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.eps == 1e-5 and module.momentum == .1
    costs = []
    for view in ('rgb', 'll1_pad'):
        macs, handles, stem = [], [], []
        def capture(module, args, output):
            if isinstance(module, torch.nn.Conv2d):
                macs.append(output.numel()*module.in_channels//module.groups*math.prod(module.kernel_size))
            else:
                macs.append(output.numel()*module.in_features)
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(capture))
        handles.append(backbone.first_conv.conv.register_forward_hook(
            lambda module, args, output: stem.append(list(output.shape))))
        model.eval_view = view
        with torch.inference_mode():
            output = model(x[:1])
        assert tuple(output.shape) == (1, 1000) and torch.isfinite(output).all()
        assert sum(macs) == 67287200, (view, sum(macs))
        for handle in handles:
            handle.remove()
        costs.append(dict(view=view, parameters=729392, conv_linear_macs=sum(macs), stem_output=stem[0]))
    assert all(torch.equal(v, original_state[k]) for k, v in backbone.state_dict().items())
    base.atomic_json(ROOT / 'audit/compute.json', dict(rows=costs,
        scope='actual forward-hook Conv/Linear MACs; excludes Haar/pad/normalization/BN/activation/decode',
        sparse_acceleration=False))
    checks.append('all pretrained tensors inherited; shared BN defaults; both views729392params/67287200MAC')

    # Both RGB and padded batches must be real optimizer updates, and both must
    # reproduce after model/BN/optimizer/scheduler/RNG+step cursor restoration.
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, modules['runtime'].lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    parts = (model, optimizer, scheduler, scaler)
    y = torch.tensor([1, 2])
    progress, contract = dict(step=0), dict(preflight=True, variant='rgb_ll1_pad_mix')
    bn_key = next(k for k in model.state_dict() if k.endswith('running_mean'))
    for view in ('rgb', 'll1_pad'):
        step = next(i for i in range(4000) if view_for_step('rgb_ll1_pad_mix', i) == view)
        progress['step'] = step
        model.set_train_step(step)
        state = modules['runtime'].capture(*parts, progress, contract)
        def failure():
            raise RuntimeError('controlled mixture rollback')
        try:
            modules['runtime'].transactional_step(*parts, x, y, progress, contract,
                                                 fail_after_forward=failure)
            raise AssertionError('failure not propagated')
        except RuntimeError as error:
            assert str(error) == 'controlled mixture rollback'
        assert equal(model.state_dict(), state['model'])
        modules['runtime'].transactional_step(*parts, x, y, progress, contract)
        first = modules['runtime'].capture(*parts, progress, contract)
        assert not torch.equal(first['model'][bn_key], state['model'][bn_key])
        restored_progress = modules['runtime'].restore(state, *parts, contract)
        model.set_train_step(restored_progress['step'])
        assert view_for_step('rgb_ll1_pad_mix', restored_progress['step']) == view
        modules['runtime'].transactional_step(*parts, x, y, progress, contract)
        replay = modules['runtime'].capture(*parts, progress, contract)
        for field in ('model', 'optimizer', 'scheduler', 'scaler', 'rng'):
            assert equal(first[field], replay[field]), (view, field)
    checks.append('RGB and LL optimizer updates; failed-update BN rollback; exact cursor/RNG/optimizer replay both routes')

    # Numerical visual audit on an actual development crop, not a synthetic image.
    ds = w80_tar_source.TarOriginalImages(config['train_tar'], ROOT / 'data/seal/tar_index.tsv',
                                         inputs['dev_rows'], 'dev')
    picture, label = ds[0]
    ds.close()
    raw = picture.unsqueeze(0)
    panels = [('RGB crop160', raw), ('LL1 80 + zero pad160', padded_view(raw, 1)),
              ('LL2 40 + zero pad160', padded_view(raw, 2)),
              ('LL3 20 + zero pad160', padded_view(raw, 3)),
              ('OLD LL1 bilinear up160', torch.nn.functional.interpolate(corrected_ll(raw, 1), (160,160), mode='bilinear',align_corners=False)),
              ('OLD LL2 bilinear up160', torch.nn.functional.interpolate(corrected_ll(raw, 2), (160,160), mode='bilinear',align_corners=False))]
    montage = Image.new('RGB', (3*340, 2*374), '#eeeeee')
    draw = ImageDraw.Draw(montage)
    metadata = []
    for index, (title, image) in enumerate(panels):
        array = image[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
        tile = Image.fromarray(array).resize((320,320), Image.Resampling.NEAREST)
        col, row = index%3, index//3
        montage.paste(tile, (col*340+10, row*374+38))
        draw.text((col*340+10, row*374+12), title, fill='black')
        metadata.append(dict(title=title, shape=list(image.shape), min=float(image.min()), max=float(image.max())))
    montage.save(ROOT / 'visualization/padding_vs_upsampling.png')
    base.atomic_json(ROOT / 'visualization/metadata.json', dict(
        source=inputs['dev_rows'][0], label=label, panels=metadata,
        display_scale='each160canvas shown320x320 nearest-neighbor; no scientific interpolation',
        coefficient_gain_correction='LLj / 2**j', placement='top-left',
        padding_value_before_normalization=0, padding_value_after_normalization=-1))
    checks.append('actual ImageNet crop montage for LL1/2/3 zero padding versus historical bilinear upsampling')
    base.atomic_json(ROOT / 'preflight.json', dict(status='passed', device='CPU', checks=checks,
        sources=config['sources'], train_images=128000, dev_images=20000,
        manifest_sha256=config['manifest_sha256'], cpu_disposable_updates=4,
        cuda_training_updates=0, parameter_count=729392, conv_linear_macs=67287200,
        coefficient_level_training=1, pad_value_before_normalization=0, pad_value_after_normalization=-1,
        finished_at=time.time(), seconds=time.time()-started))
    print(json.dumps(dict(status='passed', checks=checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
