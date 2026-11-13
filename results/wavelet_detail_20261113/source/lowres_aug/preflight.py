"""Admit real low-resolution inputs, deterministic augmentation and CUDA resume."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import torch

import train_lowres as base
import w80_tar_source
from low_resolution import LowResolution80, low_resolution_view
from augmentation import SETTINGS

ROOT = Path(__file__).resolve().parents[1]


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    started = time.time()
    config = base.read_json(ROOT / 'queue_config.json')
    for relative, checksum in config['sources'].items():
        assert base.sha256(ROOT / relative) == checksum, relative
    command = ['--variant', 'll80', '--augmentation', 'light']
    for name in ('handoff_root', 'train_tar', 'seal_dir', 'weights', 'mcunet_root'):
        command += ['--' + name.replace('_', '-'), config[name]]
    command += ['--run-dir', str(ROOT / 'runs/ll80_light'),
                '--accept-local-content-sha256', config['manifest_sha256']]
    args = base.parse_args(command)
    modules = base.load_protocol_modules(config['handoff_root'])
    inputs = base.verify_inputs(args, modules)
    assert (len(inputs['train_rows']), len(inputs['dev_rows']), len(inputs['classes'])) == (128000, 20000, 1000)
    checks = ['frozen 128k/20k membership, local JPEG/RGB seal, upstream, pretrained and decoder versions']

    # Extra transforms leave the old geometric crop/flip unchanged, do not
    # consume training RNG, and produce identical per-sample views after resume.
    previous_path = ROOT.parent / 'detail_sparsity_ll_up160_20261110/code/w80_tar_source.py'
    spec = importlib.util.spec_from_file_location('historical_tar_adapter', previous_path)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    def dataset(rows, role, augmentation, epoch=2):
        return w80_tar_source.TarOriginalImages(config['train_tar'],
            ROOT / 'data/seal/tar_index.tsv', rows, role, epoch=epoch, augmentation=augmentation)
    old = historical.TarOriginalImages(config['train_tar'], ROOT / 'data/seal/tar_index.tsv',
                                       inputs['train_rows'], 'train', epoch=2)
    plain = dataset(inputs['train_rows'], 'train', 'plain')
    light = dataset(inputs['train_rows'], 'train', 'light')
    dev_plain = dataset(inputs['dev_rows'], 'dev', 'plain')
    dev_light = dataset(inputs['dev_rows'], 'dev', 'light')
    images, labels, altered = [], [], 0
    for i in range(8):
        p, label = plain[i]
        reference, old_label = old[i]
        assert torch.equal(p, reference) and label == old_label
        before = torch.get_rng_state().clone()
        a, light_label = light[i]
        again, again_label = light[i]
        assert torch.equal(before, torch.get_rng_state())
        assert torch.equal(a, again) and label == light_label == again_label
        assert tuple(a.shape) == (3, 160, 160) and 0 <= float(a.min()) <= float(a.max()) <= 1
        altered += int(not torch.equal(a, p))
        images.append(a)
        labels.append(label)
        dp, dl = dev_plain[i], dev_light[i]
        assert torch.equal(dp[0], dl[0]) and dp[1] == dl[1]
    assert altered > 0
    for d in (old, plain, light, dev_plain, dev_light):
        d.close()
    checks.append('same historical geometry/plain pixels; deterministic shared augmentation; untouched RNG; no dev augmentation')

    # Loader resume uses an image-addressed augmentation seed, not how many
    # previous images or transforms happened to be read by a worker.
    order = modules['data'].epoch_order(128000, 2026, 2)[:64]
    local = {**inputs['loader_local'], 'extra_augmentation': 'light'}
    batches = iter(w80_tar_source.loader(local, inputs['train_rows'], 'train',
                   epoch=2, order=order, workers=0))
    next(batches)
    expected = next(batches)
    resumed = next(iter(w80_tar_source.loader(local, inputs['train_rows'], 'train',
                   epoch=2, order=order, cursor=32, workers=0)))
    assert torch.equal(expected[0], resumed[0]) and torch.equal(expected[1], resumed[1])
    del batches, expected, resumed
    checks.append('exact image/label replay at a resumed loader cursor')

    torch.set_num_threads(2)
    x = torch.stack(images)
    ll = low_resolution_view(x, 'll80')
    old_normalized = modules['model'].haar_input(x, 'll')
    assert torch.equal(ll * 2 - 1, old_normalized)
    torch.testing.assert_close(ll, torch.nn.functional.avg_pool2d(x, 2), atol=1e-7, rtol=1e-6)
    for variant in ('ll80', 'rgb80'):
        view = low_resolution_view(x, variant)
        assert tuple(view.shape) == (8, 3, 80, 80)
        assert 0 <= float(view.min()) <= float(view.max()) <= 1
        constant = low_resolution_view(torch.full_like(x, .37), variant)
        torch.testing.assert_close(constant, torch.full_like(constant, .37), atol=2e-7, rtol=0)
    checks.append('actual 80x80 input; historical normalized LL exact; DC amplitude and RGB range')

    pretrained = modules['model'].load_pretrained(inputs['config'], inputs['weights'], config['mcunet_root'])
    costs = []
    for variant in ('rgb160', 'rgb80', 'll80'):
        model = (base.RGB160(copy.deepcopy(pretrained)) if variant == 'rgb160'
                 else LowResolution80(copy.deepcopy(pretrained), variant)).eval()
        macs, stem_shapes, handles = [], [], []
        def capture(module, args, output):
            if isinstance(module, torch.nn.Conv2d):
                macs.append(output.numel() * module.in_channels // module.groups * math.prod(module.kernel_size))
            else:
                macs.append(output.numel() * module.in_features)
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(capture))
        handles.append(model.backbone.first_conv.conv.register_forward_hook(
            lambda module, args, out: stem_shapes.append(list(out.shape))))
        with torch.inference_mode():
            output = model(x[:1])
        assert tuple(output.shape) == (1, 1000) and torch.isfinite(output).all()
        expected_macs = 67287200 if variant == 'rgb160' else 18021824
        assert sum(macs) == expected_macs, (variant, sum(macs))
        assert sum(p.numel() for p in model.parameters()) == 729392
        assert all(torch.equal(v, pretrained.state_dict()[k]) for k, v in model.backbone.state_dict().items())
        for handle in handles:
            handle.remove()
        costs.append(dict(variant=variant, parameters=729392, conv_linear_macs=sum(macs),
                          stem_output=stem_shapes[0], fraction_of_rgb160=sum(macs)/67287200))
    base.atomic_json(ROOT / 'audit/compute.json', dict(rows=costs,
        scope='one image Conv/Linear MACs from forward output shapes; Haar/resize/BN/activation/IO excluded'))
    checks.append('unchanged 729392 pretrained parameters and stride2; measured 18.021824M vs 67.2872M CNN MAC')

    # Real CUDA update + failure rollback + successful resume replay. These
    # disposable admission weights never become a formal experiment checkpoint.
    assert torch.cuda.is_available()
    args.variant, args.augmentation = 'll80', 'light'
    parts = base.model_and_optimizers(args, inputs, modules)
    model, optimizer, scheduler, scaler = parts
    gpu_x = x.repeat(4, 1, 1, 1).cuda()
    gpu_y = torch.tensor(labels * 4, device='cuda')
    progress, contract = dict(step=0), dict(preflight=True, variant='ll80')
    initial = modules['runtime'].capture(*parts, progress, contract)
    def failure():
        raise RuntimeError('controlled CUDA rollback check')
    try:
        modules['runtime'].transactional_step(*parts, gpu_x, gpu_y, progress, contract,
                                             fail_after_forward=failure)
        raise AssertionError('injected failure did not propagate')
    except RuntimeError as error:
        assert str(error) == 'controlled CUDA rollback check'
    assert same(model.state_dict(), {k: v.cuda() for k, v in initial['model'].items()})
    modules['runtime'].transactional_step(*parts, gpu_x, gpu_y, progress, contract)
    first = modules['runtime'].capture(*parts, progress, contract)
    modules['runtime'].restore(initial, *parts, contract)
    modules['runtime'].transactional_step(*parts, gpu_x, gpu_y, progress, contract)
    replay = modules['runtime'].capture(*parts, progress, contract)
    for key in ('model', 'optimizer', 'scheduler', 'scaler'):
        assert same(first[key], replay[key]), key
    assert any(not torch.equal(v, initial['model'][k]) for k, v in first['model'].items())
    del parts, model, optimizer, scheduler, scaler, gpu_x, gpu_y, initial, first, replay
    torch.cuda.empty_cache()
    checks.append('batch32 CUDA FP32 forward/backward; failed-update BN rollback; exact successful-update resume replay')

    # A narrow GPU inference benchmark, not end-to-end ImageNet/MCU latency.
    timings = []
    for variant in ('rgb160', 'rgb80', 'll80'):
        model = (base.RGB160(copy.deepcopy(pretrained)) if variant == 'rgb160'
                 else LowResolution80(copy.deepcopy(pretrained), variant)).cuda().eval()
        for batch in (1, 32):
            gpu_input = x[:1].repeat(batch, 1, 1, 1).cuda()
            with torch.inference_mode():
                for _ in range(5):
                    model(gpu_input)
                torch.cuda.synchronize()
                measurements = []
                for _ in range(10):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    model(gpu_input)
                    end.record()
                    end.synchronize()
                    measurements.append(begin.elapsed_time(end))
            timings.append(dict(variant=variant, batch=batch,
                                median_ms=statistics.median(measurements),
                                mean_ms=statistics.mean(measurements)))
            del gpu_input
        del model
        torch.cuda.empty_cache()
    base.atomic_json(ROOT / 'audit/gpu_forward_latency.json', dict(
        device=torch.cuda.get_device_name(0), precision='FP32', tf32=False,
        scope='GPU resident RGB160 -> downsample/normalize -> CNN; excludes disk/decode/crop/CPU augmentation/transfers',
        caveat='10 samples after 5 warmups; laptop GPU frequency/load may vary; not MCU latency', rows=timings))
    checks.append('GPU forward latency measured separately from MAC; excludes decoding and CPU training augmentation')

    base.atomic_json(ROOT / 'preflight.json', dict(status='passed', checks=checks,
        sources=config['sources'], train_images=128000, dev_images=20000,
        manifest_sha256=inputs['manifest_sha256'], parameter_count=729392,
        conv_linear_macs_per_image=18021824, compute_reduction_percent=100*(1-18021824/67287200),
        augmentation=SETTINGS, formal_training_updates=0, disposable_cuda_updates=2,
        official_val_used=False, finished_at=time.time(), seconds=time.time()-started))
    print(json.dumps(dict(status='passed', checks=checks, costs=costs, timings=timings), indent=2), flush=True)


if __name__ == '__main__':
    main()
