"""CPU-only admission: input identity, Haar semantics, model cost and resume."""
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import train_ll as base
from direct_io import UncachedReader
from lowpass import LL160, lowpass_view

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = base.read_json(ROOT / 'queue_config.json')
    for relative, checksum in config['sources'].items():
        assert base.sha256(ROOT / relative) == checksum, relative
    commands = []
    for level in (1, 2, 3):
        # Build the same frozen-input arguments the future worker will receive.
        command = ['--level', str(level)]
        for name in ('handoff_root', 'train_tar', 'seal_dir', 'weights', 'mcunet_root'):
            command += ['--' + name.replace('_', '-'), config[name]]
        command += ['--run-dir', str(Path(config['run_root']) / f'll{level}'),
                    '--accept-local-content-sha256', config['manifest_sha256']]
        commands.append(command)
    modules = base.load_protocol_modules(config['handoff_root'])
    inputs = base.verify_inputs(base.parse_args(commands[0]), modules)
    assert len(inputs['train_rows']) == 128000 and len(inputs['dev_rows']) == 20000
    assert inputs['manifest_sha256'] == config['manifest_sha256']
    checks = ['frozen pretrained/config/upstream/environment and 128k/20k membership']

    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2026)
    x = torch.rand((2, 3, 160, 160), generator=generator)
    for level in (1, 2, 3):
        actual = lowpass_view(x, level)
        expected = F.interpolate(F.avg_pool2d(x, 2**level), (160, 160), mode='bilinear', align_corners=False)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
        constant = lowpass_view(torch.full_like(x, .37), level)
        torch.testing.assert_close(constant, torch.full_like(x, .37), atol=1e-7, rtol=0)
        assert tuple(actual.shape) == (2, 3, 160, 160)
        assert actual.min() >= 0 and actual.max() <= 1
        grad_input = x.clone().requires_grad_(True)
        lowpass_view(grad_input, level).square().mean().backward()
        assert torch.isfinite(grad_input.grad).all() and grad_input.grad.abs().sum() > 0
    checks.append('LL1/2/3 amplitude, dimensions, average-pool equivalence and gradient')

    # Direct I/O must return exactly the same bytes as conventional reads.
    import w80_tar_source
    entries = w80_tar_source.load_index(Path(config['seal_dir']) / 'tar_index.tsv')
    sampled = inputs['train_rows'][:8] + inputs['dev_rows'][:8]
    reader = UncachedReader(config['train_tar'])
    with open(config['train_tar'], 'rb') as stream:
        for row in sampled:
            offset, size = entries[row['path']]
            raw = reader.read(offset, size)
            stream.seek(offset)
            assert raw == stream.read(size)
            assert hashlib.sha256(raw).hexdigest() == row['sha']
    reader.close()
    dataset = w80_tar_source.TarOriginalImages(config['train_tar'],
              Path(config['seal_dir']) / 'tar_index.tsv', inputs['train_rows'], 'train', epoch=3)
    image1, label1 = dataset[0]
    image2, label2 = dataset[0]
    assert torch.equal(image1, image2) and label1 == label2
    assert tuple(image1.shape) == (3, 160, 160)
    dataset.close()
    checks.append('16 JPEG direct-read identity checks, decoded RGB hash, deterministic augmented view')

    backbone = modules['model'].load_pretrained(inputs['config'], inputs['weights'], config['mcunet_root'])
    assert sum(p.numel() for p in backbone.parameters()) == 729392
    assert tuple(backbone.first_conv.conv.stride) == (2, 2)
    assert backbone.first_conv.conv.in_channels == 3
    macs = []
    hooks = []
    def capture(module, args, output):
        if isinstance(module, torch.nn.Conv2d):
            import math
            macs.append(output.numel() * module.in_channels // module.groups * math.prod(module.kernel_size))
        else:
            macs.append(output.numel() * module.in_features)
    for module in backbone.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(capture))
    model = LL160(backbone, 1).eval()
    with torch.inference_mode():
        logits = model(image1.unsqueeze(0))
    assert tuple(logits.shape) == (1, 1000) and torch.isfinite(logits).all()
    assert sum(macs) == 67287200, sum(macs)
    for hook in hooks:
        hook.remove()
    original_keys = set(model.state_dict())
    for level in (2, 3):
        model = LL160(backbone, level).eval()
        assert set(model.state_dict()) == original_keys
        with torch.inference_mode():
            assert torch.isfinite(model(x[:1])).all()
    checks.append('unchanged original stem/backbone/head, 729392 parameters, 67287200 CNN MACs')

    # A failed update after BN ran must restore weights, BN, optimizer and RNG.
    # Then a successful update must replay identically after checkpoint restore.
    model = LL160(backbone, 2).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, modules['runtime'].lr_factor)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    parts = (model, optimizer, scheduler, scaler)
    progress, contract = dict(step=0), dict(level=2, preflight=True)
    state = modules['runtime'].capture(*parts, progress, contract)
    def injected():
        raise RuntimeError('controlled preflight rollback')
    try:
        modules['runtime'].transactional_step(*parts, x, torch.tensor([1, 2]), progress, contract,
                                             fail_after_forward=injected)
        raise AssertionError('injected failure did not propagate')
    except RuntimeError as error:
        assert str(error) == 'controlled preflight rollback'
    assert all(torch.equal(v, state['model'][k]) for k, v in model.state_dict().items())
    modules['runtime'].transactional_step(*parts, x, torch.tensor([1, 2]), progress, contract)
    first = modules['runtime'].capture(*parts, progress, contract)
    modules['runtime'].restore(state, *parts, contract)
    modules['runtime'].transactional_step(*parts, x, torch.tensor([1, 2]), progress, contract)
    second = modules['runtime'].capture(*parts, progress, contract)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    for name in ('model', 'optimizer', 'scheduler', 'scaler'):
        assert equal(first[name], second[name]), name
    checks.append('CPU model/BN failure rollback and exact optimizer/scheduler update replay')

    base.atomic_json(ROOT / 'preflight.json', dict(status='passed', device='cpu',
        cuda_training_updates=0, official_val_used=False, checks=checks, sources=config['sources'],
        train_images=128000, dev_images=20000, levels=[1, 2, 3], parameter_count=729392,
        conv_linear_macs=67287200, manifest_sha256=inputs['manifest_sha256'], finished_at=time.time()))
    print(json.dumps(dict(status='passed', checks=checks), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
