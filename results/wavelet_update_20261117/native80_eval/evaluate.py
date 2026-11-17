"""Read-only native LL80 development evaluation of completed ratio runs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import torch
import torch.nn.functional as F
import train_ratio as train
from ratio_input import corrected_ll
import w80_tar_source


def cnn_macs(backbone, size, device):
    total = [0]
    hooks = []
    def count(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            total[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            total[0] += output.numel() * module.in_features
    try:
        for module in backbone.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(count))
        with torch.inference_mode():
            backbone(torch.zeros(1, 3, size, size, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    return total[0]


def evaluate(arm, config, modules, device):
    run = ROOT / 'runs' / arm['name']
    complete = train.read_json(run / 'complete.json')
    pointers = train.read_json(run / 'checkpoints/checkpoints.json')
    checkpoint = run / 'checkpoints' / pointers['best']
    checkpoint_hash = train.sha256(checkpoint)
    if checkpoint_hash != checkpoint.stem:
        raise ValueError('checkpoint hash mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selected_epoch = train.selected_epoch(state['progress']['history'])
    assert selected_epoch == complete['best_epoch']
    model_config = modules['root'].parent / 'mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json'
    backbone = modules['model'].load_pretrained(model_config, config['weights'], config['mcunet_root'])
    prefix = 'backbone.'
    assert all(key.startswith(prefix) for key in state['model'])
    backbone.load_state_dict({key[len(prefix):]: value for key, value in state['model'].items()}, strict=True)
    del state
    backbone.to(device).eval()
    bn_before = {key: value.detach().clone() for key, value in backbone.state_dict().items()
                 if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    macs80 = cnn_macs(backbone, 80, device)
    macs160 = cnn_macs(backbone, 160, device)
    assert macs160 == 67287200
    assert train.sha256(config['manifest']) == config['manifest_sha256']
    rows = [row for row in modules['data'].read_manifest(config['manifest']) if row['role'] == 'dev']
    assert len(rows) == 20000
    local = dict(archive=config['train_tar'], index_path=str(Path(config['seal_dir']) / 'tar_index.tsv'))
    count = top1 = top5 = 0
    loss_sum = 0.
    started = time.time()
    with torch.inference_mode():
        for images, labels in w80_tar_source.loader(local, rows, 'dev', workers=0, prefetch=1):
            native = corrected_ll(images.to(device)) * 2 - 1
            assert tuple(native.shape[1:]) == (3, 80, 80)
            target = labels.to(device)
            logits = backbone(native)
            losses = F.cross_entropy(logits, target, reduction='none')
            assert torch.isfinite(losses).all()
            pred = logits.topk(5, dim=1).indices
            loss_sum += float(losses.double().sum())
            top1 += int((pred[:, 0] == target).sum())
            top5 += int((pred == target[:, None]).any(dim=1).sum())
            count += len(labels)
            if count % 1024 == 0 or count == 20000:
                train.atomic_json(OUT / 'progress.json', dict(status='evaluating', variant=arm['name'],
                    images=count, total_images=20000, checkpoint_epoch=selected_epoch, updated_at=time.time()))
                print(f'{arm["name"]}: {count}/20000 native LL80', flush=True)
    assert count == 20000
    for key, value in bn_before.items():
        assert torch.equal(value, backbone.state_dict()[key]), 'BN changed during evaluation'
    with (run / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
        row = next(row for row in csv.DictReader(stream) if int(row['epoch']) == selected_epoch)
    result = dict(variant=arm['name'], checkpoint_epoch=selected_epoch,
        checkpoint_sha256=checkpoint_hash, checkpoint_selection='existing minimum full LL-pad160 dev CE; no reselection',
        native_ll80=dict(images=count, top1=100*top1/count, top5=100*top5/count, ce=loss_sum/count,
                         input_shape=[3, 80, 80], conv_linear_macs=macs80),
        existing_ll_pad160=dict(top1=float(row['dev_top1']), top5=float(row['dev_top5']), ce=float(row['dev_ce'])),
        existing_rgb160=dict(top1=float(row['rgb_dev_top1']), top5=float(row['rgb_dev_top5']), ce=float(row['rgb_dev_ce'])),
        reference160_conv_linear_macs=macs160, native_to_160_mac_ratio=macs80/macs160,
        mac_scope='CNN Conv/Linear only; excludes decoding, geometric preprocessing, Haar and normalization',
        bn='shared checkpoint running statistics; eval mode; no recalibration; buffers verified unchanged',
        preprocessing='same dev Resize182 bicubic antialias + CenterCrop160 -> Haar LL1 / 2 DC correction -> 2*x-1; no padding or upsampling',
        precision='FP32; AMP and TF32 disabled', manifest_sha256=config['manifest_sha256'],
        official_val_used=False, imagenet_v2_used=False, seconds=time.time()-started,
        evaluator_sha256=train.sha256(__file__), finished_at=time.time())
    train.atomic_json(OUT / (arm['name'] + '.json'), result)
    print(json.dumps(result, indent=2), flush=True)
    del backbone, bn_before
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true', help='evaluate remaining arms after each completes')
    args = parser.parse_args()
    config = train.read_json(ROOT / 'queue_config.json')
    modules = train.load_protocol_modules(config['handoff_root'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    pending = list(config['experiments'])
    while pending:
        for arm in pending[:]:
            result_path = OUT / (arm['name'] + '.json')
            if result_path.exists():
                pending.remove(arm)
            elif (ROOT / 'runs' / arm['name'] / 'complete.json').exists():
                evaluate(arm, config, modules, device)
                pending.remove(arm)
        results = [train.read_json(OUT / (arm['name'] + '.json')) for arm in config['experiments']
                   if (OUT / (arm['name'] + '.json')).exists()]
        train.atomic_json(OUT / 'results.json', dict(status='waiting_for_training' if pending else 'complete',
                          results=results, pending=[arm['name'] for arm in pending]))
        train.atomic_json(OUT / 'progress.json', dict(status='waiting_for_training' if pending else 'complete',
                          pending=[arm['name'] for arm in pending], updated_at=time.time()))
        if not args.watch:
            break
        if pending:
            time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        train.atomic_json(OUT / 'progress.json', dict(status='failed', error=repr(error), updated_at=time.time()))
        raise
