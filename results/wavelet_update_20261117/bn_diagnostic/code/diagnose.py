"""Fixed-checkpoint, TRAIN-only BN recalibration; no parameter optimization."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'detail_sparsity_pad_mix_20261113'
sys.path.insert(0, str(SOURCE / 'code'))
import train_pad_mix as training
import w80_tar_source
from padded_input import PaddedMix160


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    training.atomic_json(path, value)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def bn_names(model):
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
                names.add(name + '.' + suffix)
    return names


def write_tensor(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    training.replace_with_retry(temporary, path)


def evaluate(model, inputs, config, name, views):
    folder = ROOT / 'outputs' / name
    folder.mkdir(exist_ok=True)
    complete = folder / 'metrics.json'
    if complete.exists():
        return read(complete)
    stats = {view: dict(images=0, correct1=0, correct5=0, ce_sum=0.0) for view in views}
    cursor = 0
    partial = folder / 'eval_progress.json'
    if partial.exists():
        previous = read(partial)
        stats, cursor = previous['totals'], previous['cursor']
    model.eval()
    original_state = state_digest(model.state_dict())
    with torch.inference_mode():
        batches = w80_tar_source.loader(inputs['loader_local'], inputs['dev_rows'], 'dev', cursor=cursor, workers=0)
        for images, labels in batches:
            images, target = images.cuda(), labels.cuda()
            assert len(target) == 32
            for view in views:
                model.eval_view = view
                logits = model(images)
                losses = torch.nn.functional.cross_entropy(logits, target, reduction='none')
                assert torch.isfinite(losses).all()
                pred = logits.topk(5, dim=1).indices
                totals = stats[view]
                totals['images'] += len(target)
                totals['correct1'] += int((pred[:, 0] == target).sum())
                totals['correct5'] += int((pred == target[:, None]).any(dim=1).sum())
                totals['ce_sum'] += float(losses.double().sum())
            cursor += len(target)
            if cursor % 1600 == 0 or cursor == 20000:
                publish(partial, dict(cursor=cursor, totals=stats))
                publish(ROOT / 'status.json', dict(status='evaluating', case=name, dev_images=cursor,
                    expected_dev_images=20000, updated_at=time.time(), worker_pid=os.getpid()))
                print(f'{name}: evaluated {cursor}/20000', flush=True)
    assert state_digest(model.state_dict()) == original_state, 'evaluation changed model buffers/parameters'
    metrics = {}
    for view, totals in stats.items():
        assert totals['images'] == 20000
        metrics[view] = dict(top1=100*totals['correct1']/20000,
                             top5=100*totals['correct5']/20000, ce=totals['ce_sum']/20000, **totals)
    publish(complete, dict(case=name, complete=True, metrics=metrics,
                          evaluation_state_digest=original_state, finished_at=time.time()))
    return read(complete)


def recalibrate(model, original, buffers, train_indices, inputs, name, view):
    folder = ROOT / 'outputs' / name
    folder.mkdir(exist_ok=True)
    final = folder / 'bn_buffers.pt'
    expected_nonbn = state_digest({k: v for k, v in original.items() if k not in buffers})
    model.load_state_dict(original, strict=True)
    model.eval()
    model.eval_view = view
    modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None  # equal-weight cumulative average over exactly500 batches
        module.train()
    cursor = 0
    partial = folder / 'calibration_state.pt'
    chosen = final if final.exists() else partial
    if chosen.exists():
        receipt = read(chosen.with_suffix('.json'))
        assert sha(chosen) == receipt['sha256']
        restored = torch.load(chosen, map_location='cpu', weights_only=True)
        assert restored['view'] == view and restored['original_nonbn_digest'] == expected_nonbn
        current = model.state_dict()
        for key, value in restored['bn_buffers'].items():
            assert key in buffers
            current[key].copy_(value)
        cursor = restored['cursor']
    if not final.exists():
        # Role=train means the same frozen RandomResizedCrop/flip as training.
        # The fixed list was sealed before any development evaluation.
        with torch.inference_mode():
            batches = w80_tar_source.loader(inputs['loader_local'], inputs['train_rows'], 'train',
                        epoch=0, order=train_indices, cursor=cursor, workers=0)
            for images, _ in batches:
                assert len(images) == 32
                model(images.cuda())
                cursor += len(images)
                if cursor % 1600 == 0 or cursor == 16000:
                    nonbn = state_digest({k:v for k,v in model.state_dict().items() if k not in buffers})
                    assert nonbn == expected_nonbn
                    state = dict(view=view, cursor=cursor, original_nonbn_digest=expected_nonbn,
                        bn_buffers={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in buffers})
                    target = final if cursor == 16000 else partial
                    write_tensor(target, state)
                    publish(target.with_suffix('.json'), dict(sha256=sha(target), view=view,
                        cursor=cursor, calibration_role='TRAIN only; labels ignored',
                        bn_reset=True, momentum=None, original_nonbn_digest=expected_nonbn))
                    publish(ROOT / 'status.json', dict(status='calibrating_train_only', case=name,
                        calibration_images=cursor, expected_calibration_images=16000,
                        worker_pid=os.getpid(), updated_at=time.time()))
                    print(f'{name}: calibrated {cursor}/16000 TRAIN images', flush=True)
    assert cursor == 16000
    assert all(int(m.num_batches_tracked) == 500 for m in modules)
    assert state_digest({k:v for k,v in model.state_dict().items() if k not in buffers}) == expected_nonbn
    model.eval()
    return sha(final)


def main():
    config = read(SOURCE / 'queue_config.json')
    modules = training.load_protocol_modules(config['handoff_root'])
    command = ['--variant', 'rgb_ll1_pad_mix']
    for arg in ('handoff_root', 'train_tar', 'seal_dir', 'weights', 'mcunet_root'):
        command += ['--' + arg.replace('_','-'), config[arg]]
    command += ['--run-dir', str(ROOT / 'unused_train_dir'),
                '--accept-local-content-sha256', config['manifest_sha256']]
    args = training.parse_args(command)
    inputs = training.verify_inputs(args, modules)
    # 16 examples/class, deterministic label-stratified TRAIN subset. No dev
    # scores, predictions or input images participate in this choice.
    by_label = {i: [] for i in range(1000)}
    for index, row in enumerate(inputs['train_rows']):
        assert row['role'] == 'train'
        if len(by_label[int(row['label'])]) < 16:
            by_label[int(row['label'])].append(index)
    indices = [by_label[label][j] for j in range(16) for label in range(1000)]
    order = torch.randperm(16000, generator=torch.Generator().manual_seed(2026)).tolist()
    indices = [indices[i] for i in order]
    selection = dict(samples=[dict(train_index=i, path=inputs['train_rows'][i]['path'],
                    label=int(inputs['train_rows'][i]['label']), sha256=inputs['train_rows'][i]['sha']) for i in indices],
                    images=16000, per_class=16, selection='first16 frozen TRAIN rows/class; seed2026 shuffle')
    sealed = ROOT / 'inputs/calibration_train_members.json'
    if sealed.exists():
        assert read(sealed) == selection
    else:
        publish(sealed, selection)
    epoch_files = {18:'8629d120a32790cce04a8184830b16f2784cfd6ab9fc7b269fd26919bd38262b.pt',
                   20:'2ec655ed73c55045f36f19d0e5fc0876eef42c19c550ce2be5cc89a9bf2ae2e0.pt'}
    contract = dict(schema=1, epochs=epoch_files, train_images=16000,
        calibration_members_sha256=sha(sealed), calibration_views=['rgb','ll1_pad'],
        calibration_transform='original train RRC160+flip at epoch0; separately each view',
        recalibration='reset BN stats, momentum=None,500 batches32; only BN modules train mode; no gradients/optimizer',
        evaluation='full20000 dev, both original routes; recalibrated model eval only on its corresponding route',
        no_parameter_changes=True, no_official_val=True, frozen_source_sha256=sha(Path(__file__)),
        source_run_config_sha256=sha(SOURCE/'runs/p2_rgb_ll1_pad_mix/config.json'))
    target = ROOT / 'diagnostic_contract.json'
    if target.exists():
        assert read(target) == contract
    else:
        publish(target, contract)
    modules['execution'].setup(2026)
    backbone = modules['model'].load_pretrained(inputs['config'], inputs['weights'], config['mcunet_root'])
    model = PaddedMix160(backbone, 'rgb_ll1_pad_mix').cuda().eval()
    model.requires_grad_(False)
    buffers = bn_names(model)
    rows = []
    for epoch, filename in epoch_files.items():
        checkpoint = ROOT / 'inputs' / filename
        assert sha(checkpoint) == checkpoint.stem
        # Only copied local hash-verified checkpoints, never uploaded code/data.
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['progress']['epoch'] == epoch
        original = state['model']
        model.load_state_dict(original, strict=True)
        original_metrics = evaluate(model, inputs, config, f'epoch{epoch}_original', ['rgb','ll1_pad'])
        for view in ('rgb','ll1_pad'):
            rows.append(dict(epoch=epoch, condition='original', view=view,
                top1=original_metrics['metrics'][view]['top1'],
                top5=original_metrics['metrics'][view]['top5'], ce=original_metrics['metrics'][view]['ce']))
        for view in ('rgb','ll1_pad'):
            name = f'epoch{epoch}_{view}_bn_recalibrated'
            digest = recalibrate(model, original, buffers, indices, inputs, name, view)
            metrics = evaluate(model, inputs, config, name, [view])
            rows.append(dict(epoch=epoch, condition='train_only_bn_recalibrated', view=view,
                top1=metrics['metrics'][view]['top1'], top5=metrics['metrics'][view]['top5'],
                ce=metrics['metrics'][view]['ce']))
            metrics['bn_buffer_sha256'] = digest
            publish(ROOT / 'outputs' / name / 'metrics.json', metrics)
        # Raw evaluation must reproduce the original history; no hidden model or
        # processing change may explain an apparent recalibration improvement.
        with (SOURCE/'runs/p2_rgb_ll1_pad_mix/history.csv').open(newline='') as stream:
            historic = next(row for row in csv.DictReader(stream) if int(row['epoch']) == epoch)
        for view, prefix in [('rgb','rgb_dev'), ('ll1_pad','dev')]:
            for public, suffix in [('top1','top1'),('top5','top5'),('ce','ce')]:
                assert abs(original_metrics['metrics'][view][public]-float(historic[f'{prefix}_{suffix}'])) < 1e-6
        publish(ROOT / 'results.json', dict(status='in_progress' if epoch == 18 else 'complete',
            results=rows, all_nonbn_tensors_unchanged=True, original_metrics_reproduced=True,
            calibration_train_only=True, developer_set_not_used_for_bn=True,
            experiment='posthoc diagnostic; does not replace original experiment', updated_at=time.time()))
        with (ROOT / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    publish(ROOT / 'status.json', dict(status='complete', worker_pid=None,
        finished_at=time.time(), results=rows, no_optimizer_steps=True))
    print(json.dumps(dict(status='complete', results=rows), indent=2), flush=True)


if __name__ == '__main__':
    with training.run_lock(ROOT / 'diagnostic_run'):
        try:
            main()
        except Exception as error:
            publish(ROOT/'status.json', dict(status='failed',error=repr(error),updated_at=time.time(),worker_pid=os.getpid()))
            raise
