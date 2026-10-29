"""Verify and archive the completed 20k pilot without modifying training artifacts."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError('Use a new export directory; do not overwrite prior evidence')
    import torch
    from experiments.w80_pilot.atomic_io import read_latest_snapshot
    from experiments.w80_pilot.data import MirrorImages
    from experiments.w80_pilot.vendor.data import digest
    torch.set_num_threads(2)
    run = ROOT / 'runs/two_route_pilot20k'
    shared = ROOT / 'runs/w80_e23_pilot20k_v2'
    state = read_latest_snapshot(run / 'queue_status.json')
    if state.get('stage') != 'complete':
        raise RuntimeError('Queue is not complete')
    config = read(run / 'queue.json')
    manifest = read(shared / 'data/manifest.json')
    if Counter(r['role'] for r in manifest['rows']) != {'train': 16000, 'dev': 4000}:
        raise RuntimeError('Unexpected pilot membership')
    dev_labels = {r['path']: r['label'] for r in manifest['rows'] if r['role'] == 'dev'}
    train_paths = {r['path'] for r in manifest['rows'] if r['role'] == 'train'}
    if train_paths & set(dev_labels):
        raise RuntimeError('Train/dev overlap')
    selection = read(shared / 'channel_selection.json')
    if len(set(manifest['calibration'])) != 1024 or not set(manifest['calibration']) <= train_paths:
        raise RuntimeError('Calibration is not 1024 TRAIN-only identities')
    if selection['indices_sha256'] != digest(selection['indices']):
        raise RuntimeError('Channel indices checksum mismatch')
    remote = args.review / 'experiments/w80_e23'
    frozen = read(remote / 'protocol_v1_frozen.json')
    if sha(remote / 'model.py') != sha(ROOT / 'experiments/w80_pilot/vendor/model.py'):
        raise RuntimeError('Model differs from the reviewed teammate version')
    expected = dict(seed=2026, precision='fp32', tf32=False, micro_batch=32, bn_batch=32,
        accumulation=1, momentum=.9, weight_decay=4e-5, backbone_head_peak_lr=.001,
        new_stem_peak_lr=.005, warmup_epochs=2)
    arms = [(j['job'], run / j['job'] / j['variant']) for j in config['jobs']]
    arms.append(('reused_rgb_e23', shared / 'rgb_e23'))
    out.mkdir(parents=True)
    result_rows, artifacts, required_sources = [], [], {}
    for name, arm in arms:
        summary = read(arm / 'summary.json')
        contract = read(arm / 'config.json')
        for key, value in expected.items():
            if name == 'reused_rgb_e23' and key == 'seed' and key not in contract:
                if 'random.seed(2026)' not in (ROOT / 'scripts/train_w80_pilot.py').read_text():
                    raise RuntimeError('Reused RGB seed missing from recorded source')
                continue
            if contract[key] != value:
                raise RuntimeError(name + ': training contract differs at ' + key)
        if contract['manifest_sha256'] != sha(shared / 'data/manifest.json') or contract['selection_sha256'] != sha(shared / 'channel_selection.json'):
            raise RuntimeError(name + ': data/calibration differs')
        required_sources.update({(path, value): None for path, value in contract['source_files'].items()})
        with (arm / 'history.csv').open(newline='') as stream:
            history = list(csv.DictReader(stream))
        if [int(row['epoch']) for row in history] != list(range(1, 21)):
            raise RuntimeError(name + ': missing or duplicate epoch')
        if any(int(row['successful_updates']) != int(row['epoch']) * 500 for row in history):
            raise RuntimeError(name + ': update budget mismatch')
        best = min(history, key=lambda row: float(row['dev_ce']))
        if int(best['epoch']) != summary['selected_epoch'] or float(best['dev_ce']) != summary['dev_ce']:
            raise RuntimeError(name + ': best is not strict minimum complete-dev CE')
        correct1 = correct5 = 0
        seen = set()
        with (arm / 'dev_predictions.csv').open(newline='') as stream:
            for prediction in csv.DictReader(stream):
                image = prediction['image_id']
                label = int(prediction['true_label'])
                ranks = list(map(int, prediction['pred_top5'].split()))
                if image in seen or dev_labels.get(image) != label or len(set(ranks)) != 5 or any(not 0 <= i < 1000 for i in ranks):
                    raise RuntimeError(name + ': invalid development prediction')
                seen.add(image)
                correct1 += ranks[0] == label
                correct5 += label in ranks
        if seen != set(dev_labels) or correct1 / 40 != summary['dev_top1'] or correct5 / 40 != summary['dev_top5']:
            raise RuntimeError(name + ': per-image predictions do not reproduce accuracy')
        checkpoints = []
        for kind in ('best', 'last'):
            path = arm / (kind + '.pt')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['contract'] != contract or 'optimizer' not in checkpoint or 'rng' not in checkpoint:
                raise RuntimeError(name + ': checkpoint contract/state mismatch')
            if kind == 'last' and (checkpoint['progress']['successful_updates'] != 10000 or not checkpoint['progress']['completed']):
                raise RuntimeError(name + ': incomplete last checkpoint')
            if kind == 'best' and checkpoint['progress']['best_epoch'] != summary['selected_epoch']:
                raise RuntimeError(name + ': incorrect best checkpoint')
            checkpoints.append(dict(kind=kind, sha256=sha(path), bytes=path.stat().st_size,
                distribution='retained locally; weights not uploaded to Git',
                contains=['model', 'BN', 'optimizer', 'scheduler', 'scaler', 'RNG', 'progress', 'contract']))
            del checkpoint
        target = out / 'artifacts' / name
        for filename in ('summary.json', 'config.json', 'history.csv', 'train.log', 'dev_predictions.csv'):
            copy(arm / filename, target / filename)
        write(target / 'checkpoints.json', checkpoints)
        route = summary.get('route', 'w80_e23')
        variant = summary['variant']
        macs = (47923200 if name == 'reused_rgb_e23' else
                (45638400 if variant == 'll' else 46118400) if route == 'w80_e23' else
                (18021824 if variant in ('ll', 'rgb80') else 18713024))
        result_rows.append(dict(job=name, route=route, variant=variant,
            train_images=16000, dev_images=4000, completed_epochs=20,
            selected_epoch=summary['selected_epoch'], dev_ce=summary['dev_ce'],
            dev_top1=summary['dev_top1'], dev_top5=summary['dev_top5'],
            conv_linear_macs=macs, committed_updates=10000, committed_exposures=320000,
            elapsed_seconds=summary['elapsed_seconds'], official_val_top1='', official_val_top5='',
            scope='preliminary resized-mirror dev only; single seed2026'))
        artifacts.append(dict(job=name, minimum_dev_ce_verified=True, prediction_images=len(seen),
            predictions_reproduce_top1_top5=True, checkpoints=checkpoints,
            final_train_top1=float(history[-1]['train_top1']), final_dev_top1=float(history[-1]['dev_top1']),
            final_train_ce=float(history[-1]['train_ce']), final_dev_ce=float(history[-1]['dev_ce'])))
    with (out / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    for filename in ('manifest.json', 'samples.csv'):
        copy(shared / 'data' / filename, out / 'data' / filename)
    copy(shared / 'channel_selection.json', out / 'data/channel_selection.json')
    mapping = ROOT / 'reports/protocol_alignment_20261028/classes_canonical.json'
    copy(mapping, out / 'data/classes_canonical.json')
    for filename in ('route_cuda_checks.json', 'monitor_fix_migration.json', 'snapshot_io_migration.json',
                     'unattended_migration.json', 'unattended_verification.json', 'queue.json', 'reused_rgb_e23.json'):
        copy(run / filename, out / 'evidence' / filename)
    copy(shared / 'cuda_checks.json', out / 'evidence/reused_rgb_cuda_checks.json')
    copy(shared / 'protocol.json', out / 'evidence/reused_rgb_protocol.json')
    copy(run / 'dev_curves.png', out / 'figures/dev_curves.png')
    for filename in ('protocol_v1_frozen.json', 'protocol.json'):
        copy(remote / filename, out / 'alignment/teammate' / filename)
    copy(args.review / 'download_receipt.json', out / 'alignment/teammate/download_receipt.json')
    source_paths = [p.relative_to(ROOT) for p in (ROOT / 'experiments/w80_pilot').rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.json')]
    source_paths += [Path('scripts') / name for name in (
        'train_route_pilot.py', 'run_route_queue.py', 'prepare_w80_pilot.py',
        'check_route_pilot_cuda.py', 'check_w80_pilot_cuda.py', 'train_w80_pilot.py',
        'plot_route_queue.py', 'watch_route_queue.py', 'download_weights.py',
        'prepare_imagenet_subset.py', 'export_two_route_pilot.py')]
    source_paths += [Path('tests') / name for name in ('test_route_io.py', 'test_unattended_recovery.py')]
    source_paths += [Path('weights/mcunet-256kb-1mb_imagenet.json')]
    for path in source_paths:
        copy(ROOT / path, out / 'code' / path)
    # Capture every historical source version referenced by exported run contracts.
    candidates = list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'experiments/w80_pilot').rglob('*.py'))
    candidates += [p for p in (ROOT / 'reports').rglob('*.py') if '__pycache__' not in p.parts]
    lookup = {sha(p): p for p in candidates}
    source_receipts = []
    for (path, checksum) in required_sources:
        source = lookup.get(checksum)
        if source is None:
            source_receipts.append(dict(original_path=path, sha256=checksum, archived_path=None,
                exact_historical_source_available=False))
            continue
        target = Path('source_history') / checksum / Path(path).name
        copy(source, out / target)
        source_receipts.append(dict(original_path=path, sha256=checksum, archived_path=target.as_posix(),
            exact_historical_source_available=True))
    write(out / 'evidence/source_contract_index.json', source_receipts)
    # Confirm the loader discrepancy, without changing any training source or state.
    rng_check = dict(cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.init(); before = torch.cuda.get_rng_state().clone()
        cpu_before = torch.get_rng_state().clone()
        row = next(r for r in manifest['rows'] if r['role'] == 'train')
        dataset = MirrorImages(ROOT.parent / 'datasets/imagenet-1k-256-subset200/train', [row], 'train')
        dataset[0]
        rng_check.update(cpu_rng_preserved=torch.equal(cpu_before, torch.get_rng_state()),
                         cuda_rng_changed_by_worker0_loader=not torch.equal(before, torch.cuda.get_rng_state()))
        torch.cuda.set_rng_state(before); torch.set_rng_state(cpu_before)
    shared_keys = {key: dict(local=value, teammate=frozen.get(key), equal=value == frozen.get(key))
                   for key, value in expected.items()}
    if not all(value['equal'] for value in shared_keys.values()):
        raise RuntimeError('Training recipe differs from the reviewed frozen protocol')
    audit = dict(as_of_utc=datetime.now(timezone.utc).isoformat(),
        classification='preliminary_validation_not_formal_protocol_match',
        teammate_repository='LumiaBlack51/tinyml-progressive-inference',
        teammate_branch='experiment/w80-e23-imagenet-20261028',
        teammate_commit='c8b4f13cbe64d19b7f404f61a6213aea541226f2',
        local_implementation_parent='ce31c1da28c8f41d81ae01d64fd30c6ad96cf439',
        model_file_byte_identical_to_latest=True, training_recipe=shared_keys,
        pretrained_sha256=selection['contract']['pretrained_sha256'],
        manifest_sha256=sha(shared / 'data/manifest.json'),
        classes_sha256=sha(mapping), indices_file_sha256=sha(shared / 'channel_selection.json'),
        indices_content_sha256=selection['indices_sha256'],
        calibration_count=1024, calibration_is_train_only=True, all_local_w80_and_rgb_use_same_indices=True,
        exact_reusable_for_teammate_formal_rgb=False,
        reused_rgb_seed_record='2026 fixed in hash-indexed trainer source; seed field absent in its older JSON contract',
        differences=dict(data='256px center-cropped mirror versus original TRAIN; membership and identity exclusions differ',
            size='16000/4000 versus 128000/20000', updates='500/epoch,10000 total versus 4000/epoch,80000 maximum',
            calibration='same algorithm/count, different image cohort, no shared production indices',
            dependencies=dict(local=read(shared/'protocol.json')['versions'],teammate=frozen['execution_environment']),
            determinism='local cudnn deterministic only; teammate deterministic_algorithms and CUBLAS_WORKSPACE_CONFIG',
            loader_rng='local torch.manual_seed touches CUDA at workers0; latest teammate seeds CPU generator only',
            dev_ce_reduction='local FP32 sum per batch; latest teammate sums per-image CE in FP64',
            operations='local snapshots/bounded restart/no RAM pause versus teammate watchdog/hysteresis/physical attempt ledger',
            scope='extra spatial_s2/RGB80 route is local pilot; not teammate W80 protocol'),
        loader_rng_check=rng_check, all_arms=artifacts,
        missing_historical_source_versions=[row for row in source_receipts if not row['exact_historical_source_available']],
        earlier_replay='At least50 extra physical updates/1600 exposures in first spatial_LL recovery; exact total uncommitted replay unknown. Counters are committed logical work.',
        online_materials='code, curves, histories, logs, predictions, manifests, indices, hashes; binary checkpoints retained locally',
        official_val_or_v2_used=False, no_new_training_started=True)
    write(out / 'alignment/audit.json', audit)
    upstream = subprocess.run(['git','-C',str(Path.home()/'Desktop/Mcunet/mcunet'),'rev-parse','HEAD'],capture_output=True,text=True)
    write(out / 'environment.json', dict(recorded_training_versions=read(shared/'protocol.json')['versions'],
        export_torch=torch.__version__, gpu='NVIDIA GeForce RTX 3060 Laptop GPU',
        upstream_mcunet_commit=upstream.stdout.strip() if upstream.returncode == 0 else None,
        precision='FP32, TF32 off', seed=2026, cpu_threads=4, workers=0))
    print(json.dumps(dict(export=str(out),arms=len(arms),checkpoint_files_verified=20,
        dev_predictions_verified=40000,source_versions=len(source_receipts),loader_rng_check=rng_check)))


if __name__ == '__main__':
    main()
