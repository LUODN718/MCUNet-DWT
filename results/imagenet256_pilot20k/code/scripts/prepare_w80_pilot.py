#!/usr/bin/env python3
"""Prepare the resized-image pilot and calibrate shared channels on TRAIN only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.w80_pilot.data import MirrorImages, prepare
from experiments.w80_pilot.vendor import model as w80
from experiments.w80_pilot.vendor.data import digest, sha


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/w80_e23_pilot20k')
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'datasets/imagenet-1k-256-subset200/train')
    parser.add_argument('--mcunet-root', type=Path, default=Path.home() / 'Desktop/Mcunet/mcunet')
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA environment is required for this production-data pilot calibration')
    expected = '63565b5773a8aa70d75baf5a349f5f384da649804c48a5e7af1f655460158c7d'
    weights = ROOT / 'weights/mcunet-256kb-1mb_imagenet.pth'
    if sha(weights) != expected:
        raise ValueError('pretrained weight checksum mismatch')
    membership = ROOT / 'runs/imagenet256_compute_s2/train_dev_seed2026.samples.csv'
    manifest = prepare(args.data_root, membership, args.run_root / 'data')
    training = [r for r in manifest['rows'] if r['role'] == 'train']
    development = [r for r in manifest['rows'] if r['role'] == 'dev']
    if len(training) != 16000 or len(development) != 4000:
        raise ValueError('pilot split size mismatch')
    by_path = {r['path']: r for r in training}
    calibration_rows = [by_path[p] for p in manifest['calibration']]
    if len(calibration_rows) != 1024 or set(by_path) & {r['path'] for r in development}:
        raise ValueError('invalid calibration or overlapping train/dev')
    dataset = MirrorImages(args.data_root, calibration_rows, 'calibration')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    contract = dict(scope=manifest['scope'], source_commit='ce31c1da28c8f41d81ae01d64fd30c6ad96cf439',
                    manifest_sha256=sha(args.run_root / 'data/manifest.json'),
                    calibration_sha256=manifest['calibration_sha256'], pretrained_sha256=expected,
                    config_sha256=sha(ROOT / 'weights/mcunet-256kb-1mb_imagenet.json'),
                    calibration_images=1024, calibration_role='TRAIN only', precision='FP32', tf32=False,
                    original_imagenet_protocol_admitted=False)
    selection_path = args.run_root / 'channel_selection.json'
    base = w80.load_pretrained(ROOT / 'weights/mcunet-256kb-1mb_imagenet.json', weights, args.mcunet_root)
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection['contract'] != contract or selection['indices_sha256'] != digest(selection['indices']):
            raise ValueError('existing channel calibration differs from the pilot contract')
        print('Reusing verified fixed pilot channel selection.', flush=True)
    else:
        before = {k: v.clone() for k, v in base.state_dict().items()}
        print('Calibrating once on 1024 fixed TRAIN mirror images, FP32, batch32...', flush=True)
        scores = w80.calibration_scores(base.cuda(), loader, expected=1024)
        base.cpu()
        if any(not torch.equal(v, base.state_dict()[k]) for k, v in before.items()):
            raise RuntimeError('calibration changed pretrained model or BN state')
        indices = w80.select_indices(base, scores)
        selection = dict(contract=contract, scores=scores, indices=indices, indices_sha256=digest(indices),
                         blocks=w80.block_audit(base), synthetic_indices=False)
        write_json(selection_path, selection)
        del before
    torch.cuda.empty_cache()
    costs = {}
    for variant in w80.VARIANTS:
        net = w80.make_variant(base, selection['indices'], variant).eval()
        costs[variant] = w80.cost_audit(net)
        print(variant, 'parameters', costs[variant]['parameters'], 'Conv/Linear MACs', costs[variant]['conv_linear_macs'], flush=True)
        del net
    receipt = dict(contract=contract, selection_file_sha256=sha(selection_path),
                   indices_sha256=selection['indices_sha256'], train_images=16000, dev_images=4000,
                   updates_per_epoch=500, maximum_epochs=20, maximum_updates_per_arm=10000,
                   maximum_training_exposures_per_arm=320000, batch=32, variants=list(w80.VARIANTS),
                   costs=costs, training_started=False, official_validation_used=False,
                   next_required='CUDA batch32 recovery/failure tests and a pilot training entry point')
    write_json(args.run_root / 'preparation.json', receipt)
    print('Pilot data and shared channel calibration prepared:', args.run_root, flush=True)


if __name__ == '__main__':
    main()
