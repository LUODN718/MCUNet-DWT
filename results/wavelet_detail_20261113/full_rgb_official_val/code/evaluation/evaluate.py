"""Evaluate the fixed epoch20 RGB160 checkpoint on all 50,000 original val JPEGs."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent
sys.path.insert(0, str(EXPERIMENT / 'code'))
import baseline_helpers as base
from full_data import UncachedReader


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def publish(path, value):
    base.atomic_json(path, value)


def build_index(archive, labels, classes):
    receipt = ROOT / 'data/tar_index.json'
    identity = dict(bytes=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns)
    if receipt.exists():
        index = read(receipt)
        if index['archive'] != identity:
            raise ValueError('validation archive changed')
        return index['images']
    class_id = {name: i for i, name in enumerate(classes)}
    entries = {}
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in labels or name in entries or member.size <= 0:
                raise ValueError('unexpected or duplicate validation member: ' + name)
            entries[name] = dict(name=name, offset=member.offset_data, bytes=member.size,
                                 label=class_id[labels[name]])
    if len(entries) != 50000 or set(entries) != set(labels):
        raise ValueError('all 50000 official validation images are required')
    ordered = [entries[name] for name in sorted(entries)]
    if set(Counter(row['label'] for row in ordered).values()) != {50}:
        raise ValueError('expected 50 images in each of 1000 classes')
    publish(receipt, dict(archive=identity, images=ordered))
    return ordered


def transform(raw):
    with Image.open(io.BytesIO(raw)) as source:
        rgb = source.convert('RGB')
    rgb_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    image = TF.center_crop(TF.resize(rgb, 182, InterpolationMode.BICUBIC, antialias=True), [160, 160])
    return TF.to_tensor(image), rgb_sha


def write_chunk(path, **arrays):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    base.replace_with_retry(tmp, path)


def metrics_from_chunks(entries, freeze, complete=False):
    count = top1 = top5 = 0
    loss_sum = 0.
    chain = hashlib.sha256()
    per_class = np.zeros((1000, 2), dtype=np.int64)
    for start in range(0, len(entries), 32):
        path = ROOT / 'chunks' / f'{start:05d}.npz'
        if not path.exists():
            break
        with np.load(path, allow_pickle=False) as data:
            expected = entries[start:start + 32]
            assert data['names'].tolist() == [r['name'] for r in expected]
            target = data['labels']
            assert target.tolist() == [r['label'] for r in expected]
            indices, losses = data['top5'], data['ce']
            assert indices.shape == (len(expected), 5) and np.isfinite(losses).all()
            assert str(data['freeze_sha256']) == freeze
            good = indices[:, 0] == target
            top1 += int(good.sum())
            top5 += int((indices == target[:, None]).any(1).sum())
            loss_sum += float(losses.sum())
            count += len(target)
            np.add.at(per_class[:, 0], target, good.astype(np.int64))
            np.add.at(per_class[:, 1], target, 1)
        chain.update(base.sha256(path).encode())
    result = dict(samples=count, expected_samples=50000, complete=count == 50000,
                  top1=100. * top1 / count if count else None,
                  top5=100. * top5 / count if count else None,
                  ce=loss_sum / count if count else None,
                  top1_correct=top1, top5_correct=top5, chunks_hash_chain=chain.hexdigest())
    if complete:
        assert count == 50000 and (per_class[:, 1] == 50).all()
        with (ROOT / 'class_results.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['class_id', 'correct', 'samples', 'top1'])
            for i, (correct, total) in enumerate(per_class):
                writer.writerow([i, int(correct), int(total), 100. * correct / total])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    freeze_path = ROOT / 'freeze.json'
    frozen = read(freeze_path)
    freeze_sha = base.sha256(freeze_path)
    complete = read(EXPERIMENT / 'run/complete.json')
    checkpoint = EXPERIMENT / 'run/checkpoints' / complete['selected_checkpoint']
    if complete['completed_epochs'] != 20 or complete['selected_epoch'] != 20 or complete['updates'] != 800740:
        raise ValueError('only the completed fixed epoch20 model may be tested')
    if base.sha256(checkpoint) != frozen['checkpoint_sha256'] or checkpoint.stem != frozen['checkpoint_sha256']:
        raise ValueError('fixed final checkpoint hash differs')
    for rel, digest in frozen['sources'].items():
        if base.sha256(ROOT / rel) != digest:
            raise ValueError('frozen evaluation source changed: ' + rel)
    if base.sha256(ROOT / 'data/labels.json') != frozen['labels_sha256']:
        raise ValueError('validation labels changed')
    download_receipt = read(ROOT / 'data/download_complete.json')
    if download_receipt['md5'] != '29b22e2961454d5413ddabcf34fc5622' or download_receipt['bytes'] != 6744924160:
        raise ValueError('original validation archive verification is missing')
    if args.archive.stat().st_size != download_receipt['bytes']:
        raise ValueError('validation tar truncated')
    if args.archive.stat().st_mtime_ns != download_receipt['mtime_ns']:
        raise ValueError('validation tar changed after verification')
    handoff = Path(frozen['handoff_root'])
    modules = base.load_protocol_modules(handoff)
    modules['execution'].setup(seed=2026)
    classes_path = handoff / 'experiments/w80_e23/handoff/classes.json'
    classes = read(classes_path)
    if base.sha256(classes_path) != frozen['classes_sha256']:
        raise ValueError('class mapping differs')
    labels = read(ROOT / 'data/labels.json')
    entries = build_index(args.archive, labels, classes)
    if args.verify_only:
        result = metrics_from_chunks(entries, freeze_sha, complete=True)
        saved = read(ROOT / 'metrics.json')
        assert all(saved[k] == value for k, value in result.items())
        print(json.dumps(dict(status='verified', **result)), flush=True)
        return
    if (ROOT / 'metrics.json').exists():
        raise FileExistsError('evaluation completed; use --verify-only, do not rerun')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['contract'] != complete['contract'] or not state['progress']['complete']:
        raise ValueError('checkpoint state does not match completed training contract')
    config = handoff / 'experiments/mcunet_dwt/weights/mcunet-256kb-1mb_imagenet.json'
    weights = EXPERIMENT.parents[1] / 'weights/mcunet-256kb-1mb_imagenet.pth'
    backbone = modules['model'].load_pretrained(config, weights, frozen['mcunet_root'])
    model = base.RGB160(backbone)
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.cuda().eval()
    reader = UncachedReader(args.archive)
    started = time.time()
    try:
        with torch.inference_mode(), base.run_lock(ROOT / 'evaluation'):
            for start in range(0, len(entries), 32):
                chunk_path = ROOT / 'chunks' / f'{start:05d}.npz'
                if chunk_path.exists():
                    continue
                rows = entries[start:start + 32]
                images, raw_hashes, rgb_hashes = [], [], []
                for row in rows:
                    raw = reader.read(row['offset'], row['bytes'])
                    image, rgb_sha = transform(raw)
                    images.append(image)
                    raw_hashes.append(hashlib.sha256(raw).hexdigest())
                    rgb_hashes.append(rgb_sha)
                target = torch.tensor([row['label'] for row in rows], device='cuda')
                logits = model(torch.stack(images).cuda())
                losses = F.cross_entropy(logits, target, reduction='none')
                if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
                    raise FloatingPointError('nonfinite validation prediction')
                indices = logits.topk(5, dim=1).indices
                write_chunk(chunk_path, names=np.array([row['name'] for row in rows]),
                            labels=target.cpu().numpy(), top5=indices.cpu().numpy(),
                            ce=losses.double().cpu().numpy(), raw_sha256=np.array(raw_hashes),
                            rgb_sha256=np.array(rgb_hashes), freeze_sha256=np.array(freeze_sha))
                count = start + len(rows)
                if count % 1024 == 0 or count == 50000:
                    publish(ROOT / 'progress.json', dict(status='evaluating', images=count, total=50000,
                            selected_epoch=20, checkpoint_sha256=frozen['checkpoint_sha256'],
                            updated_at=time.time(), elapsed_seconds=time.time() - started))
                    print(f'validation {count}/50000', flush=True)
            result = metrics_from_chunks(entries, freeze_sha, complete=True)
            result.update(selected_epoch=20, checkpoint_sha256=frozen['checkpoint_sha256'],
                          freeze_sha256=freeze_sha, parameter_count=729392, conv_linear_macs=67287200,
                          precision='fp32', amp=False, tf32=False, inference_bn='eval',
                          preprocessing='RGB decode, bicubic antialias short182, center160, 2*x-1',
                          test_used_for_selection=False, weights_updated=False,
                          dataset='original ILSVRC2012 official validation, 50000 images, 1000 classes',
                          official_archive_md5=download_receipt['md5'],
                          training_images=1281167, former_20k_dev_included_in_training=True,
                          imagenet_v2_evaluated=False, finished_at=time.time(),
                          elapsed_seconds_this_invocation=time.time()-started)
            publish(ROOT / 'metrics.json', result)
            publish(ROOT / 'progress.json', dict(status='complete', images=50000, total=50000,
                    selected_epoch=20, updated_at=time.time()))
            print(json.dumps(result, indent=2), flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
