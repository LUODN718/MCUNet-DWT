"""Read-only progress window; closing this script does not stop training."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / 'runs/two_route_pilot20k'
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.w80_pilot.atomic_io import read_latest_snapshot


def read(path):
    return read_latest_snapshot(path)


def display():
    q = read(ROOT / 'queue_status.json')
    job = q.get('job', '')
    folder = ROOT / job
    s = read(folder / 'status.json')
    # Resource snapshots from before the unattended migration are historical.
    r = {} if s.get('unattended') else read(folder / 'resource.json')
    updates = int(s.get('successful_updates', 0))
    stage = s.get('stage', '-')
    epoch = s.get('epoch')
    if epoch is None and stage in ('training', 'resource_paused', 'development'):
        epoch = min(20, updates // 500 + 1)
    cursor = s.get('batch_cursor', updates % 500)
    lines = ['MCUNet ImageNet: two-route training progress',
             datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'),
             '=' * 66,
             f"Queue: {q.get('stage', '-')} | Job {q.get('position', '-')}/{q.get('total', 9)}: {job}",
             f"Trainer: {stage} | PID: {s.get('pid', '-')}",
             f"Epoch: {epoch or '-'} / 20 maximum | Updates: {updates:,}",
             f"Training exposures: {updates * 32:,} | Batch: {cursor} / 500",
             'FP32 | BN batch 32 | seed 2026 | development subset only',
             f"Last trainer status: {s.get('updated_at', '-')}"]
    if 'host_available' in r:
        lines.append(f"Free RAM: {r['host_available']/2**30:.2f} GiB | Free GPU: {r['gpu_free']/2**30:.2f} GiB")
    if s.get('unattended'):
        lines.append('Unattended: no resource pause; recoverable errors resume automatically.')
    if s.get('stage') == 'resource_paused':
        lines.append('Resource protection: waiting for 120 seconds of healthy resources.')
    metrics = s.get('last_epoch_metrics')
    if metrics:
        lines.append(f"Last full epoch {metrics['epoch']}: dev CE={float(metrics['dev_ce']):.4f}, Top1={float(metrics['dev_top1']):.2f}%")
        lines.append(f"Best epoch so far: {s.get('best_epoch', '-')}")
    lines.extend(['-' * 66, 'QUEUE:'])
    for i, entry in enumerate(read(ROOT / 'queue.json').get('jobs', []), 1):
        done = (ROOT / entry['job'] / entry['variant'] / 'summary.json').exists()
        label = 'DONE' if done else ('CURRENT' if entry['job'] == job else 'WAIT')
        lines.append(f"{i}. [{label:7s}] {entry['job']}")
    lines.extend(['-' * 66, 'LATEST LOG:'])
    try:
        lines.extend((folder / 'pipeline.log').read_text(encoding='utf-8').splitlines()[-4:])
    except OSError:
        pass
    lines.extend(['=' * 66, 'Read-only. Ctrl+C closes this display; it does not stop the training process.',
                  'training = updating; resource_paused = waiting for RAM/GPU; failed = stopped.',
                  'The update count is published every 100 batches, not on every screen refresh.'])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            print(('' if args.once else '\x1b[2J\x1b[H') + display(), flush=True)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print('Progress display closed; training continues.')
