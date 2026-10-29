"""Sequential two-route pilot with bounded operational error recovery."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.w80_pilot.atomic_io import publish_snapshot, write_json as write
from experiments.w80_pilot.recovery import best_effort_publish, keep_awake, run_child


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/two_route_pilot20k')
    parser.add_argument('--python', type=Path, default=Path.home() / 'anaconda3/envs/dgfnet/python.exe')
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    shared = ROOT / 'runs/w80_e23_pilot20k_v2'
    checks = json.loads((root / 'route_cuda_checks.json').read_text())
    if not checks['passed'] or checks['contract']['manifest_sha256'] != sha(shared / 'data/manifest.json'):
        raise ValueError('current data must pass CUDA checks before queue launch')
    for name, expected in checks['route_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('source changed after checks: ' + name)
    jobs = [(r, b) for b in ('ll', 'll_lh', 'll_hl', 'll_hh') for r in ('spatial_s2', 'w80_e23')]
    jobs.append(('spatial_s2', 'rgb80'))
    commands = []
    for route, band in jobs:
        job = route + '_' + band
        command = [str(args.python), '-u', str(ROOT / 'scripts/train_route_pilot.py'),
                   '--run-root', str(root / job), '--shared-root', str(shared),
                   '--route', route, '--variant', band, '--workers', '0',
                   '--host-low-gib', '.75', '--host-high-gib', '1.0', '--unattended']
        commands.append(dict(job=job, route=route, variant=band, command=command))
    config = dict(manifest_sha256=sha(shared / 'data/manifest.json'),
                  selection_sha256=sha(shared / 'channel_selection.json'), jobs=commands,
                  source_files={name: sha(ROOT / name) for name in
                                ('scripts/run_route_queue.py', 'scripts/train_route_pilot.py',
                                 'experiments/w80_pilot/routes.py', 'experiments/w80_pilot/data.py',
                                 'experiments/w80_pilot/atomic_io.py', 'experiments/w80_pilot/recovery.py')},
                  seed=2026, train_images=16000, dev_images=4000, classes=1000,
                  max_epochs=20, minimum_epochs=6, precision='FP32; TF32 disabled',
                  micro_effective_bn_batch=32, updates_per_epoch=500, accumulation=1,
                  operational_policy='unattended: one child/job, no speculative RAM pause or telemetry thread; actual allocation recovery, 100-update checkpoints, optional status',
                  recovery=dict(recoverable_exit=75, max_restarts_per_job=8,
                                max_same_checkpoint_retries=3, backoff_max_seconds=60),
                  scope='resized mirror preliminary development results; no official val/V2')
    config_path = root / 'queue.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('queue changed; inspect before continuing')
    else:
        write(config_path, config)
    lock = root / '.queue.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    def status(stage, **extra):
        best_effort_publish(publish_snapshot, root / 'queue_status.json',
                            dict(stage=stage, pid=os.getpid(),
                                 at=datetime.now(timezone.utc).isoformat(), **extra))
    keep_awake(True)
    try:
        prior_protocol = json.loads((shared / 'protocol.json').read_text())
        prior_summary = json.loads((shared / 'rgb_e23/summary.json').read_text())
        for key, expected in [('manifest_sha256', config['manifest_sha256']),
                              ('selection_sha256', config['selection_sha256']),
                              ('precision', 'fp32'), ('tf32', False), ('batch', 32),
                              ('steps_per_epoch', 500), ('total_epochs', 20),
                              ('micro_batch', 32), ('bn_batch', 32), ('accumulation', 1),
                              ('loss', 'CE only'), ('momentum', .9), ('weight_decay', 4e-5),
                              ('backbone_head_peak_lr', .001), ('new_stem_peak_lr', .005),
                              ('warmup_epochs', 2), ('minimum_epochs', 6)]:
            if prior_protocol[key] != expected:
                raise ValueError('compressed RGB reuse mismatch: ' + key)
        if prior_summary['epochs_completed'] != 20 or prior_summary['failed_attempts'] != 0:
            raise ValueError('compressed RGB reuse requires successful completed run')
        reused = dict(source=str(shared / 'rgb_e23'), summary=prior_summary,
                      protocol_sha256=sha(shared / 'protocol.json'),
                      best_sha256=sha(shared / 'rgb_e23/best.pt'),
                      last_sha256=sha(shared / 'rgb_e23/last.pt'),
                      training_recipe_reused=True, operational_policy_differs=True)
        if not (root / 'reused_rgb_e23.json').exists():
            write(root / 'reused_rgb_e23.json', reused)
        collect_safely(root, commands, reused)
        for index, job in enumerate(commands, 1):
            if (root / 'STOP').exists():
                status('stopped_by_request', completed_before=index - 1); return
            summary_path = root / job['job'] / job['variant'] / 'summary.json'
            if summary_path.exists():
                continue
            def child_status(stage, **values):
                status(stage, position=index, total=len(commands), job=job['job'], **values)
            outcome, code = run_child(
                job['command'], root / job['job'], job['variant'], child_status,
                root / 'STOP',
                lambda command, **kwargs: subprocess.Popen(command, cwd=ROOT, **kwargs))
            if outcome == 'stopped':
                status('stopped_by_request', job=job['job']); return
            if outcome != 'complete':
                status('failed', job=job['job'], exit_code=code, recovery_outcome=outcome); return
            collect_safely(root, commands, reused)
        collect_safely(root, commands, reused)
        try:
            plotting = subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/plot_route_queue.py'),
                                       '--root', str(root)], cwd=ROOT, timeout=180)
            if plotting.returncode:
                print('Training complete; curve generation deferred', flush=True)
        except (OSError, subprocess.TimeoutExpired) as error:
            print('Training complete; curves deferred: ' + repr(error), flush=True)
        status('complete', completed_jobs=len(commands), reused_controls=1)
    except BaseException as error:
        status('failed', error=repr(error)); raise
    finally:
        keep_awake(False)
        lock.unlink(missing_ok=True)


def collect_safely(root, commands, reused):
    try:
        collect(root, commands, reused)
    except OSError as error:
        # The completed per-arm summaries/checkpoints remain authoritative.
        print('Aggregate CSV deferred; training continues: ' + repr(error), flush=True)


def collect(root, commands, reused):
    rows = []
    def row(route, name, summary, folder):
        return dict(route=route, variant=name, train_images=summary['train_images'],
                    dev_images=summary['dev_images'], completed_epochs=summary['epochs_completed'],
                    selected_epoch=summary['selected_epoch'], dev_ce=summary['dev_ce'],
                    dev_top1=summary['dev_top1'], dev_top5=summary['dev_top5'],
                    updates=summary['successful_updates'], exposures=summary['training_exposures'],
                    elapsed_seconds=summary['elapsed_seconds'], artifacts=str(folder),
                    official_val_top1='', official_val_top5='', scope='preliminary dev only')
    rows.append(row('w80_e23', 'rgb_e23', reused['summary'], reused['source']))
    for job in commands:
        folder = root / job['job'] / job['variant']
        p = folder / 'summary.json'
        if p.exists():
            rows.append(row(job['route'], job['variant'], json.loads(p.read_text()), folder))
    with (root / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
