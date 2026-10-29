#!/usr/bin/env python3
"""One isolated job in the two-route matched-data pilot queue."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from experiments.w80_pilot.data import MirrorImages
from experiments.w80_pilot.vendor import model as w80, runtime
from experiments.w80_pilot.vendor.data import digest, epoch_order, sha
from experiments.w80_pilot.routes import make_parts
from experiments.w80_pilot.atomic_io import publish_snapshot, write_json
from experiments.w80_pilot.recovery import best_effort_publish, recoverable


def now():
    return datetime.now(timezone.utc).isoformat()


def available_host_memory():
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in ['total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended']]]
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError('host memory query failed')
        return memory.avail_phys
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


class Guard:
    """Read-only five-second telemetry; training pauses only at batch boundaries."""
    def __init__(self, root, host_low_gib, host_high_gib):
        self.root = root
        self.host_low = host_low_gib * 2**30
        self.host_high = host_high_gib * 2**30
        self.paused = False
        self.healthy_since = None
        self.stopped = threading.Event()
        self.sample = None
        self.last_time = 0
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def watch(self):
        while not self.stopped.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                self.sample = dict(host_available=available_host_memory(), gpu_free=free, gpu_total=total)
            except Exception as error:
                self.sample = dict(query_error=str(error))
            self.last_time = time.monotonic()
            try:
                publish_snapshot(self.root / 'resource.json', dict(at=now(), **self.sample))
            except OSError:
                logging.warning('Resource snapshot publish failed; live telemetry continues')
            self.stopped.wait(5)

    def decision(self):
        if self.sample is None:
            return 'wait'
        if time.monotonic() - self.last_time > 30:
            return 'protect'
        if self.sample.get('query_error'):
            return 'protect'
        low = self.sample['host_available'] < self.host_low or self.sample['gpu_free'] < 512 * 2**20
        high = self.sample['host_available'] > self.host_high and self.sample['gpu_free'] > 1536 * 2**20
        if low:
            self.paused = True; self.healthy_since = None
            return 'pause'
        if high:
            self.healthy_since = self.healthy_since or time.monotonic()
            if time.monotonic() - self.healthy_since >= 120:
                self.paused = False
        else:
            self.healthy_since = None
        return 'pause' if self.paused else 'continue'

    def close(self):
        self.stopped.set(); self.thread.join(timeout=10)


def evaluate(model, loader):
    model.eval(); total = top1 = top5 = 0; loss_sum = 0.
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite complete-dev CE')
            ranks = logits.topk(5, dim=1).indices
            loss_sum += loss.item(); total += len(y)
            top1 += ranks[:, 0].eq(y).sum().item()
            top5 += ranks.eq(y[:, None]).any(1).sum().item()
    if total != 4000:
        raise ValueError('incomplete development evaluation')
    return dict(loss=loss_sum / total, top1=100 * top1 / total, top5=100 * top5 / total, samples=total)


def plot_and_summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for variant in ('ll', 'll_lh', 'll_hl', 'll_hh', 'rgb80'):
        folder = root / variant
        summary_path = folder / 'summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        with (folder / 'history.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(r['epoch']) for r in rows]
        for axis, field in zip(axes.flat, ('train_ce', 'dev_ce', 'train_top1', 'dev_top1')):
            axis.plot(epochs, [float(r[field]) for r in rows], label=variant)
        results.append(summary)
    for axis, title, unit in zip(axes.flat, ('Training CE', 'Complete development CE', 'Training Top-1', 'Complete development Top-1'), ('CE', 'CE', 'Accuracy (%)', 'Accuracy (%)')):
        axis.set(title=title, xlabel='Epoch', ylabel=unit)
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle('Two-route resized-image pilot / 1000 classes / preliminary dev only')
    fig.savefig(root / 'dev_curves.png', dpi=150); plt.close(fig)
    if results:
        with (root / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--shared-root', type=Path, default=ROOT / 'runs/w80_e23_pilot20k_v2')
    parser.add_argument('--route', choices=['spatial_s2', 'w80_e23'], required=True)
    parser.add_argument('--variant', choices=['ll', 'll_lh', 'll_hl', 'll_hh', 'rgb80'], required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'datasets/imagenet-1k-256-subset200/train')
    parser.add_argument('--mcunet-root', type=Path, default=Path.home() / 'Desktop/Mcunet/mcunet')
    parser.add_argument('--workers', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--host-low-gib', type=float, default=.5)
    parser.add_argument('--host-high-gib', type=float, default=1.)
    parser.add_argument('--unattended', action='store_true',
                        help='No display/resource thread or speculative RAM pause; recover actual allocation failures')
    args = parser.parse_args()
    if args.host_low_gib <= 0 or args.host_high_gib <= args.host_low_gib:
        parser.error('host recovery watermark must exceed the positive pause watermark')
    root = args.run_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(root / 'pipeline.log', encoding='utf-8')], force=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    manifest_path = args.shared_root / 'data/manifest.json'
    selection_path = args.shared_root / 'channel_selection.json'
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    checks = json.loads((root.parent / 'route_cuda_checks.json').read_text())
    data_contract = dict(manifest_sha256=sha(manifest_path), selection_sha256=sha(selection_path),
                         precision='fp32', tf32=False, batch=32, steps_per_epoch=500, total_epochs=20)
    if not checks['passed'] or checks['contract'] != data_contract or [args.route, args.variant] not in checks['covered_jobs']:
        raise ValueError('CUDA evidence does not cover the current pilot data/selection')
    if selection['contract']['manifest_sha256'] != sha(manifest_path) or selection['indices_sha256'] != digest(selection['indices']):
        raise ValueError('shared channel selection checksum mismatch')
    if sha(ROOT / 'weights/mcunet-256kb-1mb_imagenet.pth') != selection['contract']['pretrained_sha256']:
        raise ValueError('pretrained weights changed after calibration')
    vendor = ROOT / 'experiments/w80_pilot/vendor'
    vendor_source = json.loads((vendor / 'SOURCE.json').read_text())
    for name, expected in vendor_source['files_sha256'].items():
        if sha(vendor / name) != expected:
            raise ValueError('teammate component changed')
    files = ['scripts/train_route_pilot.py', 'experiments/w80_pilot/routes.py', 'experiments/w80_pilot/data.py',
             'experiments/w80_pilot/atomic_io.py', 'experiments/w80_pilot/recovery.py']
    contract = dict(**data_contract, source_commit=vendor_source['commit'],
                    source_files={name: sha(ROOT / name) for name in files},
                    train_images=16000, dev_images=4000, variants=[args.variant], route=args.route, seed=2026,
                    loss='CE only', micro_batch=32, bn_batch=32, accumulation=1,
                    momentum=.9, weight_decay=4e-5, backbone_head_peak_lr=.001, new_stem_peak_lr=.005,
                    warmup_epochs=2, total_epochs_including_warmup=20, minimum_epochs=6,
                    stopping='4 epochs without CE improvement >.001 AND last4 |OLS slope|<=.001 AND range<=.005',
                    checkpoint_selection='strict minimum complete dev CE, first exact tie',
                    scope='Two-route resized ImageNet 16k/4k pilot; development only',
                    workers=args.workers, resource_policy=(dict(mode='unattended_actual_allocation_recovery',
                        telemetry_thread=False, speculative_ram_pause=False, pin_memory=False,
                        checkpoint_updates=100, optional_status=True) if args.unattended else
                        dict(host_pause_gib=args.host_low_gib, host_resume_gib=args.host_high_gib, gpu_pause_mib=512, gpu_resume_mib=1536, healthy_seconds=120)), versions=dict(torch=torch.__version__, torchvision=__import__('torchvision').__version__,
                                                      python=sys.version.split()[0], Pillow=__import__('PIL').__version__, numpy=np.__version__))
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != contract:
            raise ValueError('pilot protocol changed; use an audited migration or new run directory')
    for name, expected in checks['route_sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('route source differs from CUDA checks: ' + name)
    if not protocol_path.exists():
        write_json(protocol_path, contract)
    lock = root / '.training.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    guard = None if args.unattended else Guard(root, args.host_low_gib, args.host_high_gib)
    def status(stage, **values):
        best_effort_publish(publish_snapshot, root / 'status.json',
                            dict(stage=stage, pid=os.getpid(), updated_at=now(), scope=contract['scope'],
                                 unattended=args.unattended, **values))
    current = None
    try:
        base = w80.load_pretrained(ROOT / 'weights/mcunet-256kb-1mb_imagenet.json',
                                   ROOT / 'weights/mcunet-256kb-1mb_imagenet.pth', args.mcunet_root)
        train_rows = [r for r in manifest['rows'] if r['role'] == 'train']
        dev_rows = [r for r in manifest['rows'] if r['role'] == 'dev']
        dev_loader = DataLoader(MirrorImages(args.data_root, dev_rows, 'dev'), batch_size=32, shuffle=False,
                                num_workers=args.workers, pin_memory=not args.unattended, generator=torch.Generator().manual_seed(2027))
        for variant in (args.variant,):
            current = variant; folder = root / variant; folder.mkdir(exist_ok=True)
            if (folder / 'summary.json').exists():
                summary = json.loads((folder / 'summary.json').read_text())
                logging.info('%s already complete; preserving outcome', variant); continue
            arm_handler = logging.FileHandler(folder / 'train.log', encoding='utf-8')
            arm_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            logging.getLogger().addHandler(arm_handler)
            random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
            parts = make_parts(base, selection['indices'], args.route, variant)
            model, optimizer, scheduler, scaler = parts
            stop = runtime.EarlyStop()
            progress = dict(epoch=0, batch_cursor=0, successful_updates=0, attempted_updates=0, failed_attempts=0,
                            training_exposures=0, attempted_training_exposures=0, best_epoch=None,
                            early_stop=dict(best=stop.best, anchor=stop.anchor, bad=stop.bad, history=[]),
                            epoch_loss_sum=0., epoch_top1=0, epoch_top5=0, epoch_samples=0, completed=False)
            checkpoints = runtime.Checkpoints(folder / 'checkpoints')
            arm_contract = dict(**contract, variant=variant)
            if (folder / 'checkpoints/checkpoints.json').exists():
                progress = runtime.restore(checkpoints.load(), *parts, arm_contract)
                stop.__dict__.update(progress['early_stop'])
                logging.info('Resume %s epoch=%d cursor=%d updates=%d', variant, progress['epoch'], progress['batch_cursor'], progress['successful_updates'])
                if progress.get('last_epoch_metrics'):
                    history_path = folder / 'history.csv'
                    previous_rows = []
                    if history_path.exists():
                        with history_path.open(newline='') as stream:
                            previous_rows = list(csv.DictReader(stream))
                    last_row = progress['last_epoch_metrics']
                    if not any(int(row['epoch']) == int(last_row['epoch']) for row in previous_rows):
                        previous_rows.append(last_row)
                        with history_path.open('w', newline='', encoding='utf-8') as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(last_row)); writer.writeheader(); writer.writerows(previous_rows)
            else:
                checkpoints.save(runtime.capture(*parts, progress, arm_contract))
            write_json(folder / 'config.json', arm_contract)
            arm_start = time.monotonic()
            previous_elapsed = progress.get('elapsed_seconds', 0.)
            last_batch_metrics = {}
            def collect_train_metrics(module, inputs, logits):
                if module.training:
                    last_batch_metrics['ranks'] = logits.detach().topk(5, dim=1).indices
            metric_hook = model.register_forward_hook(collect_train_metrics)
            def save(best=False):
                progress['elapsed_seconds'] = previous_elapsed + time.monotonic() - arm_start
                progress['early_stop'] = dict(stop.__dict__)
                checkpoints.save(runtime.capture(*parts, progress, arm_contract), best=best)
            def wait_resource():
                if guard is None:
                    return
                action = guard.decision()
                if action == 'protect':
                    save()
                    raise RuntimeError('resource telemetry failed or stale; checkpoint retained')
                saved = False
                while action in ('wait', 'pause'):
                    if not saved:
                        save(); saved = True
                    status('resource_paused', variant=variant, resource=guard.sample, successful_updates=progress['successful_updates'])
                    if (root.parent / 'STOP').exists():
                        raise RuntimeError('queue STOP requested while resource paused; checkpoint saved')
                    time.sleep(5); action = guard.decision()
                    if action == 'protect':
                        save()
                        raise RuntimeError('resource protection: telemetry failure')
            while progress['epoch'] < 20 and not progress['completed']:
                epoch = progress['epoch']
                order = epoch_order(len(train_rows), 2026, epoch)
                cursor = progress['batch_cursor']
                dataset = MirrorImages(args.data_root, train_rows, 'train', epoch=epoch)
                train_loader = DataLoader(Subset(dataset, order[cursor * 32:]), batch_size=32, shuffle=False,
                                          num_workers=args.workers, pin_memory=not args.unattended,
                                          generator=torch.Generator().manual_seed(2026 + epoch))
                status('training', variant=variant, epoch=epoch + 1, batch_cursor=cursor, successful_updates=progress['successful_updates'])
                for x, y in train_loader:
                    if (root.parent / 'STOP').exists():
                        save()
                        status('stopped_by_request', variant=variant, successful_updates=progress['successful_updates'])
                        return
                    wait_resource()
                    if len(y) != 32:
                        raise ValueError('BN batch must remain 32')
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    progress['attempted_updates'] += 1; progress['attempted_training_exposures'] += 32
                    try:
                        loss, _ = runtime.transactional_step(*parts, x, y, progress, arm_contract)
                    except BaseException as error:
                        progress['failed_attempts'] += 1; save()
                        with (folder / 'failures.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(at=now(), epoch=epoch, batch_cursor=progress['batch_cursor'], error=repr(error))) + '\n')
                        raise
                    progress['successful_updates'] += 1; progress['training_exposures'] += 32
                    ranks = last_batch_metrics.pop('ranks')
                    progress['epoch_top1'] += ranks[:, 0].eq(y).sum().item()
                    progress['epoch_top5'] += ranks.eq(y[:, None]).any(1).sum().item()
                    progress['epoch_loss_sum'] += loss * 32; progress['epoch_samples'] += 32
                    progress['batch_cursor'] += 1
                    if progress['batch_cursor'] % 100 == 0:
                        logging.info('%s epoch=%d batch=%d/500 CE=%.4f updates=%d exposures=%d', variant, epoch + 1, progress['batch_cursor'], progress['epoch_loss_sum'] / progress['epoch_samples'], progress['successful_updates'], progress['training_exposures'])
                        status('training', variant=variant, epoch=epoch + 1, batch_cursor=progress['batch_cursor'], successful_updates=progress['successful_updates'], training_exposures=progress['training_exposures'])
                    if progress['batch_cursor'] % (100 if args.unattended else 250) == 0:
                        save()
                if progress['batch_cursor'] != 500 or progress['epoch_samples'] != 16000:
                    raise ValueError('incomplete pilot training epoch')
                status('development', variant=variant, epoch=epoch + 1, successful_updates=progress['successful_updates'])
                dev = evaluate(model, dev_loader)
                best, early = stop.observe(dev['loss'])
                if best:
                    progress['best_epoch'] = epoch + 1
                row = dict(epoch=epoch + 1, train_images=16000, dev_images=4000,
                           train_top1=100 * progress['epoch_top1'] / 16000, train_top5=100 * progress['epoch_top5'] / 16000,
                           train_ce=progress['epoch_loss_sum'] / 16000, dev_ce=dev['loss'], dev_top1=dev['top1'], dev_top5=dev['top5'],
                           successful_updates=progress['successful_updates'], attempted_updates=progress['attempted_updates'], failed_attempts=progress['failed_attempts'],
                           training_exposures=progress['training_exposures'], attempted_training_exposures=progress['attempted_training_exposures'],
                           elapsed_seconds=previous_elapsed + time.monotonic() - arm_start, next_backbone_lr=optimizer.param_groups[0]['lr'])
                progress['epoch'] = epoch + 1; progress['batch_cursor'] = 0
                progress['epoch_loss_sum'] = 0.; progress['epoch_samples'] = 0
                progress['epoch_top1'] = 0; progress['epoch_top5'] = 0
                progress['completed'] = early or progress['epoch'] == 20
                progress['stop_reason'] = 'dev_plateau' if early else ('max20epochs' if progress['completed'] else None)
                progress['last_epoch_metrics'] = row
                save(best=best)
                history_path = folder / 'history.csv'
                history = []
                if history_path.exists():
                    with history_path.open(newline='') as stream:
                        history = [r for r in csv.DictReader(stream) if int(r['epoch']) < epoch + 1]
                with history_path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history); writer.writerow(row)
                logging.info('%s epoch=%d train_ce=%.5f dev_ce=%.5f dev_top1=%.2f best=%s stop=%s', variant, epoch + 1, row['train_ce'], dev['loss'], dev['top1'], best, early)
                status('training', variant=variant, epoch=progress['epoch'] + 1, batch_cursor=0,
                       successful_updates=progress['successful_updates'],
                       training_exposures=progress['training_exposures'],
                       last_epoch_metrics=row, best_epoch=progress['best_epoch'])
            best_state = checkpoints.load('best')
            model.load_state_dict(best_state['model'])
            model.eval()
            with (folder / 'dev_predictions.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['image_id', 'true_label', 'pred_top5'])
                offset = 0
                with torch.inference_mode():
                    for x, y in dev_loader:
                        ranks = model(x.cuda()).topk(5, dim=1).indices.cpu().tolist()
                        for i, top in enumerate(ranks):
                            writer.writerow([dev_rows[offset + i]['path'], int(y[i]), ' '.join(map(str, top))])
                        offset += len(y)
                if offset != 4000:
                    raise ValueError('incomplete development predictions')
            selected_metrics = best_state['progress']['last_epoch_metrics']
            summary = dict(route=args.route, variant=variant, train_images=16000, dev_images=4000,
                           epochs_completed=progress['epoch'], selected_epoch=best_state['progress']['best_epoch'],
                           dev_ce=selected_metrics['dev_ce'], dev_top1=selected_metrics['dev_top1'], dev_top5=selected_metrics['dev_top5'],
                           successful_updates=progress['successful_updates'], attempted_updates=progress['attempted_updates'], failed_attempts=progress['failed_attempts'],
                           training_exposures=progress['training_exposures'], attempted_training_exposures=progress['attempted_training_exposures'],
                           elapsed_seconds=progress['elapsed_seconds'],
                           stop_reason=progress['stop_reason'], scope='resized-image 20k pilot; dev only; no official validation accuracy')
            torch.save(best_state, folder / 'best.pt'); torch.save(checkpoints.load('latest'), folder / 'last.pt')
            write_json(folder / 'summary.json', summary)
            metric_hook.remove()
            logging.getLogger().removeHandler(arm_handler); arm_handler.close()
            if not args.unattended:
                plot_and_summarize(root)
            del parts, model, optimizer, scheduler, scaler, best_state
            torch.cuda.empty_cache()
        results = ([summary] if args.unattended else plot_and_summarize(root))
        status('complete', completed_arms=len(results), results=str(root / 'results.csv'))
    except BaseException as error:
        retryable = args.unattended and recoverable(error)
        # Resume only a committed generation. During best-model export, model
        # weights no longer match the latest optimizer; never capture them here.
        # A failed epoch/best commit must also be replayed from its prior pointer.
        if retryable:
            logging.warning('Automatic recovery uses the last committed checkpoint')
        status('recoverable_error' if retryable else 'failed', variant=current, error=repr(error),
               successful_updates=progress.get('successful_updates', 0) if 'progress' in locals() else 0)
        logging.exception('Pilot stopped; inspect error before resuming')
        if retryable:
            raise SystemExit(75) from error
        raise
    finally:
        if guard is not None:
            guard.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
