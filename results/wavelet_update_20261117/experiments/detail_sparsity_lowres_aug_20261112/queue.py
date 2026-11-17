"""Serial, resumable 128k subset queue; no resource-watermark pause or alarms."""
from contextlib import contextmanager
import ctypes
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.1 * 2**attempt, 2))


@contextmanager
def lock(path):
    with Path(path).open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate(config):
    expected = [dict(name='ll80_light', variant='ll80', augmentation='light'),
                dict(name='rgb80_light', variant='rgb80', augmentation='light'),
                dict(name='rgb80_plain', variant='rgb80', augmentation='plain')]
    if config['experiments'] != expected:
        raise ValueError('only the declared three subset arms may run')
    if (config['train_images'], config['dev_images'], config['maximum_epochs']) != (128000, 20000, 20):
        raise ValueError('subset size or horizon changed')
    for relative, checksum in config['sources'].items():
        if sha(ROOT / relative) != checksum:
            raise ValueError('frozen source changed: ' + relative)
    for path, checksum in config['reused_evidence'].items():
        if sha(path) != checksum:
            raise ValueError('historical comparison evidence changed: ' + path)
    if sha(config['manifest']) != config['manifest_sha256']:
        raise ValueError('cohort manifest changed')
    admission = read(ROOT / 'preflight.json')
    if admission['status'] != 'passed' or admission['sources'] != config['sources']:
        raise ValueError('matching successful preflight required')
    for name in ('run_root', 'cache_root', 'seal_dir'):
        if not Path(config[name]).resolve().is_relative_to(ROOT):
            raise ValueError('outputs must remain inside this fyp experiment')


def summarize(config):
    rows = []
    arms = [('rgb160_plain_reused', Path(config['reused_rgb160_plain']), 67287200),
            ('ll80_plain_reused', Path(config['reused_ll80_plain']), 18021824)]
    arms += [(arm['name'], Path(config['run_root']) / arm['name'], 18021824)
             for arm in config['experiments']]
    for name, folder, macs in arms:
        if not (folder / 'complete.json').exists():
            continue
        complete = read(folder / 'complete.json')
        with (folder / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            best = next(r for r in csv.DictReader(stream)
                        if int(r['epoch']) == complete['best_epoch'])
        rows.append(dict(variant=name, completed_epochs=complete['completed_epochs'],
                         best_epoch=complete['best_epoch'], dev_ce=float(best['dev_ce']),
                         dev_top1=float(best['dev_top1']), dev_top5=float(best['dev_top5']),
                         updates=complete['updates'], conv_linear_macs=macs,
                         reused=name.endswith('_reused')))
    publish(ROOT / 'results.json', dict(results=rows, official_val_used=False,
                                       mac_scope='CNN Conv/Linear only; preprocessing excluded',
                                       caveat='RGB160 historical control has no extra augmentation'))
    with (ROOT / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker_command(config, arm):
    run = Path(config['run_root']) / arm['name']
    command = [config['python'], '-B', '-u', str(ROOT / 'code/train_lowres.py'),
               '--variant', arm['variant'], '--augmentation', arm['augmentation']]
    for arg in ('handoff_root', 'train_tar', 'seal_dir', 'weights', 'mcunet_root'):
        command += ['--' + arg.replace('_', '-'), config[arg]]
    command += ['--run-dir', str(run), '--workers', '0', '--prefetch', '1',
                '--accept-local-content-sha256', config['manifest_sha256']]
    if (run / 'checkpoints/checkpoints.json').exists():
        command.append('--resume')
    elif run.exists() and any(run.iterdir()):
        raise ValueError('nonempty run without checkpoint; inspect: ' + str(run))
    return command, run


def run():
    config = read(ROOT / 'queue_config.json')
    validate(config)
    state = dict(queue_pid=os.getpid(), train_images=128000, dev_images=20000,
                 maximum_epochs_each=20, resource_watermark_pause=False,
                 full_training_launch_allowed=False, experiments=config['experiments'])

    def update(status, **extra):
        state.update(status=status, updated_at=time.time(), **extra)
        publish(ROOT / 'queue_status.json', state)

    with lock(ROOT / 'queue.lock'):
        if os.name == 'nt':
            # Prevent automatic idle sleep for this queue only; explicit sleep still works.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            for key, name in [('TEMP', 'temp'), ('TMP', 'temp'), ('TORCH_HOME', 'torch'),
                              ('CUDA_CACHE_PATH', 'cuda'), ('MPLCONFIGDIR', 'matplotlib')]:
                path = Path(config['cache_root']) / name
                path.mkdir(parents=True, exist_ok=True)
                env[key] = str(path)
            summarize(config)
            for arm in config['experiments']:
                run_dir = Path(config['run_root']) / arm['name']
                if (run_dir / 'complete.json').exists():
                    continue
                for attempt in range(4):
                    if (ROOT / 'PAUSE_QUEUE').exists():
                        update('paused_before_next_experiment', next_experiment=arm['name'], worker_pid=None)
                        return 0
                    validate(config)
                    command, run_dir = worker_command(config, arm)
                    stamp = time.time_ns()
                    stdout = ROOT / 'logs' / f"{arm['name']}_{stamp}.stdout.log"
                    stderr = ROOT / 'logs' / f"{arm['name']}_{stamp}.stderr.log"
                    with stdout.open('xb') as out, stderr.open('xb') as err:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                        while process.poll() is None:
                            progress_path = run_dir / 'progress.json'
                            progress = read(progress_path) if progress_path.exists() else None
                            update('running', current_experiment=arm['name'], worker_pid=process.pid,
                                   command=command, progress=progress, stdout=str(stdout), stderr=str(stderr),
                                   attempt=attempt)
                            time.sleep(15)
                        code = process.returncode
                    if code == 0 and (run_dir / 'complete.json').exists():
                        summarize(config)
                        break
                    tail = stderr.read_text(encoding='utf-8', errors='replace')[-16000:]
                    recoverable = 'CUDA out of memory' in tail or 'PermissionError' in tail
                    if not recoverable or attempt == 3 or not (run_dir / 'checkpoints/checkpoints.json').exists():
                        update('needs_attention', exit_code=code, error_tail=tail, worker_pid=None)
                        return code or 1
                    update('retrying_from_checkpoint', exit_code=code, error_tail=tail, worker_pid=None)
                    time.sleep(30)
            summarize(config)
            update('completed', worker_pid=None, progress=None,
                   completed_experiments=[arm['name'] for arm in config['experiments']])
            return 0
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    try:
        raise SystemExit(run())
    except Exception as error:
        status_path = ROOT / 'queue_status.json'
        previous = read(status_path) if status_path.exists() else {}
        if previous.get('queue_pid') in (None, os.getpid()):
            publish(status_path, dict(status='needs_attention', queue_pid=os.getpid(),
                                     error=repr(error), updated_at=time.time()))
        raise
