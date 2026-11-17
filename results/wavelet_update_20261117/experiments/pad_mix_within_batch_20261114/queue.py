"""Run one subset reproduction attempt after BN diagnosis and current RGB control."""
import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'detail_sparsity_pad_mix_20261113'
spec = importlib.util.spec_from_file_location('prior_queue_utils', PRIOR / 'queue.py')
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)


def validate(config):
    assert config['experiments'] == [dict(name='rgb_ll1_within_batch', variant='rgb_ll1_within_batch')]
    assert (config['train_images'],config['dev_images'],config['maximum_epochs']) == (128000,20000,20)
    assert (PRIOR / 'PAUSE_QUEUE').is_file(), 'old LL-only queue must remain held'
    for relative, digest in config['sources'].items():
        assert util.sha(ROOT / relative) == digest, relative
    assert util.sha(config['manifest']) == config['manifest_sha256']
    admission = util.read(ROOT / 'preflight.json')
    assert admission['status'] == 'passed' and admission['sources'] == config['sources']
    for name in ('run_root','cache_root','seal_dir'):
        assert Path(config[name]).resolve().is_relative_to(ROOT)


def main():
    config = util.read(ROOT / 'queue_config.json')
    validate(config)
    state = dict(queue_pid=os.getpid(), train_images=128000, dev_images=20000,
        maximum_epochs=20, full_training_launch_allowed=False, resource_watermark_pause=False)
    def update(status, **extra):
        state.update(status=status, updated_at=time.time(), **extra)
        util.publish(ROOT / 'queue_status.json', state)
    with util.lock(ROOT / 'queue.lock'):
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            while True:
                if (ROOT / 'PAUSE_QUEUE').exists():
                    update('paused_before_training', worker_pid=None)
                    return 0
                diag = Path(config['wait_for_diagnostic'])
                diag_status = util.read(diag / 'status.json') if (diag / 'status.json').exists() else {}
                review_path = ROOT / 'diagnostic_review.json'
                previous = Path(config['wait_for_run'])
                prior_done = (previous / 'complete.json').exists()
                prior_idle = (util.lock_is_free(previous.with_name(previous.name+'.training.lock')) and
                    util.lock_is_free(Path(config['wait_for_queue']) / 'queue.lock'))
                diagnosis_done = diag_status.get('status') == 'complete' and review_path.exists()
                if diagnosis_done and prior_done and prior_idle:
                    findings = util.read(diag / 'results.json')
                    assert findings['status'] == 'complete' and findings['original_metrics_reproduced']
                    assert findings['all_nonbn_tensors_unchanged'] and findings['calibration_train_only']
                    review = util.read(review_path)
                    assert review['launch_method'] == 'rgb_ll1_within_batch'
                    assert review['diagnostic_results_sha256'] == util.sha(diag / 'results.json')
                    util.publish(ROOT / 'prerequisites_verified.json', dict(
                        diagnostic_results_sha256=util.sha(diag / 'results.json'),
                        prior_completion_sha256=util.sha(previous / 'complete.json'),
                        diagnostic_review_sha256=util.sha(review_path), checked_at=time.time(),
                        old_rgb_control_finished=True, old_ll_only_held=True))
                    break
                update('waiting_for_prerequisites', worker_pid=None,
                    diagnostic_status=diag_status.get('status'), diagnostic_reviewed=review_path.exists(),
                    current_rgb_control_complete=prior_done, current_rgb_control_idle=prior_idle)
                if diag_status.get('status') == 'failed':
                    update('needs_attention', error='BN diagnosis failed; not starting training')
                    return 1
                time.sleep(15)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            for key,name in [('TEMP','temp'),('TMP','temp'),('TORCH_HOME','torch'),
                             ('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib')]:
                folder = Path(config['cache_root']) / name
                folder.mkdir(parents=True, exist_ok=True)
                env[key] = str(folder)
            run_dir = Path(config['run_root']) / 'rgb_ll1_within_batch'
            if (run_dir / 'complete.json').exists():
                update('complete', worker_pid=None)
                return 0
            for attempt in range(4):
                if (ROOT / 'PAUSE_QUEUE').exists():
                    update('paused_before_training', worker_pid=None)
                    return 0
                validate(config)
                command = [config['python'],'-B','-u',str(ROOT/'code/train_within_batch.py'),
                           '--variant','rgb_ll1_within_batch']
                for arg in ('handoff_root','train_tar','seal_dir','weights','mcunet_root'):
                    command += ['--'+arg.replace('_','-'),config[arg]]
                command += ['--run-dir',str(run_dir),'--workers','0','--prefetch','1',
                            '--accept-local-content-sha256',config['manifest_sha256']]
                if (run_dir / 'checkpoints/checkpoints.json').exists():
                    command.append('--resume')
                elif run_dir.exists() and any(run_dir.iterdir()):
                    raise ValueError('nonempty run without checkpoint')
                stamp = time.time_ns()
                out_path = ROOT / 'logs' / f'train_{stamp}.stdout.log'
                err_path = ROOT / 'logs' / f'train_{stamp}.stderr.log'
                with out_path.open('xb') as out, err_path.open('xb') as err:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    while process.poll() is None:
                        progress = run_dir / 'progress.json'
                        update('training', worker_pid=process.pid, command=command, attempt=attempt,
                            progress=util.read(progress) if progress.exists() else None,
                            stdout=str(out_path), stderr=str(err_path))
                        time.sleep(15)
                if process.returncode == 0 and (run_dir / 'complete.json').exists():
                    update('complete',worker_pid=None,completion=util.read(run_dir/'complete.json'))
                    return 0
                tail = err_path.read_text(encoding='utf-8',errors='replace')[-16000:]
                recoverable = 'CUDA out of memory' in tail or 'PermissionError' in tail
                if not recoverable or attempt == 3 or not (run_dir/'checkpoints/checkpoints.json').exists():
                    update('needs_attention',worker_pid=None,exit_code=process.returncode,error_tail=tail)
                    return process.returncode or 1
                update('retrying_from_checkpoint',worker_pid=None,error_tail=tail)
                time.sleep(30)
        finally:
            if os.name == 'nt':
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        util.publish(ROOT/'queue_status.json',dict(status='needs_attention',queue_pid=os.getpid(),
            error=repr(error),updated_at=time.time()))
        raise
