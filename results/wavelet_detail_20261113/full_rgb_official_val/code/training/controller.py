"""Single full-data run: preflight admission, hidden worker, bounded recovery."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from full_data import ROOT,publish,sha
from baseline_helpers import run_lock
from run_w80_original_tar import resources

def verify_sources():
    frozen=json.loads((ROOT/'source_freeze.json').read_text())['sources']
    for rel,h in frozen.items():
        if sha(ROOT/rel)!=h:raise ValueError('source freeze mismatch: '+rel)
    return frozen

def main():
    logs=ROOT/'run_logs';logs.mkdir(exist_ok=True)
    state=logs/'controller.json'
    with run_lock(ROOT/'controller'):
        sources=verify_sources()
        original=dict(controller_pid=os.getpid(),scope='full original ImageNet TRAIN RGB160 fixed20 no dev/test',started_at=time.time())
        if (ROOT/'run/complete.json').exists():
            publish(state,dict(original,status='completed',updated_at=time.time()));return
        K=ctypes.WinDLL('kernel32',use_last_error=True)
        if not K.SetThreadExecutionState(0x80000001):raise ctypes.WinError(ctypes.get_last_error())
        try:
            while not (ROOT/'data/index_receipt.json').exists():
                err=(logs/'index.stderr.log').read_text(encoding='utf-8',errors='replace')
                if err.strip():raise RuntimeError('full index preparation failed: '+err[-2000:])
                publish(state,dict(original,status='waiting_for_full_index',updated_at=time.time()))
                time.sleep(10)
            admission=ROOT/'preflight/admission.json'
            if not admission.exists():
                publish(state,dict(original,status='cuda_preflight',updated_at=time.time()))
                with (logs/'preflight.stdout.log').open('ab') as out,(logs/'preflight.stderr.log').open('ab') as err:
                    result=subprocess.run([sys.executable,'-B','-u',str(ROOT/'code/preflight.py')],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode!=0:raise RuntimeError('CUDA preflight failed; inspect preflight.stderr.log')
            passed=json.loads(admission.read_text())
            if passed['status']!='passed' or passed['sources']!=sources:raise ValueError('admission missing or sources changed')
            previous_attempts=[int(p.name.split('_')[1].split('.')[0]) for p in logs.glob('attempt_*.stdout.log')]
            attempt=max(previous_attempts,default=-1)+1
            while attempt<=3:
                if (ROOT/'run/complete.json').exists():
                    publish(state,dict(original,status='completed',attempt=attempt,updated_at=time.time()));return
                healthy=None
                while True:
                    r=resources(ROOT)
                    high=r['host_available']>=(4 if attempt else 2)*2**30 and r['gpu_free']>=1536*2**20 and r['disk_free']>=10*2**30
                    healthy=(healthy or time.monotonic()) if high else None
                    if healthy is not None and (attempt==0 or time.monotonic()-healthy>=120):break
                    publish(state,dict(original,status='waiting_for_resources',attempt=attempt,resources=r,updated_at=time.time()))
                    time.sleep(20)
                verify_sources()
                resume=(ROOT/'run/checkpoints/checkpoints.json').exists()
                command=[sys.executable,'-B','-u',str(ROOT/'code/train_full.py')]+(['--resume'] if resume else [])
                stdout=logs/f'attempt_{attempt:03d}.stdout.log';stderr=logs/f'attempt_{attempt:03d}.stderr.log'
                if stdout.exists() or stderr.exists():raise FileExistsError('attempt logs exist; inspect before controller restart')
                with stdout.open('xb') as out,stderr.open('xb') as err:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    while worker.poll() is None:
                        p=ROOT/'run/progress.json'
                        try: progress=json.loads(p.read_text()) if p.exists() else None
                        except (OSError,ValueError):progress=None
                        publish(state,dict(original,status='running',attempt=attempt,worker_pid=worker.pid,resume=resume,progress=progress,updated_at=time.time()))
                        time.sleep(10)
                if worker.returncode==0 and (ROOT/'run/complete.json').exists():
                    publish(state,dict(original,status='completed',attempt=attempt,updated_at=time.time()));return
                message=stderr.read_text(encoding='utf-8',errors='replace')
                transient=('torch.OutOfMemoryError' in message or 'torch.cuda.OutOfMemoryError' in message or 'PermissionError' in message)
                if not transient:raise RuntimeError('worker failed; no blind retry: '+message[-3000:])
                attempt+=1
                publish(state,dict(original,status='recovering_transient_failure',attempt=attempt,error=message[-2000:],updated_at=time.time()))
            raise RuntimeError('maximum3 transient recovery attempts reached')
        finally:K.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        publish(ROOT/'run_logs/controller_error.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),time=time.time()))
        raise
