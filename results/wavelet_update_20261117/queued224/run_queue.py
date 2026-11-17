"""Sequential224 runs; no memory-watermark watchdog or concurrency."""
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import train_ratio as api


def validate(config):
    for relative,digest in config['sources'].items():
        assert api.sha256(ROOT/relative)==digest,relative
    assert api.sha256(config['manifest'])==config['manifest_sha256']
    assert api.read_json(ROOT/'preflight.json')['status']=='passed'


def main():
    config=api.read_json(ROOT/'study_config.json')
    validate(config)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    for key,name in [('TEMP','temp'),('TMP','temp'),('TORCH_HOME','torch'),
                     ('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib')]:
        folder=ROOT/'cache'/name
        folder.mkdir(parents=True,exist_ok=True)
        env[key]=str(folder)
    def publish(status,**kwargs):
        api.atomic_json(ROOT/'queue_status.json',dict(status=status,queue_pid=os.getpid(),
            updated_at=time.time(),experiments=config['experiments'],resource_watermark_pause=False,**kwargs))
    with api.run_lock(ROOT/'queue'):
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            for arm in config['experiments']:
                run=ROOT/'runs'/arm['name']
                if (run/'complete.json').exists():
                    continue
                if (ROOT/'PAUSE_QUEUE').exists():
                    publish('paused_before_next',next_run=arm['name']);return
                validate(config)
                command=[config['python'],'-B','-u',str(ROOT/'code/train_ratio.py'),
                    '--variant',arm['variant'],'--batch',str(arm['batch'])]
                for key in ('handoff_root','train_tar','seal_dir','weights','mcunet_root'):
                    command+=['--'+key.replace('_','-'),config[key]]
                command+=['--run-dir',str(run),'--workers','0','--prefetch','1',
                    '--accept-local-content-sha256',config['manifest_sha256']]
                if (run/'checkpoints/checkpoints.json').exists():
                    command.append('--resume')
                stamp=time.time_ns()
                stdout=ROOT/'logs'/f'{arm["name"]}_{stamp}.stdout.log'
                stderr=ROOT/'logs'/f'{arm["name"]}_{stamp}.stderr.log'
                with stdout.open('xb') as out,stderr.open('xb') as err:
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    while proc.poll() is None:
                        progress=api.read_json(run/'progress.json') if (run/'progress.json').exists() else None
                        publish('training',active_run=arm['name'],batch=arm['batch'],worker_pid=proc.pid,
                            progress=progress,stdout=str(stdout),stderr=str(stderr))
                        time.sleep(15)
                if proc.returncode or not (run/'complete.json').exists():
                    publish('needs_attention',active_run=arm['name'],worker_pid=None,
                        returncode=proc.returncode,error=stderr.read_text(encoding='utf-8',errors='replace')[-12000:])
                    return
            results=[]
            for arm in config['experiments']:
                run=ROOT/'runs'/arm['name'];complete=api.read_json(run/'complete.json')
                with (run/'history.csv').open(encoding='utf-8',newline='') as stream:
                    row=next(row for row in csv.DictReader(stream) if int(row['epoch'])==complete['best_epoch'])
                results.append(dict(name=arm['name'],batch=arm['batch'],best_epoch=complete['best_epoch'],
                    completed_epochs=complete['completed_epochs'],primary_view=complete['contract']['primary_development_view'],
                    rgb_top1=float(row['rgb_dev_top1']),rgb_top5=float(row['rgb_dev_top5']),
                    ll_pad_top1=float(row['ll_pad_dev_top1']),ll_pad_top5=float(row['ll_pad_dev_top5'])))
            api.atomic_json(ROOT/'results.json',dict(status='complete',results=results))
            publish('complete',results=results,worker_pid=None)
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        api.atomic_json(ROOT/'queue_status.json',dict(status='needs_attention',error=repr(error),updated_at=time.time()))
        raise
