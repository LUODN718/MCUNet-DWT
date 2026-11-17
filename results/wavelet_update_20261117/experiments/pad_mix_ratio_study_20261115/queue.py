"""Run three centered-padded mixture ratios sequentially from identical init."""
import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'detail_sparsity_pad_mix_20261113'
spec=importlib.util.spec_from_file_location('prior_queue_utils',PRIOR/'queue.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)


def validate(config):
    assert len(config['experiments'])==3
    assert [(a['name'],a['dense_ratio']) for a in config['experiments']]==[
        ('rgb50_ll50',1),('rgb33_ll67',2),('rgb25_ll75',3)]
    assert (config['train_images'],config['dev_images'],config['maximum_epochs'])==(128000,20000,20)
    for relative,digest in config['sources'].items():
        assert util.sha(ROOT/relative)==digest,relative
    assert util.sha(config['manifest'])==config['manifest_sha256']
    admission=util.read(ROOT/'preflight.json')
    assert admission['status']=='passed'


def main():
    config=util.read(ROOT/'queue_config.json');validate(config)
    state=dict(queue_pid=os.getpid(),status='waiting_for_previous_experiment',
        train_images=128000,dev_images=20000,maximum_epochs_each=20,
        variants=config['experiments'],full_training_launch_allowed=False,
        resource_watermark_pause=False)
    def update(status,**extra):
        state.update(status=status,updated_at=time.time(),**extra)
        util.publish(ROOT/'queue_status.json',state)
    with util.lock(ROOT/'queue.lock'):
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            previous=Path(config['wait_for_run'])
            while not ((previous/'complete.json').exists() and
                util.lock_is_free(previous.with_name(previous.name+'.training.lock')) and
                util.lock_is_free(Path(config['wait_for_queue'])/'queue.lock')):
                update('waiting_for_previous_experiment',worker_pid=None,
                    previous_progress=util.read(previous/'progress.json') if (previous/'progress.json').exists() else None)
                time.sleep(15)
            prior=util.read(previous/'complete.json')
            assert prior['completed_epochs']==20
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
            for key,name in [('TEMP','temp'),('TMP','temp'),('TORCH_HOME','torch'),
                ('CUDA_CACHE_PATH','cuda'),('MPLCONFIGDIR','matplotlib')]:
                folder=ROOT/'cache'/name;folder.mkdir(parents=True,exist_ok=True);env[key]=str(folder)
            for arm in config['experiments']:
                run=ROOT/'runs'/arm['name']
                if (run/'complete.json').exists():continue
                validate(config)
                command=[config['python'],'-B','-u',str(ROOT/'code/train_ratio.py'),
                    '--variant',arm['variant']]
                for key in ('handoff_root','train_tar','seal_dir','weights','mcunet_root'):
                    command+=['--'+key.replace('_','-'),config[key]]
                command+=['--run-dir',str(run),'--workers','0','--prefetch','1',
                    '--accept-local-content-sha256',config['manifest_sha256']]
                if (run/'checkpoints/checkpoints.json').exists():command.append('--resume')
                elif run.exists() and any(run.iterdir()):raise RuntimeError('nonempty run without checkpoint')
                for attempt in range(4):
                    if (ROOT/'PAUSE_QUEUE').exists():
                        update('paused_before_next_variant',next_variant=arm['name'],worker_pid=None);return 0
                    stamp=time.time_ns();out_path=ROOT/'logs'/f'{arm["name"]}_{stamp}.stdout.log'
                    err_path=ROOT/'logs'/f'{arm["name"]}_{stamp}.stderr.log'
                    with out_path.open('xb') as out,err_path.open('xb') as err:
                        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        while proc.poll() is None:
                            progress=run/'progress.json'
                            update('training',variant=arm['name'],worker_pid=proc.pid,attempt=attempt,
                                progress=util.read(progress) if progress.exists() else None,
                                stdout=str(out_path),stderr=str(err_path))
                            time.sleep(15)
                    if proc.returncode==0 and (run/'complete.json').exists():break
                    tail=err_path.read_text(encoding='utf-8',errors='replace')[-16000:]
                    recoverable='CUDA out of memory' in tail or 'PermissionError' in tail
                    if not recoverable or attempt==3 or not (run/'checkpoints/checkpoints.json').exists():
                        update('needs_attention',variant=arm['name'],worker_pid=None,
                               exit_code=proc.returncode,error_tail=tail);return proc.returncode or 1
                    update('retrying_from_checkpoint',variant=arm['name'],worker_pid=None,error_tail=tail)
                    time.sleep(30)
                else:
                    return 1
            summaries=[]
            for arm in config['experiments']:
                run=ROOT/'runs'/arm['name'];complete=util.read(run/'complete.json')
                with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:
                    import csv
                    row=next(r for r in csv.DictReader(stream) if int(r['epoch'])==complete['best_epoch'])
                summaries.append(dict(variant=arm['name'],best_epoch=complete['best_epoch'],
                    ll_center_ce=float(row['dev_ce']),ll_center_top1=float(row['dev_top1']),
                    ll_center_top5=float(row['dev_top5']),rgb_ce=float(row['rgb_dev_ce']),
                    rgb_top1=float(row['rgb_dev_top1']),rgb_top5=float(row['rgb_dev_top5']),
                    updates=complete['updates'],conv_linear_macs=67287200))
            util.publish(ROOT/'results.json',dict(status='complete',results=summaries,
                selection='minimum complete centered LL1-pad CE; RGB recorded same epoch',
                official_val_used=False,imagenet_v2_used=False))
            with (ROOT/'results.csv').open('w',encoding='utf-8',newline='') as stream:
                import csv
                writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
            update('complete',worker_pid=None,results=summaries)
            return 0
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        util.publish(ROOT/'queue_status.json',dict(status='needs_attention',queue_pid=os.getpid(),
            error=repr(error),updated_at=time.time()));raise
