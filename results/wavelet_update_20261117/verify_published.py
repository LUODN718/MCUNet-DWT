"""Offline integrity and source-to-result checks; no training dependencies."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    hashes=read(ROOT/'file_manifest.json')['sha256']
    for relative,digest in hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    summary=read(ROOT/'results_summary.json')['rows']
    assert len(summary)==9
    for row in summary:
        run=ROOT/'experiments'/row['experiment']/'runs'/row['run']
        complete=read(run/'complete.json')
        with (run/'history.csv').open(encoding='utf-8-sig',newline='') as stream:
            history=list(csv.DictReader(stream))
        assert len(history)==complete['completed_epochs']==20
        selected=min(history,key=lambda item:float(item['dev_ce']))
        assert int(selected['epoch'])==row['best_epoch']==complete['best_epoch']
        assert float(selected['dev_ce'])==row['primary_ce']
    native=read(ROOT/'native80_eval/results.json')
    assert native['status']=='complete'
    for result in native['results']:
        row=next(row for row in summary if row['experiment']=='pad_mix_ratio_study_20261115' and row['run']==result['variant'])
        assert row['native80_top1']==result['native_ll80']['top1']
        assert row['ll_pad160_top1']==result['existing_ll_pad160']['top1']
        assert row['rgb160_top1']==result['existing_rgb160']['top1']
        assert result['native_ll80']['input_shape']==[3,80,80]
        assert result['native_ll80']['images']==20000
    bn=read(ROOT/'bn_diagnostic/results.json')
    assert bn['all_nonbn_tensors_unchanged'] and bn['calibration_train_only']
    plan=read(ROOT/'queued224/study_config.json')
    for relative,digest in plan['sources'].items():
        assert hashlib.sha256((ROOT/'queued224'/relative).read_bytes()).hexdigest()==digest,relative
    assert [arm['batch'] for arm in plan['experiments']]==[64,64,32]
    assert read(ROOT/'queued224/preflight.json')['status']=='passed'
    print(json.dumps(dict(status='passed',files=len(hashes),completed_runs=9,native80_evaluations=3)))


if __name__=='__main__':main()
