"""Derive a placement-only centered-LL comparison from the frozen16/16 run."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'pad_mix_within_batch_20261114'
OLD='rgb_ll1_within_batch'
NEW='rgb_ll1_center_within_batch'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sub(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


def main():
    if (ROOT/'queue_config.json').exists():
        raise FileExistsError('already prepared; do not change frozen sources')
    baseline=json.loads((BASE/'queue_config.json').read_text(encoding='utf-8'))
    for name,digest in baseline['sources'].items():
        assert sha(BASE/name)==digest,name
    for folder in ('code','data/seal','runs','logs','cache','audit','visualization'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    adapter=(BASE/'code/within_batch_input.py').read_text(encoding='utf-8')
    adapter=sub(adapter,'return F.pad(corrected_ll(x),(0,80,0,80),value=0.)',
                'return F.pad(corrected_ll(x),(40,40,40,40),value=0.)')
    (ROOT/'code/within_batch_input.py').write_text(adapter,encoding='utf-8',newline='\n')
    trainer=(BASE/'code/train_within_batch.py').read_text(encoding='utf-8').replace(OLD,NEW)
    trainer=sub(trainer,'"placement": "top-left; right and bottom zero in [0,1] domain",',
                '"placement": "centered; 40 pixels of zero on each side in [0,1] domain",')
    (ROOT/'code/train_within_batch.py').write_text(trainer,encoding='utf-8',newline='\n')
    preflight=(BASE/'code/preflight.py').read_text(encoding='utf-8').replace(OLD,NEW)
    preflight=sub(preflight,
        '    assert ll.shape==x.shape and torch.count_nonzero(ll[...,80:,:])==0\n'
        '    assert torch.count_nonzero(ll[...,:,80:])==0\n'
        '    torch.testing.assert_close(ll[...,:80,:80],corrected_ll(x))\n'
        '    assert torch.all((ll*2-1)[...,80:,:]==-1)\n'
        "    checks.append('Haar LL DC correction and top-left80 block with zero pad before normalization')",
        '    assert ll.shape==x.shape\n'
        '    torch.testing.assert_close(ll[...,40:120,40:120],corrected_ll(x))\n'
        '    outside=torch.ones((160,160),dtype=torch.bool)\n'
        '    outside[40:120,40:120]=False\n'
        '    assert torch.count_nonzero(ll[...,outside])==0\n'
        '    assert torch.all((ll*2-1)[...,outside]==-1)\n'
        '    assert outside.sum()==19200\n'
        "    checks.append('Haar LL unchanged; exact center80 block and40pixel zero border; pad normalizes to-1')")
    (ROOT/'code/preflight.py').write_text(preflight,encoding='utf-8',newline='\n')
    for name in ('w80_tar_source.py','direct_io.py'):
        shutil.copy2(BASE/'code'/name,ROOT/'code'/name)
    for path in (BASE/'data/seal').iterdir():
        if path.is_file():
            shutil.copy2(path,ROOT/'data/seal'/path.name)
    queue=(BASE/'queue.py').read_text(encoding='utf-8').replace(OLD,NEW)
    queue=queue.replace('current_rgb_control','top_left_mixed_control')
    queue=queue.replace('old_rgb_control_finished','top_left_mixed_control_finished')
    queue=queue.replace('current RGB control','top-left16/16 mixed control')
    (ROOT/'queue.py').write_text(queue,encoding='utf-8',newline='\n')
    config=dict(baseline)
    config.update(created_at=time.time(),seal_dir=str(ROOT/'data/seal'),
        manifest=str(ROOT/'data/seal/manifest.local_accepted.tsv'),run_root=str(ROOT/'runs'),
        cache_root=str(ROOT/'cache'),wait_for_run=str(BASE/'runs'/OLD),wait_for_queue=str(BASE),
        experiments=[dict(name=NEW,variant=NEW)],
        reference_method='placement-only ablation: centered40pixel pad vs top-left; same16RGB+16LL batch schedule',
        only_scientific_change='LL80 placement: top-left[0:80,0:80] -> center[40:120,40:120]',
        reference_config_sha256=sha(BASE/'queue_config.json'),sources={})
    paths=sorted(list((ROOT/'code').glob('*.py'))+[ROOT/'prepare.py',ROOT/'queue.py'])
    config['sources']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    (ROOT/'queue_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    review=json.loads((BASE/'diagnostic_review.json').read_text(encoding='utf-8'))
    review.update(launch_method=NEW,reviewed_at=time.time(),
        next_change='centered LL80 instead of top-left; all remaining training settings match16/16 reference',
        placement_evidence='not yet tested; new experiment requested by user after inspecting inputs')
    (ROOT/'diagnostic_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
    # Compare derived sources to prove no hidden optimizer/data/schedule change.
    original_trainer=(BASE/'code/train_within_batch.py').read_text(encoding='utf-8')
    reconstructed=trainer.replace(NEW,OLD).replace(
        '"placement": "centered; 40 pixels of zero on each side in [0,1] domain",',
        '"placement": "top-left; right and bottom zero in [0,1] domain",')
    assert reconstructed==original_trainer
    assert adapter.replace('(40,40,40,40)','(0,80,0,80)')==(BASE/'code/within_batch_input.py').read_text(encoding='utf-8')
    provenance=dict(source_experiment=str(BASE),source_config_sha256=sha(BASE/'queue_config.json'),
        source_sources=baseline['sources'],derived_sources=config['sources'],
        scientific_change=config['only_scientific_change'],
        exact_trainer_diff_checked=True,exact_adapter_diff_checked=True,
        mixture_rng_salt_unchanged=True,all_new_outputs_under_fyp=True)
    (ROOT/'audit/derivation.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('Prepared centered placement-only mixed subset experiment.')


if __name__=='__main__':
    main()
