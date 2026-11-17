"""Create three isolated, ratio-only variants based on centered16/16 training."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'pad_mix_center_within_batch_20261114'
ARMS=[('rgb50_ll50','rgb_ll1_ratio50'),('rgb33_ll67','rgb_ll1_ratio33'),('rgb25_ll75','rgb_ll1_ratio25')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sub(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)


def main():
    if (ROOT/'queue_config.json').exists():
        raise FileExistsError('experiment already frozen')
    for name in ('code','data/seal','runs','logs','cache','audit'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    base_source=BASE/'code/train_within_batch.py'
    trainer=base_source.read_text(encoding='utf-8')
    trainer=sub(trainer,'from within_batch_input import WithinBatch160',
                'from ratio_input import RatioMix160, rgb_count')
    trainer=sub(trainer,'choices=("rgb_ll1_center_within_batch",)',
                'choices=("rgb50_ll50", "rgb33_ll67", "rgb25_ll75")')
    trainer=sub(trainer,'model = WithinBatch160(backbone).cuda().train()',
                'model = RatioMix160(backbone,args.variant).cuda().train()')
    trainer=sub(trainer,'"variant": args.variant, "unit": "individual samples within every microbatch32",\n'
        '            "counts_per_batch": {"rgb": 16, "ll1_pad": 16},\n'
        '            "mixed_samples_per_epoch": {"rgb": 64000, "ll1_pad": 64000},\n'
        '            "schedule": "sha256 within-batch-pad-v1:2026:committed_step -> local randperm32",',
        '"variant": args.variant, "unit": "individual samples within every microbatch32",\n'
        '            "target_rgb_fraction": {"rgb50_ll50": 0.50, "rgb33_ll67": 1/3, "rgb25_ll75": 0.25}[args.variant],\n'
        '            "target_ll1_pad_fraction": {"rgb50_ll50": 0.50, "rgb33_ll67": 2/3, "rgb25_ll75": 0.75}[args.variant],\n'
        '            "counts_per_batch": "exact except1:2 uses11,11,10 RGB over a repeating3-update cycle",\n'
        '            "schedule": "sha256 center-ratio-v1:2026:variant:committed_step -> local randperm32",')
    trainer=sub(trainer,'"placement": "centered; 40 pixels of zero on each side in [0,1] domain",',
        '"placement": "centered; 40 pixels of zero on each side in [0,1] domain",')
    trainer=sub(trainer,'"variant": args.variant, "unit": "individual samples within every microbatch32",',
        '"variant": args.variant, "unit": "individual samples within every microbatch32",')
    trainer=sub(trainer,'("within_batch_input.py", "w80_tar_source.py", "direct_io.py")',
                '("ratio_input.py", "w80_tar_source.py", "direct_io.py")')
    trainer=sub(trainer,'progress["train_rgb_samples"] += 16\n'
                '                            progress["train_ll1_pad_samples"] += 16',
        'rgb_n = rgb_count(progress["step"],args.variant)\n'
        '                            progress["train_rgb_samples"] += rgb_n\n'
        '                            progress["train_ll1_pad_samples"] += len(labels)-rgb_n')
    trainer=sub(trainer,'                expected_rgb = 64000\n'
        '                if progress["train_rgb_samples"] != expected_rgb or progress["train_ll1_pad_samples"] != 128000-expected_rgb:',
        '                expected_rgb = sum(rgb_count(step,args.variant) for step in range(epoch*4000,(epoch+1)*4000))\n'
        '                if progress["train_rgb_samples"] != expected_rgb or progress["train_ll1_pad_samples"] != 128000-expected_rgb:')
    # Preserve source provenance while making the row/result variant unique.
    (ROOT/'code/train_ratio.py').write_text(trainer,encoding='utf-8',newline='\n')
    shutil.copy2(BASE/'code/w80_tar_source.py',ROOT/'code/w80_tar_source.py')
    shutil.copy2(BASE/'code/direct_io.py',ROOT/'code/direct_io.py')
    for path in (BASE/'data/seal').iterdir():
        if path.is_file(): shutil.copy2(path,ROOT/'data/seal'/path.name)
    config=json.loads((BASE/'queue_config.json').read_text(encoding='utf-8'))
    config.update(schema=1,created_at=time.time(),seal_dir=str(ROOT/'data/seal'),
        manifest=str(ROOT/'data/seal/manifest.local_accepted.tsv'),run_root=str(ROOT/'runs'),
        cache_root=str(ROOT/'cache'),wait_for_run=str(BASE/'runs/rgb_ll1_center_within_batch'),
        wait_for_queue=str(BASE),wait_for_diagnostic=str(BASE.parent/'bn_diagnostic_pad_mix_20261114'),
        experiments=[dict(name=name,variant=variant,rgb_ratio=n,dense_ratio=d)
            for (name,variant),(n,d) in zip(ARMS,((1,1),(1,2),(1,3)))],
        source_experiment=str(BASE),original_completed_comparison_preserved=True,
        selection='minimum full20k centered LL1-pad dev CE; report RGB160 metrics at same epoch',
        eval_views=['rgb','center_ll1_pad'],ratio_design='RGB:LL ratios1:1,1:2,1:3; evaluate both views')
    sources={}
    for path in sorted((ROOT/'code').glob('*.py')):
        sources[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    sources['prepare.py']=sha(ROOT/'prepare.py')
    config['sources']=sources
    (ROOT/'queue_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    (ROOT/'audit/derivation.json').write_text(json.dumps(dict(
        base_experiment=str(BASE),base_trainer_sha256=sha(base_source),
        base_adapter_sha256=sha(BASE/'code/within_batch_input.py'),
        changed='sample mixture ratio only; centered LL block retained',
        rgb_fraction_by_arm={'rgb50_ll50':'1/2','rgb33_ll67':'1/3','rgb25_ll75':'1/4'},
        rgb11ratio_cycle='11,11,10 RGB samples/update gives exact 1:2 over full80k updates',
        both_eval_views=True,all_outputs_under_fyp=True),indent=2)+'\n',encoding='utf-8')
    print('Prepared three sequential ratio experiments.')


if __name__=='__main__': main()
