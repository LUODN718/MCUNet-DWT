"""Derive an isolated within-batch trainer without changing the original run."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'detail_sparsity_pad_mix_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (ROOT / 'queue_config.json').exists():
        raise FileExistsError('already frozen')
    for name in ('code', 'data/seal', 'runs', 'logs', 'cache', 'audit'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    text = (SOURCE / 'code/train_pad_mix.py').read_text(encoding='utf-8')
    def sub(before, after):
        nonlocal text
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    sub('"""Train raw/padded-LL1 mixed or single-view controls on fixed 128k/20k."""',
        '"""Train exactly16 RGB +16 padded LL1 samples in every batch32."""')
    sub('from padded_input import PaddedMix160, view_for_step',
        'from within_batch_input import WithinBatch160')
    sub('choices=("rgb_only", "ll1_pad_only", "rgb_ll1_pad_mix")',
        'choices=("rgb_ll1_within_batch",)')
    sub('model = PaddedMix160(backbone, args.variant).cuda().train()',
        'model = WithinBatch160(backbone).cuda().train()')
    sub('"variant": args.variant, "unit": "whole microbatch32",\n'
        '            "mixed_counts_per_epoch": {"rgb": 2000, "ll1_pad": 2000},\n'
        '            "schedule": "sha256 pad-mix-v1:2026:epoch -> local torch Generator randperm4000",',
        '"variant": args.variant, "unit": "individual samples within every microbatch32",\n'
        '            "counts_per_batch": {"rgb": 16, "ll1_pad": 16},\n'
        '            "mixed_samples_per_epoch": {"rgb": 64000, "ll1_pad": 64000},\n'
        '            "schedule": "sha256 within-batch-pad-v1:2026:committed_step -> local randperm32",')
    sub('"schedule_note": "mixture schedule applies only to rgb_ll1_pad_mix; controls use their sole view",',
        '"schedule_note": "one presentation per original image per epoch; no paired duplicates",')
    sub('("padded_input.py", "w80_tar_source.py", "direct_io.py")',
        '("within_batch_input.py", "w80_tar_source.py", "direct_io.py")')
    sub('                            batch_view = view_for_step(args.variant, progress["step"])\n', '')
    sub('                            progress["train_" + batch_view + "_samples"] += len(labels)',
        '                            progress["train_rgb_samples"] += 16\n'
        '                            progress["train_ll1_pad_samples"] += 16')
    sub('expected_rgb = 64000 if args.variant == "rgb_ll1_pad_mix" else (128000 if args.variant == "rgb_only" else 0)',
        'expected_rgb = 64000')
    assert 'view_for_step' not in text and 'PaddedMix160' not in text
    (ROOT / 'code/train_within_batch.py').write_text(text, encoding='utf-8', newline='\n')
    for name in ('w80_tar_source.py', 'direct_io.py'):
        shutil.copy2(SOURCE / 'code' / name, ROOT / 'code' / name)
    for path in (SOURCE / 'data/seal').iterdir():
        if path.is_file():
            shutil.copy2(path, ROOT / 'data/seal' / path.name)
    config = json.loads((SOURCE / 'queue_config.json').read_text(encoding='utf-8'))
    for name in ('previous_queue', 'cancelled_previous_experiment', 'reference_method', 'user_request'):
        config.pop(name, None)
    config.update(schema=1, created_at=time.time(),
        seal_dir=str(ROOT / 'data/seal'),
        manifest=str(ROOT / 'data/seal/manifest.local_accepted.tsv'),
        run_root=str(ROOT / 'runs'), cache_root=str(ROOT / 'cache'),
        wait_for_run=str(SOURCE / 'runs/p0_rgb_only'),
        wait_for_queue=str(SOURCE),
        wait_for_diagnostic=str(ROOT.parent / 'bn_diagnostic_pad_mix_20261114'),
        experiments=[dict(name='rgb_ll1_within_batch', variant='rgb_ll1_within_batch')],
        original_p1_held=True,
        reference_method='MCUNet160 adaptation, exact16RGB+16LL within shared-BN batch; senior dataset/model unknown',
        selection='lowest full20k LL1-pad development CE; same-epoch RGB secondary',
        sources={})
    sources = sorted(list((ROOT / 'code').glob('*.py')) + [ROOT / 'prepare.py', ROOT / 'queue.py'])
    config['sources'] = {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in sources}
    (ROOT / 'queue_config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'audit/derivation.json').write_text(json.dumps(dict(
        source=str(SOURCE / 'code/train_pad_mix.py'),
        source_sha256=sha(SOURCE / 'code/train_pad_mix.py'),
        derived_sha256=sha(ROOT / 'code/train_within_batch.py'),
        intended_change='whole-batch randomized domains -> exact16/16 domains within each32 batch',
        inherited='data/augment/initial weights/shared BN/optimizer/budget/dual dev/selection/recovery',
        frozen_original_unchanged=True), indent=2)+'\n', encoding='utf-8')
    print('Prepared isolated128k/20k within-batch mixture.')


if __name__ == '__main__':
    main()
