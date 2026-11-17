"""Create isolated source copies; never modify previous experiments or recipes."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'detail_sparsity_ll_up160_20261110'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def substitute(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise ValueError(f'expected {count} source occurrences, found {actual}: {before[:100]}')
    return text.replace(before, after)


def main():
    if (ROOT / 'queue_config.json').exists():
        raise FileExistsError('already prepared; inspect rather than overwrite')
    for folder in ('code', 'data/seal', 'runs', 'logs', 'cache', 'audit'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    trainer = (PREVIOUS / 'code/train_ll.py').read_text(encoding='utf-8')
    trainer = substitute(trainer, '"""Train LL1/LL2/LL3 resized to 160 on the existing 128k/20k cohort."""',
                         '"""Train matched direct-RGB80 / LL80 subset arms with mild augmentation."""')
    trainer = substitute(trainer, 'from lowpass import LL160',
                         'from low_resolution import LowResolution80\nfrom augmentation import SETTINGS as AUGMENTATION_SETTINGS')
    trainer = substitute(trainer, 'parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)',
                         'parser.add_argument("--variant", choices=("rgb80", "ll80"), required=True)\n'
                         '    parser.add_argument("--augmentation", choices=("plain", "light"), required=True)')
    trainer = substitute(trainer, 'model = LL160(backbone, args.level).cuda().train()',
                         'model = LowResolution80(backbone, args.variant).cuda().train()')
    trainer = substitute(trainer, 'local = inputs["loader_local"]',
                         'local = {**inputs["loader_local"], "extra_augmentation": args.augmentation}')
    trainer = substitute(trainer, 'f"ll{args.level}_bilinear_up160_uncompressed"',
                         'f"{args.variant}_{args.augmentation}_uncompressed"', count=2)
    start = trainer.index('        "lowpass": {')
    end = trainer.index('        "experiment_sources": {', start)
    trainer = trainer[:start] + '''        "input_processing": {
            "variant": args.variant, "rgb_view": [3, 160, 160],
            "backbone_input": [3, 80, 80], "stem_output": [16, 40, 40],
            "downsample": ("Haar LL / 2, no upsampling" if args.variant == "ll80"
                           else "bicubic antialias to80; clamp [0,1]"),
            "original_stem_stride": [2, 2],
        },
        "augmentation": args.augmentation,
        "extra_augmentation": AUGMENTATION_SETTINGS if args.augmentation == "light" else None,
        "execution_scope": "independent low-resolution augmentation experiment; not frozen teammate recipe",
''' + trainer[end:]
    trainer = substitute(trainer, '("lowpass.py", "w80_tar_source.py", "direct_io.py")',
                         '("low_resolution.py", "augmentation.py", "w80_tar_source.py", "direct_io.py")')
    trainer = substitute(trainer, '"model": "MCUNet-in2, LL resized to160, original stride-2 stem, all channels and inherited 1000-class head",',
                         '"model": "MCUNet-in2; RGB80 or LL80 directly; original stride2 stem and full inherited backbone/head",')
    trainer = substitute(trainer, '"conv_linear_macs_per_image": 67287200,',
                         '"conv_linear_macs_per_image": 18021824,')
    # Clear current phase in the file users inspect; no stale training label during dev.
    needle = '                append_event(run_dir / "events.jsonl", "dev_start",'
    trainer = substitute(trainer, needle, '''                atomic_json(run_dir / "progress.json", {
                    "status": "evaluating_dev", "epoch": epoch + 1,
                    "step": progress["step"], "sampler_position": progress["sampler_position"],
                    "training_exposures": progress["training_exposures"],
                })
''' + needle)
    trainer = substitute(trainer, '    last_save = time.monotonic()\n    try:',
                         '''    atomic_json(run_dir / "progress.json", {
        "status": "training", "epoch": progress["epoch"] + 1,
        "step": progress["step"], "sampler_position": progress["sampler_position"],
        "training_exposures": progress["training_exposures"],
    })
    last_save = time.monotonic()
    try:''')
    assert 'args.level' not in trainer and 'lowpass.py' not in trainer
    (ROOT / 'code/train_lowres.py').write_text(trainer, encoding='utf-8', newline='\n')

    adapter = (PREVIOUS / 'code/w80_tar_source.py').read_text(encoding='utf-8')
    adapter = substitute(adapter, 'def __init__(self, archive, index_path, rows, role, seed=2026, epoch=0):',
                         'def __init__(self, archive, index_path, rows, role, seed=2026, epoch=0, augmentation="plain"):')
    adapter = substitute(adapter, '        self.role, self.seed, self.epoch = role, seed, epoch',
                         '        self.role, self.seed, self.epoch = role, seed, epoch\n'
                         '        if augmentation not in ("plain", "light"):\n'
                         '            raise ValueError(augmentation)\n'
                         '        self.augmentation = augmentation')
    adapter = substitute(adapter, '        return F.to_tensor(im), int(r["label"])',
                         '        from augmentation import augment_rgb_view\n'
                         '        mode = self.augmentation if self.role == "train" else "plain"\n'
                         '        return augment_rgb_view(im, self.seed, self.epoch, r["path"], mode), int(r["label"])')
    adapter = substitute(adapter, 'dataset = TarOriginalImages(local["archive"], local["index_path"], rows, role, seed=2026, epoch=epoch)',
                         'dataset = TarOriginalImages(local["archive"], local["index_path"], rows, role, seed=2026, epoch=epoch,\n'
                         '                                augmentation=local.get("extra_augmentation", "plain"))')
    (ROOT / 'code/w80_tar_source.py').write_text(adapter, encoding='utf-8', newline='\n')
    shutil.copy2(PREVIOUS / 'code/direct_io.py', ROOT / 'code/direct_io.py')
    for source in (PREVIOUS / 'data/seal').iterdir():
        if source.is_file():
            shutil.copy2(source, ROOT / 'data/seal' / source.name)

    config = json.loads((PREVIOUS / 'queue_config.json').read_text(encoding='utf-8'))
    for name in ('levels', 'wait_for_run', 'baseline_complete', 'baseline_complete_sha256'):
        config.pop(name, None)
    config.update(schema=2, created_at=time.time(),
                  experiments=[dict(name='ll80_light', variant='ll80', augmentation='light'),
                               dict(name='rgb80_light', variant='rgb80', augmentation='light'),
                               dict(name='rgb80_plain', variant='rgb80', augmentation='plain')],
                  seal_dir=str(ROOT / 'data/seal'),
                  manifest=str(ROOT / 'data/seal/manifest.local_accepted.tsv'),
                  run_root=str(ROOT / 'runs'), cache_root=str(ROOT / 'cache'),
                  reused_ll80_plain=str(ROOT.parent / 'dwt_ll_stride2_full20_20261101/run'),
                  reused_rgb160_plain=str(ROOT.parents[1] / 'runs/w80_rgb160_localcontent_128k'),
                  conv_linear_macs=18021824)
    config['reused_evidence'] = {
        str(Path(config[name]) / leaf): sha(Path(config[name]) / leaf)
        for name in ('reused_ll80_plain', 'reused_rgb160_plain')
        for leaf in ('complete.json', 'history.csv', 'config.json')
    }
    config['sources'] = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                         for p in sorted((ROOT / 'code').glob('*.py'))}
    config['sources']['queue.py'] = sha(ROOT / 'queue.py')
    config['sources']['prepare.py'] = sha(ROOT / 'prepare.py')
    (ROOT / 'queue_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'resume_queue.cmd').write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n"' + config['python'] +
        '" -B -u "%~dp0queue.py"\r\npause\r\n', encoding='utf-8', newline='')
    print(json.dumps(dict(status='prepared', experiment_root=str(ROOT),
                          experiments=config['experiments']), indent=2))


if __name__ == '__main__':
    main()
