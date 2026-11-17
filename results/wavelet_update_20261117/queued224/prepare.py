"""Derive isolated 224px trainers without altering completed 160px sources."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'pad_mix_ratio_study_20261115'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (ROOT / 'study_config.json').exists():
        raise FileExistsError('already prepared')
    for name in ('code', 'runs', 'logs', 'cache/temp', 'audit'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    text = (BASE / 'code/train_ratio.py').read_text(encoding='utf-8')
    def sub(before, after):
        nonlocal text
        assert text.count(before) == 1, (before, text.count(before))
        text = text.replace(before, after)
    sub('"""Train exactly16 RGB +16 padded LL1 samples in every batch32."""',
        '"""224px RGB control and RGB:LL112-centered-pad224 1:2 study."""')
    sub('from ratio_input import RatioMix160, rgb_count', 'from ratio_input import RatioMix224, rgb_count')
    text = re.sub(r'\b160\b', '224', text)
    for before, after in [('RGB160','RGB224'),('RandomResizedCrop160','RandomResizedCrop224'),
                          ('pad160','pad224'),('zero-padded160','zero-padded224')]:
        text = text.replace(before, after)
    sub('choices=("rgb50_ll50", "rgb33_ll67", "rgb25_ll75")', 'choices=("rgb33_ll67", "rgb_only")')
    sub('    parser.add_argument("--handoff-root",',
        '    parser.add_argument("--batch", type=int, choices=(32,64), required=True)\n    parser.add_argument("--handoff-root",')
    sub('    args = parse_args(argv)',
        '    global BATCH_SIZE, UPDATES_PER_EPOCH\n    args = parse_args(argv)\n    BATCH_SIZE = args.batch\n    UPDATES_PER_EPOCH = TRAIN_SAMPLES // BATCH_SIZE')
    sub('model = RatioMix160(backbone,args.variant).cuda().train()',
        'model = RatioMix224(backbone,args.variant,batch=BATCH_SIZE).cuda().train()')
    sub('torch.optim.lr_scheduler.LambdaLR(optimizer, modules["runtime"].lr_factor)',
        'torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: modules["runtime"].lr_factor(update, steps_per_epoch=UPDATES_PER_EPOCH))')
    sub('"coefficient_size": [80, 80]', '"coefficient_size": [112, 112]')
    sub('"placement": "centered; 40 pixels of zero on each side in [0,1] domain"',
        '"placement": "centered; 56 pixels of zero on each side in [0,1] domain"')
    sub('"target_rgb_fraction": {"rgb50_ll50": 0.50, "rgb33_ll67": 1/3, "rgb25_ll75": 0.25}[args.variant]',
        '"target_rgb_fraction": 1. if args.variant == "rgb_only" else 1/3')
    sub('"target_ll1_pad_fraction": {"rgb50_ll50": 0.50, "rgb33_ll67": 2/3, "rgb25_ll75": 0.75}[args.variant]',
        '"target_ll1_pad_fraction": 0. if args.variant == "rgb_only" else 2/3')
    sub('"counts_per_batch": "exact except1:2 uses11,11,10 RGB over a repeating3-update cycle"',
        '"counts_per_batch": "RGB control: all; mixture64:21,21,22 RGB; mixture32:11,11,10 RGB; repeating3-update cycle"')
    text = text.replace('microbatch32', 'microbatch defined by batch field')
    text = text.replace('local randperm32', 'local randperm(batch)')
    text = text.replace('with microbatch 32', 'with declared microbatch')
    sub('"primary_development_view": "ll1_pad"', '"primary_development_view": "rgb" if args.variant == "rgb_only" else "ll1_pad"')
    sub('"execution_scope": "reference-inspired 224 adaptation with explicit assumptions; not exact224 reproduction or frozen teammate recipe"',
        '"execution_scope": "document-sized224 adaptation; original data/model/recipe unknown; not exact reproduction or frozen teammate recipe"')
    sub('"checkpoint_selection": "lowest complete LL1-pad224 development CE; first exact tie; RGB metrics secondary"',
        '"checkpoint_selection": "lowest complete primary-view dev CE: RGB for control, LL-pad224 for mixture; all views recorded same epoch"')
    sub('"resize_short_edge": 182', '"resize_short_edge": 255')
    sub('"conv_linear_macs_per_image": 67287200',
        '"conv_linear_macs_per_image": read_json(Path(__file__).resolve().parents[1] / "preflight.json")["conv_linear_macs_224"]')
    sub('    local = inputs["loader_local"]', '    local = {**inputs["loader_local"], "batch": BATCH_SIZE}')
    sub('            if len(labels) != BATCH_SIZE:\n                raise ValueError("dev batch must remain32")',
        '            if not 0 < len(labels) <= BATCH_SIZE:\n                raise ValueError("invalid dev batch")')
    text = text.replace('training BN/effective batch must remain 32', 'training BN/effective batch differs from declared batch')
    text = text.replace('rgb_count(progress["step"],args.variant)', 'rgb_count(progress["step"],args.variant,BATCH_SIZE)')
    sub('sum(rgb_count(step,args.variant) for step in range(epoch*4000,(epoch+1)*4000))',
        'sum(rgb_count(step,args.variant,BATCH_SIZE) for step in range(epoch*UPDATES_PER_EPOCH,(epoch+1)*UPDATES_PER_EPOCH))')
    sub('                dev = metrics["ll1_pad"]',
        '                dev = metrics["rgb" if args.variant == "rgb_only" else "ll1_pad"]')
    sub('"dev_top5", "rgb_dev_ce", "rgb_dev_top1", "rgb_dev_top5",',
        '"dev_top5", "rgb_dev_ce", "rgb_dev_top1", "rgb_dev_top5",\n              "ll_pad_dev_ce", "ll_pad_dev_top1", "ll_pad_dev_top5",')
    sub('                    "rgb_dev_ce": metrics["rgb"]["ce"],',
        '                    "ll_pad_dev_ce": metrics["ll1_pad"]["ce"],\n                    "ll_pad_dev_top1": metrics["ll1_pad"]["top1"],\n                    "ll_pad_dev_top5": metrics["ll1_pad"]["top5"],\n                    "rgb_dev_ce": metrics["rgb"]["ce"],')
    sub('f"LL1_pad_top1={dev[\'top1\']:.3f}, "', 'f"LL1_pad_top1={metrics[\'ll1_pad\'][\'top1\']:.3f}, "')
    (ROOT / 'code/train_ratio.py').write_text(text, encoding='utf-8', newline='\n')
    loader = (BASE / 'code/w80_tar_source.py').read_text(encoding='utf-8')
    assert loader.count('(160, 160)') == 1 and loader.count('F.resize(im, 182,') == 1
    loader = loader.replace('(160, 160)', '(224, 224)').replace('F.resize(im, 182,', 'F.resize(im, 255,').replace('[160, 160]', '[224, 224]')
    loader = loader.replace('batch_size=32,', 'batch_size=local.get("batch",64),')
    (ROOT / 'code/w80_tar_source.py').write_text(loader, encoding='utf-8', newline='\n')
    shutil.copy2(BASE / 'code/direct_io.py', ROOT / 'code/direct_io.py')
    config = json.loads((BASE / 'queue_config.json').read_text(encoding='utf-8'))
    keep = ('python','handoff_root','train_tar','seal_dir','manifest','manifest_sha256','weights','pretrained_sha256','mcunet_root')
    config = {key: config[key] for key in keep}
    config.update(train_images=128000, dev_images=20000, maximum_epochs=20,
        source_experiment=str(BASE), canvas=224, coefficient=112, placement='centered56px',
        dev_resize_short_edge=255, dev_resize_reason='round(182*224/160); same relative crop extent as160 recipe',
        batch_decision='64: faster synthetic throughput than32 with GPU headroom;128 OOM at80% probe cap',
        primary_selection='LL-pad224 CE for mixtures; RGB224 CE for RGB-only control',
        official_val_used=False, resource_watermark_pause=False,
        assumptions=['document does not specify original dataset/model/training recipe',
                     '1:2 mixture requested by user; center placement retained; normalized LL uses DC correction',
                     'same .001 peak LR at batch32/64; epoch-based2warmup+18cosine; no linear LR scaling'],
        experiments=[dict(name='mix224_rgb33_ll67_b64',variant='rgb33_ll67',batch=64),
                     dict(name='rgb224_only_b64',variant='rgb_only',batch=64),
                     dict(name='mix224_rgb33_ll67_b32',variant='rgb33_ll67',batch=32)])
    (ROOT / 'study_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    (ROOT / 'audit/derivation.json').write_text(json.dumps(dict(base=str(BASE),
        base_trainer_sha256=sha(BASE/'code/train_ratio.py'), base_loader_sha256=sha(BASE/'code/w80_tar_source.py'),
        changes=['224 crop;112LL;56centerpad','Resize255 dev','batch64 with32 comparison',
                 'epoch-based scheduler scaled to actual updates/epoch','RGB224 control selects RGB CE'],
        original160_sources_unchanged=True),indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
