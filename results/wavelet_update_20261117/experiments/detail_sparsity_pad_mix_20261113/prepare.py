"""Derive an isolated trainer from the validated subset loop; retain provenance."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'detail_sparsity_ll_up160_20261110'
LOWRES = ROOT.parent / 'detail_sparsity_lowres_aug_20261112'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sub(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError('unexpected source match count: ' + before[:120])
    return text.replace(before, after)


def main():
    if (ROOT / 'queue_config.json').exists():
        raise FileExistsError('already prepared; do not replace frozen contracts')
    for folder in ('code', 'data/seal', 'runs', 'logs', 'cache', 'audit', 'visualization'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    source = PREVIOUS / 'code/train_ll.py'
    trainer = source.read_text(encoding='utf-8')
    trainer = sub(trainer, '"""Train LL1/LL2/LL3 resized to 160 on the existing 128k/20k cohort."""',
                  '"""Train raw/padded-LL1 mixed or single-view controls on fixed 128k/20k."""')
    trainer = sub(trainer, 'from lowpass import LL160',
                  'from padded_input import PaddedMix160, view_for_step')
    trainer = sub(trainer, 'parser.add_argument("--level", type=int, choices=(1, 2, 3), required=True)',
                  'parser.add_argument("--variant", choices=("rgb_only", "ll1_pad_only", "rgb_ll1_pad_mix"), required=True)')
    trainer = sub(trainer, 'model = LL160(backbone, args.level).cuda().train()',
                  'model = PaddedMix160(backbone, args.variant).cuda().train()')
    trainer = sub(trainer, '"dev_top5", "selected_best")',
                  '"dev_top5", "rgb_dev_ce", "rgb_dev_top1", "rgb_dev_top5",\n'
                  '              "train_rgb_samples", "train_ll1_pad_samples", "selected_best")')
    trainer = sub(trainer, '        "train_samples": 0,',
                  '        "train_samples": 0,\n        "train_rgb_samples": 0,\n'
                  '        "train_ll1_pad_samples": 0,')
    trainer = sub(trainer, 'f"ll{args.level}_bilinear_up160_uncompressed"', 'args.variant', count=2)
    start = trainer.index('        "lowpass": {')
    end = trainer.index('        "experiment_sources": {', start)
    trainer = trainer[:start] + '''        "input_processing": {
            "wavelet": "orthonormal Haar, level1 LL only, divide coefficient by2 for DC correction",
            "coefficient_size": [80, 80], "canvas_size": [160, 160],
            "placement": "top-left; right and bottom zero in [0,1] domain",
            "normalization_order": "RGB geometric view -> corrected LL -> zero-pad -> 2*x-1; pad becomes -1",
            "upsampling": False, "inference_skip_zero_regions": False,
        },
        "training_mixture": {
            "variant": args.variant, "unit": "whole microbatch32",
            "mixed_counts_per_epoch": {"rgb": 2000, "ll1_pad": 2000},
            "schedule": "sha256 pad-mix-v1:2026:epoch -> local torch Generator randperm4000",
            "extra_updates_or_duplicates": False,
            "schedule_note": "mixture schedule applies only to rgb_ll1_pad_mix; controls use their sole view",
        },
        "primary_development_view": "ll1_pad",
        "secondary_development_view": "rgb",
        "additional_augmentation": None,
        "batch_norm_sharing": "one shared pretrained BN set; no split BN or recalibration",
        "execution_scope": "reference-inspired 160 adaptation with explicit assumptions; not exact224 reproduction or frozen teammate recipe",
''' + trainer[end:]
    trainer = sub(trainer, '("lowpass.py", "w80_tar_source.py", "direct_io.py")',
                  '("padded_input.py", "w80_tar_source.py", "direct_io.py")')
    trainer = sub(trainer, '"checkpoint_selection": "lowest complete development CE; first exact tie",',
                  '"checkpoint_selection": "lowest complete LL1-pad160 development CE; first exact tie; RGB metrics secondary",')
    trainer = sub(trainer, '"model": "MCUNet-in2, LL resized to160, original stride-2 stem, all channels and inherited 1000-class head",',
                  '"model": "MCUNet-in2, RGB160 and/or LL1 zero-padded160, original stride2/full backbone/head",')
    # Evaluate both views in one decode pass. Select on LL-pad CE, record RGB
    # separately. Never update BN or optimize during either development forward.
    start = trainer.index('def evaluate(model, rows, loader, local, args):')
    end = trainer.index('\n\ndef train(args, inputs, modules):', start)
    trainer = trainer[:start] + '''def evaluate(model, rows, loader, local, args):
    model.eval()
    result = {view: dict(count=0, top1=0, top5=0, loss_sum=0.0) for view in ("ll1_pad", "rgb")}
    batches = loader(local, rows, "dev", workers=args.workers, prefetch=args.prefetch)
    with torch.inference_mode():
        for images, labels in batches:
            if len(labels) != BATCH_SIZE:
                raise ValueError("dev batch must remain32")
            images, target = images.cuda(), labels.cuda()
            for view, totals in result.items():
                model.eval_view = view
                logits = model(images)
                losses = F.cross_entropy(logits, target, reduction="none")
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("nonfinite " + view + " development CE")
                pred = logits.topk(5, dim=1).indices
                totals["loss_sum"] += float(losses.double().sum())
                totals["top1"] += int((pred[:, 0] == target).sum())
                totals["top5"] += int((pred == target[:, None]).any(dim=1).sum())
                totals["count"] += len(labels)
    model.last_logits = None
    metrics = {}
    for view, totals in result.items():
        if totals["count"] != DEV_SAMPLES:
            raise ValueError("incomplete " + view + " development pass")
        metrics[view] = dict(ce=totals["loss_sum"]/DEV_SAMPLES,
                             top1=100.0*totals["top1"]/DEV_SAMPLES,
                             top5=100.0*totals["top5"]/DEV_SAMPLES, images=DEV_SAMPLES)
    return metrics
''' + trainer[end:]
    trainer = sub(trainer, '                            loss, _ = modules["runtime"].transactional_step(',
                  '                            model.set_train_step(progress["step"])\n'
                  '                            batch_view = view_for_step(args.variant, progress["step"])\n'
                  '                            loss, _ = modules["runtime"].transactional_step(')
    trainer = sub(trainer, '                            progress["train_samples"] += len(labels)',
                  '                            progress["train_samples"] += len(labels)\n'
                  '                            progress["train_" + batch_view + "_samples"] += len(labels)')
    trainer = sub(trainer, '                    "dev_top5": dev["top5"],',
                  '                    "dev_top5": dev["top5"],\n'
                  '                    "rgb_dev_ce": metrics["rgb"]["ce"],\n'
                  '                    "rgb_dev_top1": metrics["rgb"]["top1"],\n'
                  '                    "rgb_dev_top5": metrics["rgb"]["top5"],\n'
                  '                    "train_rgb_samples": progress["train_rgb_samples"],\n'
                  '                    "train_ll1_pad_samples": progress["train_ll1_pad_samples"],')
    trainer = sub(trainer, '                progress["train_samples"] = 0',
                  '                progress["train_samples"] = 0\n'
                  '                progress["train_rgb_samples"] = 0\n'
                  '                progress["train_ll1_pad_samples"] = 0')
    trainer = sub(trainer, '                dev = evaluate(model, dev_rows, loader,\n                               local, args)',
                  '                metrics = evaluate(model, dev_rows, loader, local, args)\n'
                  '                dev = metrics["ll1_pad"]')
    trainer = sub(trainer, '                             dev_ce=dev["ce"], selected_best=selected_best,',
                  '                             dev_ce=dev["ce"], rgb_dev_ce=metrics["rgb"]["ce"],\n'
                  '                             selected_best=selected_best,')
    trainer = sub(trainer, '                      f"dev_top1={dev[\'top1\']:.3f}, best={selected_best}", flush=True)',
                  '                      f"LL1_pad_top1={dev[\'top1\']:.3f}, "\n'
                  '                      f"RGB_top1={metrics[\'rgb\'][\'top1\']:.3f}, best={selected_best}", flush=True)')
    trainer = sub(trainer, '                                "training_exposures": progress["training_exposures"],\n                                "checkpoint_sha256":',
                  '                                "training_exposures": progress["training_exposures"],\n'
                  '                                "train_rgb_samples": progress["train_rgb_samples"],\n'
                  '                                "train_ll1_pad_samples": progress["train_ll1_pad_samples"],\n'
                  '                                "checkpoint_sha256":')
    trainer = sub(trainer, '                append_event(run_dir / "events.jsonl", "dev_start",',
                  '''                expected_rgb = 64000 if args.variant == "rgb_ll1_pad_mix" else (128000 if args.variant == "rgb_only" else 0)
                if progress["train_rgb_samples"] != expected_rgb or progress["train_ll1_pad_samples"] != 128000-expected_rgb:
                    raise ValueError("complete epoch mixture count differs from declared contract")
                atomic_json(run_dir / "progress.json", {
                    "status": "evaluating_dev", "epoch": epoch + 1, "step": progress["step"],
                    "training_exposures": progress["training_exposures"],
                    "sampler_position": progress["sampler_position"],
                    "train_rgb_samples": progress["train_rgb_samples"],
                    "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
                })
                append_event(run_dir / "events.jsonl", "dev_start",''')
    trainer = sub(trainer, '    last_save = time.monotonic()\n    try:',
                  '''    atomic_json(run_dir / "progress.json", {
        "status": "training", "epoch": progress["epoch"] + 1, "step": progress["step"],
        "sampler_position": progress["sampler_position"], "training_exposures": progress["training_exposures"],
        "train_rgb_samples": progress["train_rgb_samples"],
        "train_ll1_pad_samples": progress["train_ll1_pad_samples"],
    })
    last_save = time.monotonic()
    try:''')
    assert 'args.level' not in trainer and 'lowpass.py' not in trainer and 'bilinear' not in trainer
    (ROOT / 'code/train_pad_mix.py').write_text(trainer, encoding='utf-8', newline='\n')
    for name in ('w80_tar_source.py', 'direct_io.py'):
        shutil.copy2(PREVIOUS / 'code' / name, ROOT / 'code' / name)
    for source in (PREVIOUS / 'data/seal').iterdir():
        if source.is_file():
            shutil.copy2(source, ROOT / 'data/seal' / source.name)
    config = json.loads((LOWRES / 'queue_config.json').read_text(encoding='utf-8'))
    for name in ('reused_evidence', 'reused_ll80_plain', 'reused_rgb160_plain'):
        config.pop(name, None)
    config.update(schema=3, created_at=time.time(),
        experiments=[dict(name='p2_rgb_ll1_pad_mix', variant='rgb_ll1_pad_mix'),
                     dict(name='p0_rgb_only', variant='rgb_only'),
                     dict(name='p1_ll1_pad_only', variant='ll1_pad_only')],
        seal_dir=str(ROOT / 'data/seal'), manifest=str(ROOT / 'data/seal/manifest.local_accepted.tsv'),
        run_root=str(ROOT / 'runs'), cache_root=str(ROOT / 'cache'), conv_linear_macs=67287200,
        wait_for_run=str(LOWRES / 'runs/rgb80_light'), previous_queue=str(LOWRES),
        cancelled_previous_experiment='rgb80_plain',
        reference_method='independent160 adaptation; top-left corrected Haar LL, zero before normalization',
        primary_metric='minimum complete LL1-pad160 development CE',
        user_request='2026-11-13: finish current run; cancel RGB80-plain; then start today padded-LL mixed plan')
    config['sources'] = {p.relative_to(ROOT).as_posix(): sha(p)
                         for p in sorted((ROOT / 'code').glob('*.py'))}
    config['sources'].update({'queue.py': sha(ROOT / 'queue.py'), 'prepare.py': sha(ROOT / 'prepare.py')})
    (ROOT / 'queue_config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    (ROOT / 'source_derivation.json').write_text(json.dumps(dict(
        inherited_trainer=str(source), inherited_sha256=sha(source),
        actual_trainer_sha256=sha(ROOT / 'code/train_pad_mix.py'),
        original_data_adapter_sha256=sha(PREVIOUS / 'code/w80_tar_source.py'),
        differences=['LL padding instead of interpolation', 'deterministic50:50 RGB/LL batch routing',
                     'dual complete development views with LL-pad CE selection',
                     'mixture counts/progress; no extra augmentations or changes to optimization']), indent=2)+'\n', encoding='utf-8')
    (ROOT / 'resume_queue.cmd').write_text('@echo off\r\ncd /d "%~dp0"\r\n"'+config['python']+
        '" -B -u "%~dp0queue.py"\r\npause\r\n', encoding='utf-8', newline='')
    print(json.dumps(dict(status='prepared', experiments=config['experiments']), indent=2))


if __name__ == '__main__':
    main()
