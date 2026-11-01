"""Real frozen-cohort batch32 CUDA/rollback/restore admission; no formal updates."""
import argparse
import json
from pathlib import Path
import time

import torch

import train_w80_dwt_localcontent as baseline
import w80_tar_source


def exact_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact_state(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=baseline.VARIANTS, required=True)
    parser.add_argument('--channel-indices', type=Path)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--train-tar", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mcunet-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--accept-local-content-sha256")
    args = parser.parse_args()
    baseline.require_fyp_output(args.report_dir)
    args.audit_workers = 2
    args.train_root = None
    modules = baseline.load_protocol_modules(args.handoff_root)
    inputs = baseline.verify_inputs(args, modules)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    runtime = modules["runtime"]
    started = time.time()
    model, optimizer, scheduler, scaler = baseline.model_and_optimizers(args, inputs, modules)
    if args.variant == 'dwt_uncompressed_ll':
        reference = modules['model'].load_pretrained(inputs['config'], inputs['weights'], args.mcunet_root)
        actual = model.backbone.state_dict()
        for key, value in reference.state_dict().items():
            if not key.startswith('first_conv.') and not torch.equal(actual[key].cpu(), value.cpu()):
                raise AssertionError('uncompressed backbone or inherited head was altered: ' + key)
        reference_widths = [b.mobile_inverted_conv.point_linear.conv.in_channels for b in reference.blocks]
        actual_widths = [b.mobile_inverted_conv.point_linear.conv.in_channels for b in model.backbone.blocks]
        if actual_widths != reference_widths:
            raise AssertionError('uncompressed backbone widths changed')
        del reference
    parts = (model, optimizer, scheduler, scaler)
    progress = baseline.initial_progress()
    contract = dict(scope="admission only; no formal training updates", seed=2026,
                    manifest_sha256=inputs["manifest_sha256"], storage=inputs["storage"])
    order = modules["data"].epoch_order(len(inputs["train_rows"]), 2026, 0)
    batches = w80_tar_source.loader(inputs["loader_local"], inputs["train_rows"], "train", order=order)
    images, labels = next(iter(batches))
    if len(labels) != 32:
        raise ValueError("admission must use the real batch32")
    x, y = images.cuda(), labels.cuda()
    torch.cuda.reset_peak_memory_stats()
    initial = runtime.capture(*parts, progress, contract)
    transformed = modules['model'].haar_input(x, baseline.HAAR_VARIANTS.get(args.variant, args.variant))
    expected = ((32,3,160,160) if args.variant == 'rgb_e23' else
                (32,3 if args.variant in ('ll', 'dwt_uncompressed_ll') else 6,80,80))
    if tuple(transformed.shape) != expected:
        raise AssertionError('wrong actual DWT input shape')
    stem_output = model.backbone.first_conv(transformed)
    if tuple(stem_output.shape) != (32,16,80,80):
        raise AssertionError('backbone feature-map layout changed')
    runtime.restore(initial, *parts, contract)
    if args.variant not in ('ll', 'rgb_e23', 'dwt_uncompressed_ll'):
        stem = model.backbone.first_conv
        if torch.count_nonzero(stem.pw.weight[:,3:]):
            raise AssertionError('high-frequency PW not zero initialized')
        model(x).sum().backward()
        if not torch.isfinite(stem.pw.weight.grad).all() or not torch.count_nonzero(stem.pw.weight.grad[:,3:]):
            raise AssertionError('high-frequency columns not learnable')
        runtime.restore(initial, *parts, contract)
    first_loss, _ = runtime.transactional_step(*parts, x, y, progress, contract)
    after_first = runtime.capture(*parts, progress, contract)
    checkpoints = runtime.Checkpoints(args.report_dir / "smoke_checkpoints")
    checkpoints.save(after_first, best=True)
    runtime.restore(initial, *parts, contract)
    replay_loss, _ = runtime.transactional_step(*parts, x, y, progress, contract)
    replay = runtime.capture(*parts, progress, contract)
    if first_loss != replay_loss:
        raise AssertionError("replayed CUDA loss differs")
    if not all(torch.equal(after_first["model"][k], replay["model"][k]) for k in after_first["model"]):
        raise AssertionError("replayed parameters/BN differ")
    if not exact_state(after_first["optimizer"], replay["optimizer"]):
        raise AssertionError("replayed optimizer differs")
    restored = checkpoints.load()
    runtime.restore(restored, *parts, contract)
    if not all(torch.equal(v.cpu(), restored["model"][k]) for k, v in model.state_dict().items()):
        raise AssertionError("saved model/BN restore differs")
    before_failure = runtime.capture(*parts, progress, contract)
    def injected_failure():
        raise RuntimeError("controlled admission interruption after BN forward")
    try:
        runtime.transactional_step(*parts, x, y, progress, contract, fail_after_forward=injected_failure)
        raise AssertionError("controlled interruption was not raised")
    except RuntimeError as error:
        if "controlled admission interruption" not in str(error):
            raise
    after_failure = runtime.capture(*parts, progress, contract)
    if not all(torch.equal(v, after_failure["model"][k]) for k, v in before_failure["model"].items()):
        raise AssertionError("failed batch did not restore model/BN")
    if before_failure["scheduler"] != after_failure["scheduler"]:
        raise AssertionError("failed batch advanced scheduler")
    if not exact_state(before_failure["optimizer"], after_failure["optimizer"]):
        raise AssertionError("failed batch changed optimizer")
    if not torch.equal(before_failure["rng"]["cpu"], after_failure["rng"]["cpu"]):
        raise AssertionError("failed batch changed CPU RNG")
    if not all(torch.equal(a, b) for a, b in zip(before_failure["rng"]["cuda"], after_failure["rng"]["cuda"])):
        raise AssertionError("failed batch changed CUDA RNG")
    report = dict(variant=args.variant, channel_indices_sha256=baseline.sha256(args.channel_indices) if args.channel_indices else None,
                  uncompressed_backbone_and_head_preserved=(args.variant == 'dwt_uncompressed_ll'),
                  actual_input_shape=list(transformed.shape), stem_output_shape=list(stem_output.shape),
                  status="passed", scope="admission only; 0 formal updates", batch=32, bn_batch=32,
                  precision="fp32", amp=scaler.is_enabled(), tf32=torch.backends.cudnn.allow_tf32,
                  original_jpeg_storage="nested tar, no JPEG extraction", manifest_sha256=inputs["manifest_sha256"],
                  replay_loss_exact=True, replay_parameters_and_bn_exact=True, checkpoint_restore_exact=True,
                  controlled_failure_restores_parameters_bn_rng=True, first_loss=first_loss,
                  peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                  seconds=time.time()-started, adapter_sha256=inputs["storage"]["adapter_sha256"],
                  trainer_sha256=baseline.sha256(Path(baseline.__file__)),
                  dataset_alignment=inputs.get("dataset_alignment"),
                  admission_sha256=baseline.sha256(Path(__file__)))
    baseline.atomic_json(args.report_dir / "cuda_admission.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
