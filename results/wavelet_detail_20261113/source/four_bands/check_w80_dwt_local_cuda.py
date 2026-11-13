"""Real frozen-cohort batch32 CUDA/rollback/restore admission; no formal updates."""
import argparse
import json
from pathlib import Path
import time

import torch
import numpy as np

import train_w80_dwt_localcontent as baseline
import fourband_model
import w80_tar_source


def exact_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
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
    reference = modules['model'].load_pretrained(inputs['config'], inputs['weights'], args.mcunet_root)
    initialization = fourband_model.check_initialization(model, reference)
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
    transformed = fourband_model.haar_input(x)
    expected = (32,12,80,80)
    if tuple(transformed.shape) != expected:
        raise AssertionError('wrong actual DWT input shape')
    stem_output = model.backbone.first_conv(transformed)
    if tuple(stem_output.shape) != (32,16,40,40):
        raise AssertionError('stride2 four-band stem must produce 32x16x40x40')
    torch.testing.assert_close(transformed[:, :3], torch.nn.functional.avg_pool2d(x, 2)*2-1, rtol=0, atol=1e-6)
    for offset, name in ((3, 'll_lh'), (6, 'll_hl'), (9, 'll_hh')):
        expected_hf = modules['model'].haar_input(x, name)[:, 3:]
        torch.testing.assert_close(transformed[:, offset:offset+3], expected_hf, rtol=0, atol=0)
    torch.testing.assert_close(fourband_model.inverse_haar(transformed), x, rtol=0, atol=1e-6)
    ll, hf = fourband_model.stem_parameters(model)
    if len(optimizer.param_groups) != 2:
        raise AssertionError('inherited and new HF parameters require two optimizer groups')
    old_group, new_group = optimizer.param_groups
    if (old_group['initial_lr'], new_group['initial_lr']) != (.001, .005):
        raise AssertionError('inherited/new-HF peak learning rates differ from .001/.005')
    if {id(p) for p in new_group['params']} != {id(hf)} or id(ll) not in {id(p) for p in old_group['params']}:
        raise AssertionError('optimizer weight-column partition is incorrect')
    grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(grouped) != len(set(grouped)) or set(grouped) != {id(p) for p in model.parameters()}:
        raise AssertionError('optimizer omits or duplicates model parameters')
    runtime.restore(initial, *parts, contract)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    for band in range(3):
        grad = hf.grad[:, band*3:(band+1)*3]
        if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
            raise AssertionError('a high-frequency band is not learnable')
    runtime.restore(initial, *parts, contract)
    first_loss, _ = runtime.transactional_step(*parts, x, y, progress, contract)
    if not torch.count_nonzero(hf):
        raise AssertionError('new HF weights did not update on the first CUDA step')
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
    # The enlarged stem has optimizer-only weight parametrization. Confirm that
    # a newly constructed model/optimizer restores it and continues identically.
    next_loss, _ = runtime.transactional_step(*parts, x, y, progress, contract)
    expected_next = runtime.capture(*parts, progress, contract)
    fresh_parts = baseline.model_and_optimizers(args, inputs, modules)
    fresh_progress = runtime.restore(restored, *fresh_parts, contract)
    fresh_loss, _ = runtime.transactional_step(*fresh_parts, x, y, fresh_progress, contract)
    fresh_next = runtime.capture(*fresh_parts, fresh_progress, contract)
    if next_loss != fresh_loss or not all(exact_state(expected_next[k], fresh_next[k])
                                          for k in ('model', 'optimizer', 'scheduler', 'rng')):
        raise AssertionError('fresh model/optimizer checkpoint continuation differs')
    del fresh_parts, fresh_next, expected_next
    runtime.restore(restored, *parts, contract)
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
                  uncompressed_backbone_and_head_preserved=True,
                  actual_input_shape=list(transformed.shape), stem_output_shape=list(stem_output.shape),
                  initialization=initialization, new_hf_parameters=1296,
                  inherited_stem_peak_lr=.001, new_hf_peak_lr=.005,
                  normalized_ll_matches_average_pool=True,
                  all_three_high_frequency_bands_learnable=True,
                  inverse_haar_restores_input=True, distillation=False,
                  optimizer_partition_verified=True,
                  fresh_model_optimizer_continuation_exact=True,
                  status="passed", scope="admission only; 0 formal updates", batch=32, bn_batch=32,
                  precision="fp32", amp=scaler.is_enabled(), tf32=torch.backends.cudnn.allow_tf32,
                  original_jpeg_storage="nested tar, no JPEG extraction", manifest_sha256=inputs["manifest_sha256"],
                  replay_loss_exact=True, replay_parameters_and_bn_exact=True, checkpoint_restore_exact=True,
                  controlled_failure_restores_parameters_bn_rng=True, first_loss=first_loss,
                  peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                  seconds=time.time()-started, adapter_sha256=inputs["storage"]["adapter_sha256"],
                  trainer_sha256=baseline.sha256(Path(baseline.__file__)),
                  model_implementation_sha256=baseline.sha256(Path(fourband_model.__file__)),
                  dataset_alignment=inputs.get("dataset_alignment"),
                  admission_sha256=baseline.sha256(Path(__file__)))
    baseline.atomic_json(args.report_dir / "cuda_admission.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
