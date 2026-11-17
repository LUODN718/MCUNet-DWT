# Centered LL80 placement comparison

User requested a centered-padding experiment on2026-11-14 after viewing the
current top-left input. The LL80 block is placed at **[40:120,40:120]** in the
160canvas, with40pixels of zero on all four sides. Padding is performed before
2*x-1 normalization, so the border entering the CNN is-1. The LL coefficients
and gain are identical to the top-left experiment; nothing is upsampled.

## Matched comparison

Reference: `../pad_mix_within_batch_20261114`, exact16RGB+16LL per batch32.
Both runs independently inherit the same official pretrained model. Only LL
placement changes. `audit/derivation.json` verifies the exact trainer/adapter
differences. The image/geometry order, mixture masks and RNG salt,128kTRAIN/20kdev,
original dense stride2/full backbone, shared BN32,FP32, SGD momentum.9/WD4e-5,
peakLR.001,2warmup+cosine within20epochs,4000updates/epoch, original augmentation,
checkpoint/earlystop/selection rules and no officialval/V2 are identical.

Full dev is evaluated separately as RGB160 and **centered LL1-pad160**. Select
the first minimum centered-LL dev CE and record RGB at that same epoch. Compare
against the reference's first minimum top-left-LL CE at the same maximum budget;
record actual completion/earlystop. No result is available yet.

Original old whole-batch mixture is not the placement-only control because it
also differs in how the batch is mixed. The new top-left16/16 run is the direct
control. The senior's exact dataset/model/padding details are unknown, so neither
placement is claimed to be their original implementation.

Centering might change boundary/position effects and compatibility with center
cropped object distributions. Improvement is an experimental question, not a
guarantee. It preserves the same160x160 tensor and **67,287,200 Conv/LinearMAC**,
729,392parameters. This does not implement sparse acceleration. MAC count excludes
Haar/pad/normalization/BN/activation and decoding.

## Execution

Preflight checks exact center region/zero border, unchanged LL coefficients,
80,000 deterministic16/16 masks, actual input decoding/cursor replay, actual
Conv/Linear cost, real32image disposable optimizer update, rollback and exact
state replay. Formal initialization is not changed by admission checks.

The queue waits for the top-left16/16 mixed run to finish and release its locks,
then automatically starts this centered run. The existing RGB control finishes
first through that reference queue. Old LL-only remains held. This is one new
subset experiment, not fullImageNet training. No memory watermark alarm/pause.
All artifacts/caches remain under this fyp directory. `resume_queue.cmd` resumes
the queue and uses the latest committed checkpoint when needed.

Visualization: `visualization/rgb_left_center_comparison.png`, with identical
TRAIN crops displayed as RGB / top-left LL / centered LL. Individual exports are
native160PNG. Training transforms are applied online; no new dataset copy is made.
