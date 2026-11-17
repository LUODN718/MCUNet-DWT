# RGB + LL1 within-batch reproduction attempt (MCUNet160 adaptation)

This is a new experiment, not a replacement for the frozen epoch18/epoch20
mixed-model results. The senior's dataset, class count, architecture, wavelet
implementation and full training recipe are unavailable (user confirmed on
2026-11-14). The 81–85% reference accuracy therefore cannot be a directly
comparable target. This attempt tests the known mixed-training idea on the
existing ImageNet1000 / MCUNet subset.

## Change being tested

The previous mixed run randomly selected an entire batch as RGB or LL1. Its
shared BN moving averages could favor whichever domain occurred near the end
of an epoch. The new run keeps **exactly16 RGB +16 LL1 images in every batch32**,
with deterministic random positions. Every original TRAIN member still appears
once per epoch; images and labels are not duplicated. This tests mixing at the
sample level, rather than alternating whole batches.

This is still **one shared BN set** (BN batch32), not split BN. Balanced batches
reduce changes in domain proportions but do not guarantee matching either pure
inference domain. A separate train-only, per-domain BN diagnosis is retained
independently. A later split-BN experiment would be a separate protocol change.

## Fixed method

- Original frozen membership128k TRAIN +20k dev,1000 classes; same accepted
  local content seal as the prior run. Member identities match the teammate
  cohort, but local image content does not match the teammate's content seal.
- RGB: deterministic RRC160, bicubic+antialias, scale[.08,1], ratio[.75,4/3],
  flip.5. Dev: Resize182+CenterCrop160. No extra augmentation.
- LL1: Haar LL80 divided by2 (DC correction), placed top-left on a160 canvas;
  right/bottom zero in[0,1], then normalize2*x-1 (pad becomes-1). These placement
  and gain choices are explicit assumptions, not verified details of the senior.
- Official MCUNet-in2 pretrained weights and1000-class head, original dense
  stride2 stem and complete backbone. All parameters and BN update.
- FP32, no AMP/TF32, seed2026, batch32, no accumulation. SGD momentum.9,
  WD4e-5 on all parameters, peakLR.001,2warmup epochs included in20, then cosine
  to the fixed20-epoch endpoint.4000updates/epoch. Original plateau rule retained.
- Full20k dev evaluated separately as RGB and LL1-pad. Select first minimum
  LL1-pad CE; RGB is recorded at that same epoch. No official val/V2 use.
- Conv/Linear cost remains **67,287,200MAC/image**,729,392parameters for either
  input. Zero padding does not provide sparse execution or reduce dense CNN MACs.
  Preprocessing, BN, activations and decoding are outside that count.
- Checkpoints retain optimizer/scheduler/RNG/data cursor. No memory watermark
  alarm or memory-triggered pause. All caches and outputs remain in this folder.

## Admission and launch

`preflight.json` checks all80000 deterministic mixture masks, original image
identity/geometry and cursor replay, actual Conv/Linear MACs, a disposable real
32-image CPU optimizer update, transactional rollback, and exact state replay.
These disposable checks do not change formal initialization.

`queue.py` first waits for the epoch18/20 BN diagnosis and its written review,
then for the already-running RGB-only control to finish and release its lock.
The old LL-only arm is held by the prior experiment's `PAUSE_QUEUE`. Only this
new mixed subset arm is scheduled here; full ImageNet training is prohibited.
`queue_status.json` distinguishes waiting, training, completion and errors.

Use `resume_queue.cmd` to resume the queue. The queue prevents duplicate runs
and resumes from the latest committed checkpoint after a recoverable failure.

## What this can establish

Compare against the previous mixed run at the same budget and selection rule.
Improved LL performance or reduced epoch-to-epoch RGB/LL tradeoff would support
sample-level mixing as an improvement in this setting. This alone cannot
reproduce the senior's absolute accuracy, demonstrate hardware acceleration,
or establish that all loss of LL accuracy was due to BN.
