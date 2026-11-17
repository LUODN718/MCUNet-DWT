# Native LL80 evaluation

This supplement evaluates the actual 3x80x80 LL tensor with the unchanged
MCUNet backbone. It does not pad or resize LL back to 160.

- Development data: the same fixed 20,000 TRAIN-side development images and
  content-checked local manifest used by the ratio experiments.
- Processing: Resize short edge 182, bicubic with antialias, CenterCrop160,
  level-one Haar LL with /2 DC correction (2x2 average), then 2*x-1.
- Checkpoint: each completed arm's existing best checkpoint selected by full
  centered-pad160 development CE. No native80 checkpoint reselection occurs.
- BN: existing shared checkpoint statistics, eval mode, no recalibration and
  no training. BN buffers are checked for changes.
- Precision: FP32, no AMP or TF32. Evaluation batches contain 32 images.
- Cost: measured Conv2d and Linear MACs from actual 80 and 160 forwards. These
  exclude image decode, crop/resize, Haar, normalization and other operations.
  MAC savings are not a measurement of latency or energy savings.

The standalone evaluator only reads training checkpoints and writes here.
With --watch it evaluates each remaining arm after that arm completes;
it does not alter the frozen trainer, queue, training history or checkpoints.
stdout.log contains progress; progress.json shows evaluation/waiting status;
results.json aggregates completed evaluations. Per-arm JSON files contain
checkpoint hashes, accuracy, CE, MACs and the corresponding recorded 160 scores.

These are transfer evaluations: the model was trained with RGB160 and padded
LL80 at 160, so native80 also changes spatial context and the BN input domain.
The evaluation is not evidence of training at native80 resolution.
