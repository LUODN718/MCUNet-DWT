# Centered LL80 mixture-ratio study

Compare RGB160:LL80-centered-zero-pad training ratios1:1,1:2,1:3. All samples
are presented to the unchanged MCUNet as160x160 tensors. LL80 is DC-corrected
Haar, placed at [40:120,40:120] with40px zero border before2*x-1 normalization.
One original TRAIN image is shown once per epoch; only its assigned view changes.

Each ratio gets an independent copy of the same official pretrained initialization,
the same fixed128kTRAIN/20kdev subset, geometry and flip, no extra augmentation,
batch/effective BN32, FP32, SGD/learning schedule and20-epoch maximum. Every
complete20k dev pass evaluates both RGB160 and centered LL80-pad160. Checkpoints
are selected on centered LL dev CE; RGB is recorded at the same selected epoch.
No official validation or ImageNet-V2 is used.

For the1:2 arm, batch size32 cannot represent exactly one-third RGB per batch.
The schedule cycles11,11,10 RGB samples in every three updates, yielding exactly
one-third RGB and two-thirds LL over the full80,000 updates. The1:1 and1:3 arms
have exact per-batch proportions. See preflight.json for checked global counts.

This is an input-mixture and accuracy experiment, **not** a compute-saving test:
both views use the full160 dense backbone and cost67,287,200 Conv/Linear MAC/image.
The lower-resolution branch does not skip CNN operations. Outputs, checkpoints,
and caches remain in this fyp folder. The arms train sequentially to avoid
concurrent GPU contention; `queue_status.json` reports the active arm.

Results characterize the existing subset and local accepted image-content seal.
They are not an exact content-matched teammate reproduction. Ratios are the only
intended variable. The existing centered16/16 experiment is retained separately
as the matched1:1 reference.
