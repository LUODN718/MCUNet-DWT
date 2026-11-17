# RGB224 and centered padded LL112 experiment

Status: prepared on2026-11-17; live progress is in queue_status.json and runs/*/progress.json.
This is a document-sized adaptation of the prior160 study, not an exact reproduction
of the senior student's unspecified dataset, model or training recipe.

Sequential queue:

| Run | Training views | Batch | Maximum epochs | Best checkpoint criterion |
|---|---|---:|---:|---|
| mix224_rgb33_ll67_b64 | RGB224:LL112-pad224 approximately1:2 |64|20|full20k LL-pad CE|
| rgb224_only_b64 | RGB224 only |64|20|full20k RGB CE|
| mix224_rgb33_ll67_b32 | RGB224:LL112-pad224 approximately1:2 |32|20|full20k LL-pad CE|

Each run starts independently from the same official MCUNet-in2 pretrained
weights with the original dense stride2 stem and full backbone/head. This source
checkpoint was pretrained at160; evaluating/fine-tuning at224 does not create
an independent224-pretrained model. All1000 classes, fixed128k TRAIN/20k dev
membership and accepted local content seal are retained. No full-data training,
official validation or ImageNet-V2 evaluation is scheduled.

Processing: per-image seeded RandomResizedCrop224 (scale .08..1, ratio3/4..4/3,
bicubic antialias) and horizontal flip .5. Dev uses Resize short edge255
(round182*224/160), CenterCrop224. Haar LL1 is DC-corrected by dividing the
orthonormal coefficient by2, then centered with56 pixels of zero in[0,1]
on each side before2*x-1 normalization. Padding therefore becomes-1 in the
network input. No interpolation of LL, extra color augmentation or mixup.
The original document does not specify its mixture ratio, placement,
normalization, model or complete training budget. The user requested1:2 here.

One presentation of each original TRAIN image per epoch; no view duplication.
Batch64 cycles21,21,22 RGB samples and43,43,42 LL samples; batch32 cycles
11,11,10 RGB and21,21,22 LL. Rounding is under one image over the whole run.
All training and evaluation CNN inputs are224x224, including padded LL112.
Padding does not reduce dense CNN compute. preflight.json records measured
224 and native112 Conv/Linear MACs; native112 accuracy is not a padded result.

FP32; AMP/TF32 off; seed2026; all parameters and BN trainable; shared BN;
no gradient accumulation; CE without smoothing; SGD momentum.9, WD4e-5 on
all parameters; LR.001 without batch scaling;2warmup epochs followed by
cosine to fixed epoch20. Batch64 has2000 updates/epoch and batch32 has4000.
Epoch/sample budgets match; update counts and BN batch differ deliberately.
The original minimum6-epoch plus4-observation CE-plateau early-stop rule is retained.
Separate RGB and LL development metrics are recorded every epoch at the same
checkpoint; each run's selection criterion is explicit above.

batch_probe.json contains disposable synthetic GPU compute tests, not measured
end-to-end training speed. Batch64 was selected over32 after faster throughput;
128 exceeded the probe's80%-VRAM allocator cap. The cap is only used by the
disposable probe, not formal training. preflight.py uses disposable real-data
steps to verify exact checkpoint resumption; these weights are discarded.
Formal runs begin again at the pinned official initialization.

No memory-watermark pause/watchdog is installed. Only one trainer runs at a
time; OS locks prevent duplicate queue/run launches. Checkpoints are saved
transactionally every100 updates or60 seconds and at phase boundaries.
On an error the queue records needs_attention and retains the resume point.
Rerun run_queue.py using the same environment to resume; do not launch duplicates.
All new sources, logs, weights, caches and reports are under this fyp directory.
