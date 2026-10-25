# Tiny ImageNet Pipeline Test

This is a 200-class development experiment, not the required ImageNet-1K
benchmark. Do not report its accuracy as ImageNet-1K accuracy. The original
64x64 images are augmented/cropped to 160x160 RGB. An online one-level
orthonormal Haar transform then produces 80x80 subbands. No second image
dataset is written to disk.

## Protocol

- Groups: `rgb`, `ll`, `ll_lh`, `ll_hl`, `ll_hh`, `all`.
- Sorted WordNet IDs from `wnids.txt` define one common 0-based class mapping.
- Each class supplies 475 train and 25 development images: 95,000/5,000 total.
- The separate 10,000-image validation set is read using `val_annotations.txt`.
  The unlabeled test set is not used. Validation never selects a checkpoint.
- RGB uses random resized crop (scale 0.08 to 1.0, bicubic) and horizontal
  flip. Development/validation use bicubic resize to 182 and center crop 160.
  All RGB augmentation precedes DWT; signed coefficients are retained.
- Order: `[LL/RGB, LH/RGB, HL/RGB, HH/RGB]`.
- Haar LL is `(a+b+c+d)/2`; input RGB values are in `[0,1]`. Normalize RGB
  with `(x-0.5)/0.5`, LL with `(LL/2-0.5)/0.5`, and high bands with `band/0.5`.
- All groups load the same official MCUNet-in2 ImageNet checkpoint. The
  1000-class head is replaced by an identically seeded 200-class head.
- RGB retains its stride-2 first convolution. Wavelet stems use stride 1,
  copy the pretrained first three input columns and BN, and initialize all
  added high-frequency columns to zero. Later feature-map sizes match RGB.
- Full Tiny runs: seed 2026, 20 epochs, SGD, momentum 0.9, LR 0.001,
  weight decay 0.00004, cosine decay, batch 32, 2 workers, CUDA AMP.
  All parameters and BN are fine-tuned. Select `best.pt` by minimum dev CE.
- Shared split JSON and the corresponding `.samples.csv` record every image
  ID, target and train/dev membership. A fingerprint detects mismatched data.

## Local Commands

From this repository in PowerShell:

```powershell
# Six-group pipeline test: two interrupted/resumed partial passes per group.
.\scripts\run_tiny.ps1 -Mode Test

# Full Tiny ImageNet LL training. This is still a preliminary 200-class run.
.\scripts\run_tiny.ps1 -Mode Train -Variant ll

# Resume with the same total budget/settings.
.\scripts\run_tiny.ps1 -Mode Train -Variant ll -Resume

# Evaluate the selected checkpoint on all 10,000 Tiny validation images.
.\scripts\run_tiny.ps1 -Mode Eval -Variant ll
```

Repeat `Train`/`Eval` with each variant to compare all six on Tiny ImageNet.
The launcher uses the existing CUDA-enabled `dgfnet` environment and the local
upstream MCUNet checkout. Override `-Python` and `-McunetRoot` for other paths.
It does not replace the base environment's CPU PyTorch installation.

## Artifacts And Limits

The completed six-group experiment, verified validation predictions, portable
checkpoints and plots are published in [results/tiny_imagenet](results/tiny_imagenet/README.md).
All six groups completed 20 epochs and were evaluated on all 10,000 validation
images. The small-batch pipeline test below is a separate debugging tool.

Pipeline outputs are under `runs/tiny_pipeline_test/`. Each group has
`config.json`, `class_mapping.json`, `train.log`, `history.csv`, `curves.png`,
`best.pt`, `last.pt`, validation predictions and a metrics JSON. The common
`pipeline_results.csv` leaves ImageNet official-validation columns empty.

The pipeline test processes only two training batches, one development batch,
and two validation batches per pass/group (batch 16). Two truncated passes do
not constitute two completed dataset epochs; their accuracy is diagnostic
only. It verifies real updates, high-band training, checkpoint selection,
scheduler/scaler/RNG restoration, Windows workers and prediction export.

Full runs instead write `runs/tiny_imagenet/<variant>/`. Checkpoints include
optimizer, scheduler, scaler, RNG, loader-generator states, class mapping and
dataset fingerprint. For resume, keep the original epoch budget and settings;
`--stop-after-epochs` can interrupt cleanly without changing cosine scheduling.
CUDA kernels may still prevent bitwise reproducibility across hardware.

The existing `results_template.csv` remains reserved for the ImageNet-1K
experiment. A full ImageNet-1K experiment requires its own data and evaluation;
the Tiny development split and validation set do not substitute for it.
