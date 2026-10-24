# MCUNet DWT ImageNet Baseline

This repository contains the ImageNet-1K fine-tuning baseline for comparing
raw RGB input with one-level orthonormal Haar subband inputs. The six groups
are `rgb`, `ll`, `ll_lh`, `ll_hl`, `ll_hh` and `all`.

The model is the ImageNet `mcunet-in2` checkpoint. RGB images are cropped to
160x160. Wavelet groups are transformed online after RGB augmentation and are
fed to the adapted first convolution at 80x80. The first convolution uses
stride 1 for wavelet groups so that its output and every later backbone feature
map keep the same spatial sizes as the 160x160 RGB model. Its first three input
channels are initialized from the RGB MCUNet stem; all added high-frequency
columns start at zero. All parameters and batch-normalization layers are
fine-tuned.

The subband order is `[LL/RGB, LH/RGB, HL/RGB, HH/RGB]`. Coefficient signs are
preserved. RGB input uses the official MCUNet normalization `(x-0.5)/0.5`.
For wavelet input, LL is mapped as `(LL/2-0.5)/0.5`; signed high-frequency
coefficients use `band/0.5`. These choices are recorded in every run's
`config.json`.

## Setup

The repository is intentionally kept separate from the upstream MCUNet source.
From this directory, set:

```bash
export MCUNET_ROOT=../mcunet
python -m pip install -r requirements.txt
```

The matching configuration and checkpoint are already downloaded locally:

```text
weights/mcunet-256kb-1mb_imagenet.json
weights/mcunet-256kb-1mb_imagenet.pth
```

The original ImageNet files are not included. Point `--data-root` to an
ImageFolder-style directory containing `train/` or `val/` class folders.

On Windows PowerShell, download the matching files to an explicit directory:

```powershell
cd D:\MCUNet\MCUNet-DWT
python .\scripts\download_weights.py `
  --out-dir D:\MCUNet\MCUNet-DWT\weights
```

The upstream MCUNet source is expected at `D:\MCUNet\mcunet`. If it is stored
elsewhere, pass `--mcunet-root` explicitly to the training and evaluation
commands.

For a fresh Windows CUDA environment:

```powershell
conda create -n mcunet-dwt python=3.10 -y
conda activate mcunet-dwt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r D:\MCUNet\MCUNet-DWT\requirements.txt
git clone https://github.com/mit-han-lab/mcunet.git D:\MCUNet\mcunet
```

## Smoke test

```bash
python scripts/smoke_test.py
```

This checks all six input shapes and confirms that every adapted model produces
1000 ImageNet logits.

## Train one group

The first run creates a deterministic stratified development split. Reuse the
same manifest for every group so the comparisons use identical samples:

```bash
python scripts/train_imagenet.py \
  --data-root /path/to/imagenet/train \
  --variant rgb \
  --run-dir runs/rgb \
  --split-manifest splits/imagenet_train_dev_seed2026.json \
  --epochs 20 --batch-size 256 --workers 8 --amp
```

Run the remaining groups by changing `--variant` and `--run-dir`:

```bash
for v in ll ll_lh ll_hl ll_hh all; do
  python scripts/train_imagenet.py \
    --data-root /path/to/imagenet/train \
    --variant "$v" \
    --run-dir "runs/$v" \
    --split-manifest splits/imagenet_train_dev_seed2026.json \
    --epochs 20 --batch-size 256 --workers 8 --amp
done
```

Each run writes `best.pt`, `last.pt`, `history.csv` and `config.json`.
`best.pt` is selected by the lowest development cross-entropy, and the official
validation set is never used for model selection.

The same training command in Windows PowerShell is:

```powershell
cd D:\MCUNet\MCUNet-DWT
$env:MCUNET_ROOT = "D:\MCUNet\mcunet"
python .\scripts\train_imagenet.py `
  --data-root D:\datasets\imagenet\train `
  --variant rgb `
  --run-dir D:\MCUNet\MCUNet-DWT\runs\rgb `
  --split-manifest D:\MCUNet\MCUNet-DWT\splits\imagenet_train_dev_seed2026.json `
  --config D:\MCUNet\MCUNet-DWT\weights\mcunet-256kb-1mb_imagenet.json `
  --weights D:\MCUNet\MCUNet-DWT\weights\mcunet-256kb-1mb_imagenet.pth `
  --mcunet-root D:\MCUNet\mcunet `
  --epochs 20 --batch-size 256 --workers 8 --amp
```

For the five wavelet groups, keep the same paths and change only `--variant`
and `--run-dir`. The validation directory should be
`D:\datasets\imagenet\val`.

## Evaluate the official validation set

```bash
python scripts/eval_imagenet.py \
  --data-root /path/to/imagenet/val \
  --variant ll \
  --checkpoint runs/ll/best.pt \
  --predictions runs/ll/val_predictions.csv \
  --batch-size 256 --workers 8
```

The evaluation writes Top-1/Top-5 accuracy and a per-image CSV containing the
relative image path, true class, predicted Top-1 class and predicted Top-5 IDs.

Windows PowerShell example:

```powershell
python .\scripts\eval_imagenet.py `
  --data-root D:\datasets\imagenet\val `
  --variant ll `
  --checkpoint D:\MCUNet\MCUNet-DWT\runs\ll\best.pt `
  --predictions D:\MCUNet\MCUNet-DWT\runs\ll\val_predictions.csv `
  --config D:\MCUNet\MCUNet-DWT\weights\mcunet-256kb-1mb_imagenet.json `
  --weights D:\MCUNet\MCUNet-DWT\weights\mcunet-256kb-1mb_imagenet.pth `
  --mcunet-root D:\MCUNet\mcunet `
  --batch-size 256 --workers 8
```

