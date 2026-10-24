# MCUNet in2 ImageNet weights

Source:

`https://hanlab18.mit.edu/projects/tinyml/mcunet/release/`

Files:

- `mcunet-256kb-1mb_imagenet.json` — network configuration, input resolution 160, 1000 classes.
- `mcunet-256kb-1mb_imagenet.pth` — official ImageNet pretrained PyTorch checkpoint.

The `.pth` file is intentionally ignored by Git because it is a binary artifact.
Run `scripts/download_weights.sh` to fetch it again.

