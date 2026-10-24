#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/weights"
BASE='https://hanlab18.mit.edu/projects/tinyml/mcunet/release'
curl -L --fail --retry 2 "$BASE/mcunet-256kb-1mb_imagenet.json" \
  -o "$ROOT/weights/mcunet-256kb-1mb_imagenet.json"
curl -L --fail --retry 2 "$BASE/mcunet-256kb-1mb_imagenet.pth" \
  -o "$ROOT/weights/mcunet-256kb-1mb_imagenet.pth"
shasum -a 256 "$ROOT/weights/mcunet-256kb-1mb_imagenet.json" \
  "$ROOT/weights/mcunet-256kb-1mb_imagenet.pth"

