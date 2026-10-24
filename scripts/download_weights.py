#!/usr/bin/env python3
"""Download the official MCUNet-in2 ImageNet config and checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://hanlab18.mit.edu/projects/tinyml/mcunet/release"
FILES = {
    "mcunet-256kb-1mb_imagenet.json": "bf1dcb428e0bc21ad102463ee4764eefddaf4151363f8d43898a1e2e52f80485",
    "mcunet-256kb-1mb_imagenet.pth": "63565b5773a8aa70d75baf5a349f5f384da649804c48a5e7af1f655460158c7d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "weights"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = out_dir / name
        print(f"Downloading {name} -> {target}")
        with urlopen(f"{BASE_URL}/{name}") as response, target.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {target}: {actual} != {expected}")
        print(f"Verified SHA256 {actual}")


if __name__ == "__main__":
    main()

