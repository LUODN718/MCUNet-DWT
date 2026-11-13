"""Read saved SHA256 records only; never rerun a classifier or alter membership."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def packed_hex(values):
    return np.array([bytes.fromhex(value) for value in values], dtype='V32')


def main():
    metrics = json.loads((ROOT / 'metrics.json').read_text())
    assert metrics['complete'] and metrics['samples'] == 50000
    raw, rgb = [], []
    for path in sorted((ROOT / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            raw.extend(data['raw_sha256'].tolist())
            rgb.extend(data['rgb_sha256'].tolist())
    assert len(raw) == len(rgb) == 50000
    val_raw, val_rgb = packed_hex(raw), packed_hex(rgb)
    store = np.load(ROOT.parent / 'data/content_hashes.npy', mmap_mode='r')
    assert len(store) == 1281167 and np.all(store['valid'] == 1)
    train_raw = np.ascontiguousarray(store['raw']).view('V32').reshape(-1)
    train_rgb = np.ascontiguousarray(store['rgb']).view('V32').reshape(-1)
    result = dict(scope='exact raw JPEG SHA256 and decoded size+RGB SHA256, saved records only',
                  training_images=len(store), val_images=len(raw),
                  val_duplicate_byte_rows=len(raw)-len(np.unique(val_raw)),
                  val_duplicate_rgb_rows=len(rgb)-len(np.unique(val_rgb)),
                  val_rows_matching_train_bytes=int(np.isin(val_raw, train_raw).sum()),
                  val_rows_matching_train_rgb=int(np.isin(val_rgb, train_rgb).sum()),
                  perceptual_near_duplicate_audit=False, removed_or_reassigned_images=0,
                  evaluation_rerun=False, checkpoint_unchanged=True, finished_at=time.time())
    (ROOT / 'content_overlap_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
