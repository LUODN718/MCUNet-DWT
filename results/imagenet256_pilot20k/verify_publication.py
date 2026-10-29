"""Read-only SHA256, checkpoint-selection and prediction verification."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    expected, name = line.split('  ', 1)
    path = (root / name).resolve()
    assert path.is_relative_to(root), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
manifest = json.loads((root / 'data/manifest.json').read_text(encoding='utf-8'))
labels = {row['path']: row['label'] for row in manifest['rows'] if row['role'] == 'dev'}
assert len(labels) == 4000
with (root / 'results.csv').open(newline='', encoding='utf-8') as stream:
    results = list(csv.DictReader(stream))
assert len(results) == 10
for result in results:
    folder = root / 'artifacts' / result['job']
    with (folder / 'history.csv').open(newline='', encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    assert [int(row['epoch']) for row in history] == list(range(1, 21))
    best = min(history, key=lambda row: float(row['dev_ce']))
    assert int(best['epoch']) == int(result['selected_epoch'])
    assert float(best['dev_ce']) == float(result['dev_ce'])
    seen = set()
    top1 = top5 = 0
    with (folder / 'dev_predictions.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            image, target = row['image_id'], int(row['true_label'])
            assert image not in seen and labels[image] == target
            seen.add(image)
            ranks = list(map(int, row['pred_top5'].split()))
            assert len(set(ranks)) == 5 and all(0 <= value < 1000 for value in ranks)
            top1 += ranks[0] == target
            top5 += target in ranks
    assert seen == set(labels)
    assert top1 / 40 == float(result['dev_top1'])
    assert top5 / 40 == float(result['dev_top5'])
print('PASS: all published SHA256, 10 complete histories, CE selections and 40,000 predictions')
print('Scope: preliminary development results; binary checkpoint downloads are not included.')
