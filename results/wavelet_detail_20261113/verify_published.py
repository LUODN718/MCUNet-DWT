"""Read-only publication verification using only Python standard libraries."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    manifest = read(ROOT / 'file_manifest.json')['sha256']
    for relative, expected in manifest.items():
        path = ROOT / relative
        assert path.resolve().is_relative_to(ROOT), relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, relative
    results = read(ROOT / 'completed_results.json')
    assert len(results['results']) == 8
    for row in results['results']:
        folder = ROOT / 'records' / row['variant']
        complete = read(folder / 'complete.json')
        with (folder / 'history.csv').open(encoding='utf-8', newline='') as stream:
            history = list(csv.DictReader(stream))
        assert complete['completed_epochs'] == row['completed_epochs'] == 20
        assert len(history) == 20
        selected = min(history, key=lambda r: float(r['dev_ce']))
        assert int(selected['epoch']) == complete['best_epoch'] == row['best_epoch']
        for public, raw in [('dev_ce', 'dev_ce'), ('dev_top1', 'dev_top1'), ('dev_top5', 'dev_top5')]:
            assert float(selected[raw]) == row[public]
    lowres = read(ROOT / 'audit/lowres_compute.json')['rows']
    assert {r['variant']: r['conv_linear_macs'] for r in lowres} == {
        'rgb160': 67287200, 'rgb80': 18021824, 'll80': 18021824}
    cleanup = read(ROOT / 'cleanup/execution_summary.json')
    assert cleanup['status'] == 'complete_and_verified'
    assert cleanup['deleted_files'] == 14 and cleanup['deleted_logical_bytes'] == 71968155
    assert read(ROOT / 'cleanup/verification.json')['verified_retained_records'] == 4090
    removed = read(ROOT / 'cleanup/own_repository_weight_removals.json')['files']
    assert len(removed) == 12 and all(len(row['sha256']) == 64 for row in removed)
    full = read(ROOT / 'full_rgb_official_val/summary.json')
    assert full['official_val_top1'] == 60.23 and full['official_val_top5'] == 82.576
    scope = read(ROOT / 'publication_scope.json')
    assert scope['mixed_padding_training_started'] is False
    print(json.dumps(dict(status='passed', files_verified=len(manifest),
                          completed_arms_verified=8, mixed_padding_training_started=False), indent=2))


if __name__ == '__main__':
    main()
