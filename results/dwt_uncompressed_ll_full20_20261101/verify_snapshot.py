"""Check this publication without torch, CUDA, training or system changes."""
import ast,csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    p=(root/name).resolve()
    assert p.is_relative_to(root)
    with p.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, name
    count+=1
for p in (root/'code').rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'))
s=json.loads((root/'summary.json').read_text())
c=json.loads((root/'evidence/complete.json').read_text())
with (root/'history.csv').open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==20 and int(rows[-1]['updates'])==80000==c['updates']
best=min(rows,key=lambda row:float(row['dev_ce']))
assert int(best['epoch'])==s['best_epoch']==c['best_epoch']
assert float(best['dev_ce'])==s['dev_ce']==c['best_dev_ce']
assert float(best['dev_top1'])==s['dev_top1']
comparison=json.loads((root/'comparison.json').read_text())
assert all(comparison['common_config_fields'].values())
assert not s['official_val_evaluated'] and not s['teammate_content_seal_matched']
for item in json.loads((root/'SOURCE_RECEIPTS.json').read_text()):
    p=root/item['published']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256']
    if item['transformation']=='exact': assert item['original_sha256']==item['published_sha256']
print(f'PASS: {count} archived files; 20 epochs; 80000 updates; selected dev metrics and provenance verified.')
