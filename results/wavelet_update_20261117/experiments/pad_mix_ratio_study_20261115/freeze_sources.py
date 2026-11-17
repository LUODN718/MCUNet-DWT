"""Freeze source checksums after preparation and before preflight/launch."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'queue_config.json').read_text(encoding='utf-8'))
for arm in config['experiments']:
    arm['variant']=arm['name']
paths=sorted((ROOT/'code').glob('*.py'))+[ROOT/'prepare.py',ROOT/'queue.py',ROOT/'freeze_sources.py']
config['sources']={str(p.relative_to(ROOT)).replace('\\','/'):
    hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(ROOT/'queue_config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
print(json.dumps(config['sources'],indent=2))
