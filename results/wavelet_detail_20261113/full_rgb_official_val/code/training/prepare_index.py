import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_data import prepare
if __name__ == '__main__':
    import json
    print(json.dumps(prepare(os.environ['IMAGENET_TRAIN_TAR'],os.environ['IMAGENET_CLASSES_JSON']),indent=2),flush=True)