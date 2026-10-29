"""Atomic JSON snapshots with bounded retries for Windows sharing conflicts."""
import json
import os
from pathlib import Path
import time
import uuid


def publish_snapshot(path, value):
    """Publish an immutable JSON observation; readers never lock a rewritten file."""
    path = Path(path)
    directory = path.parent / (path.stem + '_snapshots')
    directory.mkdir(parents=True, exist_ok=True)
    name = f'{time.time_ns():020d}-{uuid.uuid4().hex}.json'
    target = directory / name
    temporary = directory / (name + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Every destination is new, so this rename never replaces an open file.
        temporary.rename(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def read_latest_snapshot(path):
    path = Path(path)
    directory = path.parent / (path.stem + '_snapshots')
    try:
        latest = max(directory.glob('*.json'), key=lambda item: item.name)
        return json.loads(latest.read_text(encoding='utf-8'))
    except (OSError, ValueError, StopIteration):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            return {}


def write_json(path, value, timeout=5.0):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)
