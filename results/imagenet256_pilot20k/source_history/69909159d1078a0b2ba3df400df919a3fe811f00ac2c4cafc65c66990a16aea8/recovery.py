"""Operational recovery without changing model/optimizer/data semantics."""
import errno
import logging
import os
from pathlib import Path
import time


def recoverable(error):
    if isinstance(error, (MemoryError, PermissionError)):
        return True
    if type(error).__name__ == 'OutOfMemoryError' and type(error).__module__.startswith('torch'):
        return True
    return isinstance(error, OSError) and (
        error.errno in (errno.EACCES, errno.EBUSY, errno.ENOMEM)
        or getattr(error, 'winerror', None) in (5, 32, 33, 1450, 1455))


def best_effort_publish(publisher, path, value):
    """Observability must never become a training or queue failure."""
    try:
        publisher(path, value)
        return True
    except Exception as error:
        logging.warning('Optional status unavailable: %s', type(error).__name__)
        return False


def run_child(command, folder, band, publish, stop_path, popen, sleep=time.sleep,
              max_restarts=8, max_no_progress=3):
    """Restart recoverable exit75; never swallow invalid data/checkpoint failures."""
    import json
    from .atomic_io import read_latest_snapshot
    failures = 0
    unchanged = 0
    previous_generation = None
    log_path = folder.parent / (folder.name + '.console.log')
    pointer = folder / band / 'checkpoints/checkpoints.json'
    while True:
        if stop_path.exists():
            return 'stopped', None
        with log_path.open('a', encoding='utf-8') as log:
            child = popen(command, stdout=log, stderr=-2)
            publish('running', child_pid=child.pid, restart_count=failures)
            code = child.wait()
        if stop_path.exists():
            return 'stopped', code
        if code == 0 and (folder / band / 'summary.json').exists():
            return 'complete', code
        if code != 75:
            return 'fatal', code
        failures += 1
        generation = None
        if pointer.exists():
            generation = json.loads(pointer.read_text())['latest']
        observation = read_latest_snapshot(folder / 'status.json')
        progress_key = observation.get('successful_updates', generation)
        unchanged = unchanged + 1 if progress_key == previous_generation else 0
        previous_generation = progress_key
        receipt = dict(restart=failures, exit_code=code, latest_checkpoint=generation,
                       progress_key=progress_key, no_progress_retries=unchanged, at=time.time())
        # Retain a durable recovery record, independent of optional display.
        with (folder / 'automatic_restarts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(receipt) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if failures > max_restarts or unchanged >= max_no_progress:
            return 'exhausted', code
        stale_lock = folder / '.training.lock'
        if stale_lock.exists():
            recorded = int(stale_lock.read_text().strip())
            if recorded != child.pid:
                return 'foreign_lock', code
            stale_lock.rename(folder / ('.training.lock.recovered-' + str(failures)))
        delay = min(60, 10 * failures)
        publish('recovering', exit_code=code, restart_count=failures,
                retry_seconds=delay, latest_checkpoint=generation)
        sleep(delay)


def keep_awake(active):
    """Prevent idle sleep only while the training controller owns this request."""
    if os.name != 'nt':
        return False
    try:
        import ctypes
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint32]
        function.restype = ctypes.c_uint32
        return bool(function(0x80000001 if active else 0x80000000))
    except Exception:
        return False
