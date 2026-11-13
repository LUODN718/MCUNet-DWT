"""Read-only Windows resource guard; never changes scientific parameters."""
import json
from pathlib import Path
import threading
import time
from run_w80_original_tar import resources, publish


class Guard:
    def __init__(self, root, policy_class):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy_class()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.failures = 0
        self.action = 'pause'
        self.latest_time = 0.
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _sample(self):
        try:
            row = resources(self.root)
            now = time.time()
            action = self.policy.decide(row, now)
            if row['disk_free'] < 10*2**30:
                self.policy.paused = True
                self.policy.healthy_since = None
                action = 'pause'
            with self.lock:
                self.failures = 0
                self.action, self.latest_time = action, now
            # Fixed workers0/prefetch1 are within the frozen allowed range.
            # This Windows adapter does not query Windows driver event logs.
            receipt = dict(time=now, action=action, workers=0, prefetch=1,
                           microbatch=32, bn_batch=32, concurrency=1, cache_bytes=0,
                           driver_event_audit=False, **row)
            publish(self.root/'latest.json', receipt)
            with (self.root/'resources.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(receipt)+'\n')
        except Exception as exc:
            with self.lock:
                self.failures += 1
                self.action = 'pause'
                if self.failures >= 2:
                    self.error = 'Resource telemetry failed twice: '+type(exc).__name__+': '+str(exc)

    def _loop(self):
        while not self.done.wait(30):
            self._sample()

    def wait(self, on_pause=None):
        saved = False
        while True:
            with self.lock:
                error, action, stamp = self.error, self.action, self.latest_time
            if error:
                raise RuntimeError(error)
            if action == 'continue' and time.time()-stamp < 90:
                return
            if on_pause is not None and not saved:
                on_pause()
                saved = True
            if self.done.wait(1):
                raise RuntimeError('Resource guard closed')

    def close(self):
        self.done.set()
        self.thread.join(timeout=15)
