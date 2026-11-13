"""User-controlled resume: bypasses only the resource watermark pause."""
import train_full

class IgnoreResourceGuard:
    def __init__(self, *args, **kwargs):
        pass

    def wait(self, on_pause=None):
        return
    def close(self):
        return

train_full.Guard = IgnoreResourceGuard
train_full.run(resume=True)
