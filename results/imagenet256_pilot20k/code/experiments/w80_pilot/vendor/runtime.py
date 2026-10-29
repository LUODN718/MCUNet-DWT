"""Transactional updates and durable immutable checkpoint generations.

No process discovery, signalling, scheduler mutation or historical worker use.
"""
import copy
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .data import sha


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_dir(path.parent)


def fsync_dir(path):
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def capture(model, optimizer, scheduler, scaler, progress, contract):
    return dict(schema=1, model=cpu_copy(model.state_dict()), model_training=model.training,
                optimizer=cpu_copy(optimizer.state_dict()), scheduler=cpu_copy(scheduler.state_dict()),
                scaler=cpu_copy(scaler.state_dict()), rng=rng_state(),
                progress=copy.deepcopy(progress), contract=copy.deepcopy(contract))


def restore(state, model, optimizer, scheduler, scaler, contract):
    if state['schema'] != 1 or state['contract'] != contract:
        raise ValueError('checkpoint manifest/protocol/source mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler']); restore_rng(state['rng'])
    model.train(state['model_training']); optimizer.zero_grad(set_to_none=True)
    return copy.deepcopy(state['progress'])


class Checkpoints:
    """One fsynced pointer commits latest, best and previous together.

    Generations are content-addressed. Uncommitted files after a crash are safe
    orphans, never selected implicitly. Explicit fallback needs a recovery receipt.
    """
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def save(self, state, best=False):
        temp = self.root / (uuid.uuid4().hex+'.partial')
        with temp.open('xb') as f:
            torch.save(state, f); f.flush(); os.fsync(f.fileno())
        checksum = sha(temp)
        name = checksum+'.pt'
        os.replace(temp, self.root/name); fsync_dir(self.root)
        pointer = self.root/'checkpoints.json'
        old = json.loads(pointer.read_text()) if pointer.exists() else {}
        refs = dict(latest=name, previous=old.get('latest'),
                    best=name if best else old.get('best'), updated=time.time())
        durable_json(pointer, refs)
        # Only delete our verified old content-addressed generations. Three
        # referenced checkpoints remain; arbitrary files are never removed.
        keep = {refs[k] for k in ('latest','previous','best')}
        for key in ('latest','previous','best'):
            stale = old.get(key)
            if stale and stale not in keep and self._valid_name(stale):
                p = self.root/stale
                if p.exists() and sha(p) == stale[:-3]:
                    p.unlink()
        return name

    @staticmethod
    def _valid_name(name):
        return len(name) == 67 and name.endswith('.pt') and all(c in '0123456789abcdef' for c in name[:-3])

    def load(self, which='latest'):
        if which not in ('latest','best','previous'):
            raise ValueError(which)
        name = json.loads((self.root/'checkpoints.json').read_text())[which]
        if not name or not self._valid_name(name) or sha(self.root/name) != name[:-3]:
            raise ValueError('checkpoint corruption; inspect before explicit fallback')
        # Only our local hash-verified file, never an untrusted upload.
        return torch.load(self.root/name, map_location='cpu', weights_only=False)


def lr_factor(update, steps_per_epoch=4000):
    warmup, total = 2*steps_per_epoch, 20*steps_per_epoch
    if update < warmup:
        return (update+1)/warmup
    # Last scheduled update reaches zero, using the same fixed horizon per arm.
    return .5*(1+math.cos(math.pi*(update-warmup)/(total-warmup-1)))


class EarlyStop:
    def __init__(self):
        self.best, self.anchor, self.bad, self.history = float('inf'), float('inf'), 0, []

    def observe(self, ce):
        if not math.isfinite(ce):
            raise FloatingPointError('nonfinite complete-dev CE')
        best = ce < self.best
        self.best = min(self.best, ce)
        if self.anchor-ce > .001:
            self.anchor, self.bad = ce, 0
        else:
            self.bad += 1
        self.history.append(ce)
        window = self.history[-4:]
        slope = float(np.polyfit(np.arange(4), window, 1)[0]) if len(window) == 4 else float('inf')
        plateau = len(window) == 4 and abs(slope) <= .001 and max(window)-min(window) <= .005
        stop = len(self.history) >= 6 and self.bad >= 4 and plateau
        return best, stop


def transactional_step(model, optimizer, scheduler, scaler, x, y, progress, contract,
                       fail_after_forward=None):
    """Batch=32 without accumulation in formal training. Roll back BN and RNG
    as well as optimizer after OOM/nonfinite/other failure; no skipped update.
    Caller keeps the same sampler cursor and audits failed attempts separately.
    """
    before = capture(model, optimizer, scheduler, scaler, progress, contract)
    try:
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=x.device.type, enabled=scaler.is_enabled()):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
        if fail_after_forward:
            fail_after_forward()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite CE')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite gradient; no implicit scaler skip')
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
            raise FloatingPointError('nonfinite updated state')
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), before
    except BaseException:
        restore(before, model, optimizer, scheduler, scaler, contract)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        raise


class ResourcePolicy:
    """Only loader resources adapt; microbatch/BN/precision stay fixed.

    Concurrency=1, cache=0 always. Watermarks refer to host and GPU available
    bytes sampled by a separate watchdog. No review-age pause exists.
    """
    def __init__(self):
        self.workers, self.prefetch = 0, 1
        self.healthy_since, self.last_change = None, -float('inf')
        self.paused = False

    def decide(self, sample, now):
        if sample.get('driver_fault') or sample.get('query_error'):
            return 'protect'
        host = min(sample['host_available'], sample.get('windows_available', float('inf')))
        gpu = sample['gpu_free']
        low = host < 2*2**30 or gpu < 512*2**20
        high = host > 4*2**30 and gpu > 1536*2**20
        if low:
            self.paused = True
            self.healthy_since = None
            self.workers, self.prefetch, self.last_change = 0, 1, now
            return 'pause'
        if high:
            self.healthy_since = now if self.healthy_since is None else self.healthy_since
            if now-self.healthy_since >= 120 and now-self.last_change >= 300:
                if self.workers < 2:
                    self.workers += 1
                elif self.prefetch < 2:
                    self.prefetch += 1
                self.last_change = now
            if now-self.healthy_since >= 120:
                self.paused = False
            return 'pause' if self.paused else 'continue'
        self.healthy_since = None
        return 'pause' if self.paused else 'continue'
