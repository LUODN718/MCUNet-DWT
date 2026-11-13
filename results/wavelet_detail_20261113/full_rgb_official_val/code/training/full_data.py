"""Full original ImageNet TRAIN, compact offset index, uncached Windows reads."""
import ctypes as c
from ctypes import wintypes as w
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

ROOT = Path(__file__).resolve().parents[1]
N = 1_281_167
BATCH = 32
STEPS = math.ceil(N / BATCH)
PADDED = STEPS * BATCH
DTYPE = np.dtype([('offset', '<u8'), ('size', '<u4'), ('label', '<u2'), ('path', 'S64')])
HASH_DTYPE = np.dtype([('raw', 'u1', (32,)), ('rgb', 'u1', (32,)), ('valid', 'u1')])

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def publish(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    for attempt in range(8):
        try: os.replace(temp, path); return
        except PermissionError:
            if attempt == 7: raise
            time.sleep(min(.1 * 2**attempt, 2))

def identity(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)

def prepare(archive, classes_path):
    root = ROOT / 'data'; root.mkdir(exist_ok=True)
    receipt_path, index_path = root / 'index_receipt.json', root / 'full_index.npy'
    classes = json.loads(Path(classes_path).read_text(encoding='utf-8'))
    if len(classes) != 1000 or classes != sorted(set(classes)): raise ValueError('invalid class mapping')
    expected = identity(archive)
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        if r['archive'] != expected or r['class_mapping_sha256'] != sha(classes_path) or r['index_sha256'] != sha(index_path):
            raise ValueError('archive, class mapping or index changed')
        return r
    pending = root / 'full_index.pending.npy'
    array = np.lib.format.open_memmap(pending, mode='w+', dtype=DTYPE, shape=(N,))
    count = 0; counts = {}; started = time.time()
    with tarfile.open(archive, 'r:') as outer:
        containers = {}
        for m in outer.getmembers():
            if m.isfile() and m.name.endswith('.tar'):
                cls = PurePosixPath(m.name).stem
                if cls in containers: raise ValueError('duplicate outer class')
                containers[cls] = m
        if sorted(containers) != classes: raise ValueError('expected exactly the fixed 1000 classes')
        for label, cls in enumerate(classes):
            container = containers[cls]; entries = []
            with outer.extractfile(container) as stream:
                with tarfile.open(fileobj=stream, mode='r:') as inner:
                    for m in inner:
                        if not m.isfile(): continue
                        name = PurePosixPath(m.name).name
                        if m.name.removeprefix('./') != name or not name.lower().endswith(('.jpeg', '.jpg')):
                            raise ValueError('unexpected image member: ' + m.name)
                        path = cls + '/' + name
                        if len(path.encode('ascii')) > 64 or m.size <= 0 or m.offset_data + m.size > container.size:
                            raise ValueError('invalid nested image bounds')
                        entries.append((container.offset_data + m.offset_data, m.size, label, path.encode('ascii')))
            entries.sort(key=lambda r: r[3])
            if len({r[3] for r in entries}) != len(entries): raise ValueError('duplicate image path')
            end = count + len(entries)
            if end > N: raise ValueError('too many ImageNet TRAIN images')
            array[count:end] = np.array(entries, dtype=DTYPE)
            count = end; counts[cls] = len(entries)
            if (label + 1) % 50 == 0:
                publish(root / 'index_progress.json', dict(status='indexing', classes=label+1, images=count, elapsed_seconds=time.time()-started))
                print(f'index {label+1}/1000 classes, {count} images', flush=True)
    if count != N or identity(archive) != expected: raise ValueError('TRAIN cardinality or archive identity mismatch')
    array.flush(); del array; os.replace(pending, index_path)
    (root / 'classes.json').write_bytes(Path(classes_path).read_bytes())
    receipt = dict(schema=1, archive=expected, images=count, classes=1000, class_counts=counts,
        class_mapping_sha256=sha(classes_path), index_sha256=sha(index_path),
        index_order='sorted class then sorted original JPEG filename',
        integrity='tar header/bounds/count checked; JPEG and decoded-RGB SHA256 established on first use and verified thereafter',
        source='all original ImageNet TRAIN; no exclusions or held-out development images',
        elapsed_seconds=time.time()-started, created_at=time.time())
    publish(receipt_path, receipt); return receipt

class UncachedReader:
    """VirtualAlloc-aligned FILE_FLAG_NO_BUFFERING reads, no OS data cache."""
    def __init__(self, path):
        if os.name != 'nt': raise RuntimeError('this experiment explicitly requires Windows direct I/O')
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.k.CreateFileW.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        self.k.CreateFileW.restype = w.HANDLE
        self.k.VirtualAlloc.argtypes = [c.c_void_p,c.c_size_t,w.DWORD,w.DWORD]
        self.k.VirtualAlloc.restype = c.c_void_p
        self.k.VirtualFree.argtypes = [c.c_void_p,c.c_size_t,w.DWORD]
        self.k.ReadFile.argtypes = [w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
        self.k.SetFilePointerEx.argtypes = [w.HANDLE,c.c_longlong,c.c_void_p,w.DWORD]
        self.k.CloseHandle.argtypes = [w.HANDLE]
        # GetFileInformationByHandleEx(FileStorageInfo): query physical/logical sectors.
        self.k.GetFileInformationByHandleEx.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        self.handle = self.k.CreateFileW(str(Path(path).resolve()),0x80000000,1,None,3,0x20000000|0x10000000,None)
        if self.handle == c.c_void_p(-1).value: raise c.WinError(c.get_last_error())
        info = (w.DWORD * 7)()
        if not self.k.GetFileInformationByHandleEx(self.handle,16,c.byref(info),c.sizeof(info)):
            self.close(); raise c.WinError(c.get_last_error())
        self.alignment = max(4096, int(info[0]), int(info[1]), int(info[2]))
        if self.alignment > 65536 or self.alignment & (self.alignment-1):
            self.close(); raise ValueError('unsupported disk alignment')
        self.pointer = None; self.capacity = 0
    def read(self, offset, size):
        start = offset // self.alignment * self.alignment
        delta = offset - start
        amount = (delta + size + self.alignment - 1) // self.alignment * self.alignment
        if amount > self.capacity:
            if self.pointer: self.k.VirtualFree(self.pointer,0,0x8000)
            self.capacity = max(amount,2**20)
            self.pointer = self.k.VirtualAlloc(None,self.capacity,0x1000|0x2000,4)
            if not self.pointer: raise c.WinError(c.get_last_error())
        if not self.k.SetFilePointerEx(self.handle,start,None,0): raise c.WinError(c.get_last_error())
        actual = w.DWORD()
        if not self.k.ReadFile(self.handle,self.pointer,amount,c.byref(actual),None): raise c.WinError(c.get_last_error())
        if actual.value < delta + size: raise OSError('truncated direct JPEG read')
        return c.string_at(self.pointer+delta,size)
    def close(self):
        if getattr(self,'pointer',None): self.k.VirtualFree(self.pointer,0,0x8000); self.pointer=None
        handle = getattr(self,'handle',None)
        if handle is not None and handle != -1 and handle != 18446744073709551615:
            self.k.CloseHandle(handle); self.handle=None
    def __del__(self): self.close()

class FullImages(torch.utils.data.Dataset):
    def __init__(self, archive, epoch=0):
        self.archive = str(archive); self.epoch = epoch
        self.index = np.load(ROOT / 'data/full_index.npy', mmap_mode='r')
        hash_path = ROOT / 'data/content_hashes.npy'
        if hash_path.exists(): self.hashes = np.lib.format.open_memmap(hash_path, mode='r+')
        else: self.hashes = np.lib.format.open_memmap(hash_path, mode='w+', dtype=HASH_DTYPE, shape=(N,))
        if self.hashes.shape != (N,) or self.hashes.dtype != HASH_DTYPE: raise ValueError('invalid content hash store')
        self.reader = UncachedReader(archive)
    def __len__(self): return N
    def __getitem__(self, i):
        r = self.index[i]; path = bytes(r['path']).decode('ascii')
        raw = self.reader.read(int(r['offset']),int(r['size']))
        raw_hash = hashlib.sha256(raw).digest()
        with Image.open(io.BytesIO(raw)) as source: im = source.convert('RGB')
        rgb_hash = hashlib.sha256(str(im.size).encode()+im.tobytes()).digest()
        h = self.hashes[i]
        if h['valid']:
            if h['raw'].tobytes() != raw_hash or h['rgb'].tobytes() != rgb_hash: raise ValueError('TRAIN content changed: '+path)
        else:
            self.hashes[i]['raw'] = np.frombuffer(raw_hash,dtype='u1')
            self.hashes[i]['rgb'] = np.frombuffer(rgb_hash,dtype='u1')
            self.hashes[i]['valid'] = 1
        key = json.dumps([2026,self.epoch,path],sort_keys=True,separators=(',',':'),allow_nan=False)
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:16],16) % 2**63
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            top,left,height,width = T.RandomResizedCrop.get_params(im,(.08,1.),(.75,4/3))
            im = F.resized_crop(im,top,left,height,width,(160,160),T.InterpolationMode.BICUBIC,antialias=True)
            if torch.rand(()) < .5: im = F.hflip(im)
        return F.to_tensor(im), int(r['label'])
    def flush(self): self.hashes.flush()
    def close(self): self.flush(); self.reader.close()

def epoch_order(epoch):
    order = torch.randperm(N,generator=torch.Generator().manual_seed(2026+1000003*epoch)).numpy()
    return np.concatenate([order,order[:PADDED-N]])

def batches(dataset, order, cursor=0):
    return torch.utils.data.DataLoader(dataset,batch_size=32,sampler=order[cursor:],num_workers=0,
        pin_memory=False,drop_last=False,generator=torch.Generator().manual_seed(2026+dataset.epoch))
