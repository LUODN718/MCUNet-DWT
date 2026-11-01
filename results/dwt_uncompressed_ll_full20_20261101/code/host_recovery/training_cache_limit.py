"""Temporary Windows file-cache limit for this one existing training controller.

Uses documented Get/SetSystemFileCacheSize; never changes training state or RAM
guard thresholds. Run elevated. Original settings are saved before mutation and
restored when the controller exits, the run completes, or STOP_CACHE_HELPER exists.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import datetime as dt
import json
import msvcrt
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'run'
LOGS = ROOT.parent / 'run_logs'
GiB = 2**30
K = c.WinDLL('kernel32', use_last_error=True)
A = c.WinDLL('advapi32', use_last_error=True)
P = c.WinDLL('psapi', use_last_error=True)
K.GetSystemFileCacheSize.argtypes = [c.POINTER(c.c_size_t), c.POINTER(c.c_size_t), c.POINTER(w.DWORD)]
K.GetSystemFileCacheSize.restype = w.BOOL
K.SetSystemFileCacheSize.argtypes = [c.c_size_t, c.c_size_t, w.DWORD]
K.SetSystemFileCacheSize.restype = w.BOOL
K.GetCurrentProcess.restype = w.HANDLE
K.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
K.OpenProcess.restype = w.HANDLE
K.CloseHandle.argtypes = [w.HANDLE]
K.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
K.WaitForSingleObject.restype = w.DWORD


class Memory(c.Structure):
    _fields_ = [('length', w.DWORD), ('load', w.DWORD)] + [(n, c.c_ulonglong) for n in
        ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'unused')]


class Performance(c.Structure):
    _fields_ = [('size', w.DWORD)] + [(n, c.c_size_t) for n in
        ('commit_total', 'commit_limit', 'commit_peak', 'physical_total', 'physical_available',
         'system_cache', 'kernel_total', 'kernel_paged', 'kernel_nonpaged', 'page_size')] + [
        ('handles', w.DWORD), ('processes', w.DWORD), ('threads', w.DWORD)]


class Luid(c.Structure):
    _fields_ = [('low', w.DWORD), ('high', w.LONG)]


class Privileges(c.Structure):
    _fields_ = [('count', w.DWORD), ('luid', Luid), ('attributes', w.DWORD)]


def check(ok):
    if not ok:
        raise c.WinError(c.get_last_error())


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def publish(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def cache_settings():
    minimum, maximum, flags = c.c_size_t(), c.c_size_t(), w.DWORD()
    check(K.GetSystemFileCacheSize(c.byref(minimum), c.byref(maximum), c.byref(flags)))
    return dict(minimum=minimum.value, maximum=maximum.value, flags=flags.value)


def snapshot():
    memory = Memory(length=c.sizeof(Memory))
    check(K.GlobalMemoryStatusEx(c.byref(memory)))
    perf = Performance(size=c.sizeof(Performance))
    check(P.GetPerformanceInfo(c.byref(perf), c.sizeof(perf)))
    return dict(time=dt.datetime.now().astimezone().isoformat(),
                available_gib=round(memory.avail_phys / GiB, 3),
                system_cache_gib=round(perf.system_cache * perf.page_size / GiB, 3),
                nonpaged_pool_gib=round(perf.kernel_nonpaged * perf.page_size / GiB, 3),
                cache_settings=cache_settings())


def enable_quota_privilege():
    A.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
    A.LookupPrivilegeValueW.argtypes = [w.LPCWSTR, w.LPCWSTR, c.POINTER(Luid)]
    A.AdjustTokenPrivileges.argtypes = [w.HANDLE, w.BOOL, c.POINTER(Privileges), w.DWORD, c.c_void_p, c.c_void_p]
    token = w.HANDLE()
    check(A.OpenProcessToken(K.GetCurrentProcess(), 0x20 | 0x8, c.byref(token)))
    try:
        value = Luid()
        check(A.LookupPrivilegeValueW(None, 'SeIncreaseQuotaPrivilege', c.byref(value)))
        privilege = Privileges(count=1, luid=value, attributes=2)
        c.set_last_error(0)
        check(A.AdjustTokenPrivileges(token, False, c.byref(privilege), 0, None, None))
        if c.get_last_error():
            raise c.WinError(c.get_last_error())
    finally:
        K.CloseHandle(token)


def restore(original):
    # Explicitly restore both hard-limit flags, including their disabled state.
    flags = (0x1 if original['flags'] & 0x1 else 0x2) | (0x4 if original['flags'] & 0x4 else 0x8)
    check(K.SetSystemFileCacheSize(original['minimum'], original['maximum'], flags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(snapshot(), indent=2), flush=True)
        return
    with (ROOT / 'cache_helper.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if not lock.tell():
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        enable_quota_privilege()
        original_path = ROOT / 'cache_original_settings.json'
        if args.restore:
            original = read(original_path)['settings']
            restore(original)
            publish(ROOT / 'cache_helper_status.json', dict(status='manually_restored', **snapshot()))
            return
        if (RUN / 'complete.json').exists():
            raise RuntimeError('Training is already complete; no cache limit needed')
        controller = read(LOGS / 'controller.json')
        # Retain a handle to this exact process, so PID reuse cannot extend the limit.
        handle = K.OpenProcess(0x100000, False, controller['controller_pid'])
        check(handle)
        original = cache_settings()
        if original_path.exists():
            original = read(original_path)['settings']
        else:
            publish(original_path, dict(settings=original, before=snapshot()))
        applied = False
        outcome = 'controller_exited'
        try:
            # Hard maximum of 2 GiB, no hard minimum. This affects only file cache.
            check(K.SetSystemFileCacheSize(64 * 2**20, 2 * GiB, 0x1 | 0x8))
            applied = True
            while True:
                if (RUN / 'complete.json').exists():
                    outcome = 'training_completed'; break
                if (ROOT / 'STOP_CACHE_HELPER').exists():
                    outcome = 'stop_requested'; break
                if K.WaitForSingleObject(handle, 0) != 0x102:
                    break
                data = dict(status='active', helper_pid=os.getpid(),
                            controller_pid=controller['controller_pid'], **snapshot())
                publish(ROOT / 'cache_helper_status.json', data)
                with (ROOT / 'cache_helper_samples.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(data) + '\n')
                time.sleep(15)
        finally:
            K.CloseHandle(handle)
            if applied:
                restore(original)
                publish(ROOT / 'cache_helper_status.json', dict(status='restored', reason=outcome, **snapshot()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        publish(ROOT / 'cache_helper_error.json', dict(time=dt.datetime.now().astimezone().isoformat(),
                error_type=type(exc).__name__, error=str(exc)))
        raise
