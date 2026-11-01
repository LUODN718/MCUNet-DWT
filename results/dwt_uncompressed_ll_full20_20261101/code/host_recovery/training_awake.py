"""Keep this controller's session awake without changing the Windows power plan."""
import ctypes as c
from ctypes import wintypes as w
import datetime as dt
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
K = c.WinDLL('kernel32', use_last_error=True)
K.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
K.OpenProcess.restype = w.HANDLE
K.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
K.WaitForSingleObject.restype = w.DWORD
K.CloseHandle.argtypes = [w.HANDLE]
K.SetThreadExecutionState.argtypes = [w.DWORD]
K.SetThreadExecutionState.restype = w.DWORD


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def publish(data):
    path = ROOT / 'training_session_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def main():
    controller = read(BASE / 'run_logs/controller.json')
    handle = K.OpenProcess(0x100000, False, controller['controller_pid'])
    if not handle:
        raise c.WinError(c.get_last_error())
    if not K.SetThreadExecutionState(0x80000001):
        K.CloseHandle(handle)
        raise c.WinError(c.get_last_error())
    reason = 'controller_exited'
    last = dict(helper_pid=os.getpid(), controller_pid=controller['controller_pid'])
    try:
        while K.WaitForSingleObject(handle, 0) == 0x102:
            if (BASE / 'run/complete.json').exists():
                reason = 'completed'; break
            try:
                progress = read(BASE / 'run/progress.json')
                pointer = read(BASE / 'run/checkpoints/checkpoints.json')
                resource = read(BASE / 'run/resources/latest.json')
                last.update(epoch=progress.get('epoch'),
                    latest_checkpoint_step=pointer.get('step'), progress_step=progress.get('step'),
                    available_gib=round(resource['host_available'] / 2**30, 3),
                    resource_sample_age_seconds=round(time.time()-resource['time'], 1),
                    resource_action=resource['action'])
                stage = 'resource_paused' if resource['action'] == 'pause' else 'running'
                if time.time() - resource['time'] >= 90:
                    stage = 'resource_telemetry_stale'
                publish(dict(last, status=stage, idle_sleep_prevented=True,
                    time=dt.datetime.now().astimezone().isoformat()))
            except (OSError, ValueError, KeyError) as exc:
                publish(dict(last, status='status_read_retry', error=str(exc),
                    idle_sleep_prevented=True, time=dt.datetime.now().astimezone().isoformat()))
            time.sleep(15)
    finally:
        K.SetThreadExecutionState(0x80000000)
        K.CloseHandle(handle)
        publish(dict(last, status=reason, idle_sleep_prevented=False,
            time=dt.datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
