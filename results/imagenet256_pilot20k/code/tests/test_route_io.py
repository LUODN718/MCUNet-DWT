"""Regression checks for the observed Windows resource-display failure."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.w80_pilot.atomic_io import publish_snapshot, read_latest_snapshot, write_json


class ResourceIOTests(unittest.TestCase):
    def test_transient_replace_conflict_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'resource.json'
            replace = Path.replace
            attempts = []
            def conflict(path, destination):
                attempts.append(path)
                if len(attempts) < 3:
                    raise PermissionError('injected WinError5 sharing conflict')
                return replace(path, destination)
            with mock.patch.object(Path, 'replace', conflict):
                write_json(target, {'gpu_free': 123}, timeout=1)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text()), {'gpu_free': 123})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_failed_replace_keeps_previous_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'resource.json'
            write_json(target, {'old': True})
            with mock.patch.object(Path, 'replace', side_effect=PermissionError('persistent conflict')):
                with self.assertRaises(PermissionError):
                    write_json(target, {'old': False}, timeout=.05)
            self.assertEqual(json.loads(target.read_text()), {'old': True})
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_display_write_error_does_not_kill_guard(self):
        spec = importlib.util.spec_from_file_location('route_trainer_io_test', ROOT / 'scripts/train_route_pilot.py')
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        count = []
        published = threading.Event()
        def fail_once(path, value):
            count.append(value)
            if len(count) == 1:
                raise PermissionError('injected resource display lock')
            publish_snapshot(path, value)
            published.set()
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(trainer.torch.cuda, 'mem_get_info', return_value=(4*2**30, 6*2**30)), \
                 mock.patch.object(trainer, 'available_host_memory', return_value=3*2**30), \
                 mock.patch.object(trainer, 'publish_snapshot', side_effect=fail_once), \
                 mock.patch.object(trainer.logging, 'warning') as warning:
                guard = trainer.Guard(Path(folder), .75, 1)
                try:
                    self.assertTrue(published.wait(timeout=10))
                    self.assertGreaterEqual(len(count), 2)
                    self.assertTrue(guard.thread.is_alive())
                    self.assertEqual(guard.decision(), 'continue')
                    self.assertEqual(read_latest_snapshot(Path(folder)/'resource.json')['host_available'], 3*2**30)
                    warning.assert_called_once()
                finally:
                    guard.close()

    def test_immutable_snapshots_allow_concurrent_readers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'resource.json'
            failures = []
            done = threading.Event()
            def reader():
                while not done.is_set():
                    value = read_latest_snapshot(path)
                    if value and not isinstance(value.get('sequence'), int):
                        failures.append(value)
            readers = [threading.Thread(target=reader) for _ in range(4)]
            for thread in readers:
                thread.start()
            try:
                for sequence in range(200):
                    publish_snapshot(path, {'sequence': sequence})
                final = read_latest_snapshot(path)
                self.assertEqual(final['sequence'], 199)
                self.assertGreaterEqual(len(list((Path(folder)/'resource_snapshots').glob('*.json'))), 200)
                self.assertFalse(list((Path(folder)/'resource_snapshots').glob('*.tmp')))
                self.assertEqual(failures, [])
            finally:
                done.set()
                for thread in readers:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive())

    @unittest.skipUnless(os.name == 'nt', 'Windows file-sharing regression')
    def test_real_windows_reader_lock_cannot_block_new_snapshots(self):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            write_json(path, {'old_fixed_file': True})
            # Allow read/write, deliberately omit FILE_SHARE_DELETE, reproducing
            # the Windows condition that forbids replacing a reader-held file.
            handle = kernel.CreateFileW(str(path), 0x80000000, 1 | 2, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                with self.assertRaises(PermissionError):
                    write_json(path, {'old_fixed_file': False}, timeout=.05)
                held_snapshot = publish_snapshot(path, {'updates': 8863})
                with held_snapshot.open('r') as reader:
                    for updates in range(8864, 8914):
                        publish_snapshot(path, {'updates': updates})
                    self.assertEqual(json.loads(reader.read()), {'updates': 8863})
                self.assertEqual(read_latest_snapshot(path), {'updates': 8913})
                self.assertEqual(json.loads(path.read_text()), {'old_fixed_file': True})
            finally:
                kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
