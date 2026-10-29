"""Verify real recovery decisions without launching or killing a GPU job."""
import errno
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.w80_pilot.recovery import best_effort_publish, recoverable, run_child


class UnattendedTests(unittest.TestCase):
    def test_display_failure_is_nonfatal(self):
        publisher = mock.Mock(side_effect=PermissionError('locked display'))
        with mock.patch('logging.warning'):
            self.assertFalse(best_effort_publish(publisher, Path('unused.json'), {'updates': 100}))
        publisher.assert_called_once()

    def test_retry_only_operational_errors(self):
        self.assertTrue(recoverable(MemoryError()))
        self.assertTrue(recoverable(PermissionError()))
        self.assertTrue(recoverable(OSError(errno.ENOMEM, 'allocation failed')))
        self.assertFalse(recoverable(ValueError('checkpoint checksum mismatch')))
        self.assertFalse(recoverable(FloatingPointError('nonfinite gradient')))
        self.assertFalse(recoverable(OSError(errno.ENOSPC, 'disk full')))

    def run_case(self, codes, complete_last=False, stop_before=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        parent = Path(temporary.name)
        folder = parent / 'job'
        band = folder / 'll'
        checkpoints = band / 'checkpoints'
        checkpoints.mkdir(parents=True)
        (checkpoints/'checkpoints.json').write_text(json.dumps({'latest': 'fixed-checkpoint.pt'}))
        stop = parent/'STOP'
        if stop_before:
            stop.touch()
        spawned, statuses, sleeps = [], [], []
        class Child:
            def __init__(self, code):
                self.pid = 1000 + len(spawned)
                self.code = code
            def wait(self):
                if self.code == 0 and complete_last:
                    (band/'summary.json').write_text('{}')
                return self.code
        def popen(*args, **kwargs):
            child = Child(codes[len(spawned)])
            spawned.append(child)
            return child
        result = run_child(['python', 'job.py'], folder, 'll',
                           lambda stage, **extra: statuses.append((stage, extra)), stop,
                           popen, sleep=sleeps.append, max_restarts=8, max_no_progress=3)
        return result, spawned, statuses, sleeps, folder

    def test_recoverable_exit_restarts_and_completes(self):
        result, spawned, statuses, sleeps, folder = self.run_case([75, 0], complete_last=True)
        self.assertEqual(result, ('complete', 0))
        self.assertEqual(len(spawned), 2)
        self.assertEqual(sleeps, [10])
        self.assertEqual(len((folder/'automatic_restarts.jsonl').read_text().splitlines()), 1)
        self.assertIn('recovering', [s[0] for s in statuses])

    def test_invalid_model_or_data_is_not_retried(self):
        result, spawned, statuses, sleeps, folder = self.run_case([1])
        self.assertEqual(result, ('fatal', 1))
        self.assertEqual(len(spawned), 1)
        self.assertFalse(sleeps)

    def test_repeated_same_checkpoint_failure_is_bounded(self):
        result, spawned, statuses, sleeps, folder = self.run_case([75]*4)
        self.assertEqual(result, ('exhausted', 75))
        self.assertEqual(len(spawned), 4)
        self.assertEqual(sleeps, [10, 20, 30])

    def test_explicit_stop_prevents_restart(self):
        result, spawned, statuses, sleeps, folder = self.run_case([75], stop_before=True)
        self.assertEqual(result, ('stopped', None))
        self.assertFalse(spawned)

    def test_real_child_exit75_then_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            folder = parent / 'job'
            (folder / 'll').mkdir(parents=True)
            child = folder / 'child.py'
            child.write_text(
                "import pathlib, sys\n"
                "root = pathlib.Path(__file__).parent\n"
                "marker = root / 'attempted'\n"
                "if not marker.exists():\n"
                "    marker.write_text('1')\n"
                "    sys.exit(75)\n"
                "(root / 'll/summary.json').write_text('{}')\n",
                encoding='utf-8')
            outcome = run_child([sys.executable, str(child)], folder, 'll',
                                lambda *a, **k: None, parent / 'STOP', subprocess.Popen,
                                sleep=lambda seconds: None)
            self.assertEqual(outcome, ('complete', 0))
            records = (folder / 'automatic_restarts.jsonl').read_text().splitlines()
            self.assertEqual(len(records), 1)

    def test_new_checkpoint_name_without_update_is_not_progress(self):
        from experiments.w80_pilot.atomic_io import publish_snapshot
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            folder = parent / 'job'
            checkpoint = folder / 'll/checkpoints'
            checkpoint.mkdir(parents=True)
            calls = []
            class Child:
                pid = 1234
                def wait(self):
                    calls.append(1)
                    (checkpoint / 'checkpoints.json').write_text(json.dumps({'latest': str(len(calls))}))
                    publish_snapshot(folder / 'status.json', {'successful_updates': 100})
                    return 75
            outcome = run_child(['unused'], folder, 'll', lambda *a, **k: None,
                                parent / 'STOP', lambda *a, **k: Child(), sleep=lambda seconds: None)
            self.assertEqual(outcome, ('exhausted', 75))
            self.assertEqual(len(calls), 4)


if __name__ == '__main__':
    unittest.main()
