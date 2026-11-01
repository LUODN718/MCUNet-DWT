"""Run one admitted RGB160 baseline; resume only transient OOM/file-lock failures.

Never kills another process, changes batch/precision, or substitutes datasets.
Unrecognized failures stop for inspection. No uninterrupted-runtime guarantee.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

MANIFEST = "c3e2d54e9bcd5d5d1e03b96be5f45ad04fa384fad2d1cdf4878296e215f80179"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def publish(path, value):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    for n in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if n == 7:
                raise
            time.sleep(min(.1*2**n, 2))


@contextmanager
def controller_lock(path):
    with path.open("a+b") as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b"0"); f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def resources(run_parent):
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    memory = MemoryStatus(); memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError("cannot query host available memory")
    free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                   text=True, timeout=15)
    return dict(host_available=memory.avail_phys, gpu_free=int(free.strip().splitlines()[0])*2**20,
                disk_free=shutil.disk_usage(run_parent).free)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff-root", "train-tar", "seal-dir", "weights", "mcunet-root", "run-dir", "admission-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-restarts", type=int, default=3)
    p.add_argument("--accept-local-content-sha256")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    trainer = repo / "scripts/train_w80_rgb160_baseline.py"
    adapter = repo / "scripts/w80_tar_source.py"
    admission = read(args.admission_dir / "cuda_admission.json")
    accepted_manifest = args.accept_local_content_sha256 or MANIFEST
    if (admission.get("status") != "passed" or admission.get("manifest_sha256") != accepted_manifest
            or admission.get("adapter_sha256") != sha(adapter)
            or admission.get("trainer_sha256") != sha(trainer)):
        raise ValueError("missing/mismatched CUDA and original-content admission")
    run = args.run_dir.resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    logs = run.with_name(run.name + "_logs")
    logs.mkdir(exist_ok=True)
    state_file = logs / "controller.json"
    command = [sys.executable, "-u", str(trainer)]
    for name in ("handoff_root", "train_tar", "seal_dir", "weights", "mcunet_root", "run_dir"):
        command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
    command += ["--workers", "0", "--prefetch", "1"]
    if args.accept_local_content_sha256:
        command += ["--accept-local-content-sha256", args.accept_local_content_sha256]
    base = dict(controller_pid=os.getpid(), run_dir=str(run), command=command,
                scope="one uncompressed RGB160 baseline", manifest_sha256=accepted_manifest,
                dataset_alignment=admission.get("dataset_alignment"),
                controller_sha256=sha(__file__), auto_resume="transient CUDA OOM / PermissionError only",
                maximum_restarts=args.max_restarts)
    with controller_lock(logs / "controller.lock"):
        previous = read(state_file) if state_file.exists() else {}
        attempt = int(previous.get("attempt", -1)) + 1
        restarts = 0
        recovery = False
        failed_gpu_free = None
        while True:
            if (run / "complete.json").exists():
                complete = read(run / "complete.json")
                publish(state_file, dict(base, status="completed", attempt=attempt, complete=complete, updated_at=time.time()))
                return 0
            # Resource waits never change the training recipe or interrupt a worker.
            healthy_since = None
            while True:
                sample = resources(run.parent)
                required_host = (4 if recovery else 2) * 2**30
                high = (sample["host_available"] >= required_host and sample["gpu_free"] >= 1536*2**20
                        and sample["disk_free"] >= 10*2**30
                        and (failed_gpu_free is None or sample["gpu_free"] >= failed_gpu_free + 256*2**20))
                if high:
                    healthy_since = healthy_since or time.monotonic()
                    if not recovery or time.monotonic()-healthy_since >= 120:
                        break
                else:
                    healthy_since = None
                publish(state_file, dict(base, status="waiting_for_resources", attempt=attempt,
                                        resources=sample, updated_at=time.time()))
                time.sleep(30)
            if sha(trainer) != admission["trainer_sha256"] or sha(adapter) != admission["adapter_sha256"]:
                raise ValueError("admitted source changed; do not resume with changed code")
            resume = (run / "checkpoints/checkpoints.json").exists()
            current = command + (["--resume"] if resume else [])
            stdout = logs / f"attempt_{attempt:03d}.stdout.log"
            stderr = logs / f"attempt_{attempt:03d}.stderr.log"
            if stdout.exists() or stderr.exists():
                raise FileExistsError("attempt logs already exist")
            with stdout.open("xb") as out, stderr.open("xb") as err:
                process = subprocess.Popen(current, cwd=repo, stdout=out, stderr=err,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                while process.poll() is None:
                    progress_path = run / "progress.json"
                    progress = read(progress_path) if progress_path.exists() else None
                    publish(state_file, dict(base, status="running", attempt=attempt, worker_pid=process.pid,
                                            resume=resume, committed_progress=progress,
                                            stdout=str(stdout), stderr=str(stderr), updated_at=time.time()))
                    time.sleep(30)
                code = process.returncode
            if code == 0 and (run / "complete.json").exists():
                continue
            tail = stderr.read_text(encoding="utf-8", errors="replace")[-12000:]
            transient = "PermissionError" in tail or "CUDA out of memory" in tail
            details = dict(base, attempt=attempt, worker_pid=process.pid, exit_code=code,
                           stdout=str(stdout), stderr=str(stderr), error_tail=tail, updated_at=time.time())
            if not transient or restarts >= args.max_restarts:
                publish(state_file, dict(details, status="needs_attention"))
                return code or 1
            if not (run / "checkpoints/checkpoints.json").exists():
                publish(state_file, dict(details, status="needs_attention_no_checkpoint"))
                return code or 1
            publish(state_file, dict(details, status="transient_failure_pending_resume"))
            restarts += 1
            attempt += 1
            recovery = True
            # The worker has exited. Require 256 MiB more than the observed
            # failed-attempt free VRAM before replaying an OOM at the same batch.
            failed_gpu_free = None
            if "CUDA out of memory" in tail:
                event_path = run / "events.jsonl"
                for line in reversed(event_path.read_text(encoding="utf-8").splitlines()):
                    event = json.loads(line)
                    if event.get("event") == "batch_failure" and "out of memory" in event.get("error", "").lower():
                        failed_gpu_free = event["gpu_free_before_attempt"]
                        break
                if failed_gpu_free is None:
                    publish(state_file, dict(details, status="needs_attention_no_oom_watermark"))
                    return code or 1


if __name__ == "__main__":
    raise SystemExit(main())
