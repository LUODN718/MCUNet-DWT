"""Read the frozen W80 cohort directly from an uncompressed nested ImageNet tar.

Only an offset index, manifests and receipts are written; JPEGs stay in the tar.
The content manifest must reproduce the teammate's exact SHA256 before training.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor


INDEX_FIELDS = ["path", "offset", "size"]
MANIFEST_FIELDS = ["role", "path", "label", "sha", "rgb"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(8):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.1 * 2**attempt, 2))


def load_index(path, rows=None):
    entries = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != INDEX_FIELDS:
            raise ValueError("unexpected tar index schema")
        for row in reader:
            name = row["path"]
            offset, size = int(row["offset"]), int(row["size"])
            if name in entries or offset < 0 or size <= 0:
                raise ValueError("duplicate or invalid tar index entry: " + name)
            entries[name] = (offset, size)
    if rows is not None and list(entries) != [row["path"] for row in rows]:
        raise ValueError("tar index membership/order differs from frozen cohort")
    return entries


def build_index(archive, rows, index_path, receipt_path):
    archive, index_path, receipt_path = map(Path, (archive, index_path, receipt_path))
    identity = archive_identity(archive)
    if index_path.exists() or receipt_path.exists():
        if not index_path.exists() or not receipt_path.exists():
            raise ValueError("incomplete index publication; inspect the seal directory")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["archive"] != identity or receipt["index_sha256"] != sha256(index_path):
            raise ValueError("tar archive or index changed after indexing")
        return load_index(index_path, rows), receipt

    wanted = {}
    for row in rows:
        parts = PurePosixPath(row["path"]).parts
        if len(parts) != 2 or ".." in parts or "\\" in row["path"]:
            raise ValueError("invalid frozen image path")
        wanted.setdefault(parts[0], set()).add(parts[1])
    entries = {}
    # r: deliberately rejects tar.gz/zip, whose offsets cannot be read directly.
    with tarfile.open(archive, "r:") as outer:
        classes = {}
        for member in outer.getmembers():
            if member.isfile() and member.name.endswith(".tar"):
                name = PurePosixPath(member.name).stem
                if name in classes:
                    raise ValueError("duplicate outer class archive: " + name)
                if member.offset_data + member.size > identity["bytes"]:
                    raise ValueError("truncated outer class archive: " + name)
                classes[name] = member
        missing = sorted(set(wanted) - set(classes))
        if missing:
            raise FileNotFoundError("missing class archives: " + ", ".join(missing))
        for position, class_id in enumerate(sorted(wanted), 1):
            container = classes[class_id]
            found = set()
            with outer.extractfile(container) as source:
                with tarfile.open(fileobj=source, mode="r:") as inner:
                    for member in inner:
                        if not member.isfile():
                            continue
                        name = PurePosixPath(member.name).name
                        if name not in wanted[class_id]:
                            continue
                        if member.name.removeprefix("./") != name:
                            raise ValueError("nested image has unexpected path")
                        path = class_id + "/" + name
                        if path in entries:
                            raise ValueError("duplicate selected image: " + path)
                        if member.offset_data + member.size > container.size:
                            raise ValueError("truncated selected image: " + path)
                        entries[path] = (container.offset_data + member.offset_data, member.size)
                        found.add(name)
            absent = wanted[class_id] - found
            if absent:
                raise FileNotFoundError(f"{class_id}: {len(absent)} frozen images missing")
            if position % 50 == 0 or position == len(wanted):
                print(f"tar index: {position}/{len(wanted)} classes, {len(entries)} selected images", flush=True)
    if archive_identity(archive) != identity:
        raise ValueError("archive changed while indexing")
    if len(entries) != len(rows):
        raise ValueError("selected index image count mismatch")
    temp = index_path.with_suffix(".pending.tsv")
    with temp.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, INDEX_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            offset, size = entries[row["path"]]
            writer.writerow(dict(path=row["path"], offset=offset, size=size))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, index_path)
    receipt = {"schema": 1, "archive": identity, "index_sha256": sha256(index_path),
               "images": len(rows), "selected_jpeg_bytes": sum(n for _, n in entries.values()),
               "storage": "uncompressed nested tar; absolute JPEG byte offsets", "created_at": time.time()}
    write_json(receipt_path, receipt)
    return load_index(index_path, rows), receipt


_AUDIT_FILE = None


def _audit_worker_init(archive):
    global _AUDIT_FILE
    from PIL import Image, features
    if Image.__version__ != "11.2.1" or features.version("jpg") != "6.2":
        raise RuntimeError("frozen Pillow/JPEG decoder required")
    _AUDIT_FILE = Path(archive).open("rb")


def _audit_chunk(chunk):
    from PIL import Image
    result = []
    for row, offset, size in chunk:
        _AUDIT_FILE.seek(offset)
        raw = _AUDIT_FILE.read(size)
        if len(raw) != size:
            raise OSError("truncated JPEG: " + row["path"])
        with Image.open(io.BytesIO(raw)) as source:
            rgb = source.convert("RGB")
        result.append(dict(row, sha=hashlib.sha256(raw).hexdigest(),
                           rgb=hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()))
    return result


def validate_content_manifest(bundle, pending, seal_dir, membership, metadata):
    """A failed content seal is explicit and never admitted as a manifest."""
    try:
        return membership.validate(bundle, pending)
    except ValueError as error:
        failure = {k: v for k, v in metadata.items()
                   if k not in ("byte_identity", "decoded_rgb_identity")}
        failure.update(status="blocked_content_mismatch", error=str(error),
                       expected_manifest_sha256=membership.sealed()["manifest_sha256"],
                       actual_manifest_sha256=sha256(pending), updated_at=time.time(),
                       processed_images=metadata["images"],
                       formal_training_updates=0,
                       next_step="obtain the teammate's original full manifest and decoder provenance; compare row hashes")
        write_json(Path(seal_dir) / "content_mismatch.json", failure)
        write_json(Path(seal_dir) / "audit_progress.json", failure)
        raise


def validate_local_membership(bundle, manifest, membership):
    """Validate fixed IDs/order and local identities without claiming reference pixels."""
    result = membership.validate(bundle)
    _, members = membership.read_members(bundle)
    seen = {"path": set(), "sha": set(), "rgb": set()}
    with Path(manifest).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError("local content manifest schema mismatch")
        count = 0
        for count, row in enumerate(reader, 1):
            if count > len(members):
                raise ValueError("extra local manifest row")
            if any(row[k] != str(members[count-1][k]) for k in ("role", "path", "label")):
                raise ValueError("local manifest membership/order mismatch")
            for key in seen:
                value = row[key]
                if key != "path" and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError("invalid local identity hash")
                if value in seen[key]:
                    raise ValueError("duplicate local identity: " + key)
                seen[key].add(value)
        if count != len(members):
            raise ValueError("local manifest rows missing")
    return dict(result, content_identity="all local JPEG/RGB identities recorded; teammate content seal differs",
                matches_teammate_content_seal=False, duplicate_local_byte_and_rgb_identities=0,
                local_byte_and_rgb_historical_exclusions="not independently re-audited")


def accept_local_content(archive, bundle, seal_dir, membership, accepted_sha, expected):
    """Explicit user acceptance binds only the named, previously audited local hash."""
    if len(accepted_sha) != 64 or any(c not in "0123456789abcdef" for c in accepted_sha):
        raise ValueError("accept-local-content-sha256 must be an exact lowercase SHA256")
    pending = seal_dir / "manifest.unverified.tsv"
    failure = json.loads((seal_dir / "content_mismatch.json").read_text(encoding="utf-8"))
    if any(failure.get(k) != expected[k] for k in
           ("archive", "index_sha256", "Pillow", "jpeg", "libjpeg_turbo", "images")):
        raise ValueError("local content audit archive/index/decoder changed; acceptance cannot reuse it")
    if (failure.get("actual_manifest_sha256") != accepted_sha
            or sha256(pending) != accepted_sha or failure.get("processed_images") != expected["images"]):
        raise ValueError("acceptance does not match the complete local content audit")
    validation = validate_local_membership(bundle, pending, membership)
    local_manifest = seal_dir / "manifest.local_accepted.tsv"
    if local_manifest.exists():
        if sha256(local_manifest) != accepted_sha:
            raise ValueError("existing accepted local manifest differs")
    else:
        with local_manifest.open("xb") as stream:
            stream.write(pending.read_bytes())
            stream.flush(); os.fsync(stream.fileno())
    receipt = {**expected, "manifest_sha256": accepted_sha,
               "reference_manifest_sha256": membership.sealed()["manifest_sha256"],
               "matches_teammate_content_seal": False,
               "status": "user_accepted_local_content",
               "byte_identity": "enforced against local manifest at each sample read",
               "decoded_rgb_identity": "enforced against local manifest at each sample read",
               "user_authorization": "2026-10-30: 没事你开始训练吧，应该没多大区别",
               "limitation": "same frozen sample IDs; raw/RGB content not an exact teammate-matched control",
               "validation": validation}
    write_json(seal_dir / "local_content_acceptance.json", receipt)
    return receipt


def prepare_tar_seal(archive, bundle, seal_dir, membership, workers=2, accepted_content_sha256=None):
    from PIL import Image, features
    if Image.__version__ != "11.2.1" or features.version("jpg") != "6.2":
        raise RuntimeError("frozen Pillow/JPEG decoder required before tar audit")
    archive, bundle, seal_dir = map(Path, (archive, bundle, seal_dir))
    membership.validate(bundle)
    classes, rows = membership.read_members(bundle)
    seal_dir.mkdir(parents=True, exist_ok=True)
    index_path = seal_dir / "tar_index.tsv"
    entries, index_receipt = build_index(archive, rows, index_path, seal_dir / "tar_index_receipt.json")
    manifest = seal_dir / "manifest.tsv"
    audit_path = seal_dir / "content_audit.json"
    expected = {"schema": 1, "storage": "tar", "archive": archive_identity(archive),
                "index_sha256": sha256(index_path), "images": len(rows),
                "Pillow": Image.__version__, "jpeg": features.version("jpg"),
                "libjpeg_turbo": features.version_feature("libjpeg_turbo") if features.check_feature("libjpeg_turbo") else None,
                "byte_identity": "passed", "decoded_rgb_identity": "passed"}
    if accepted_content_sha256 is not None:
        return accept_local_content(archive, bundle, seal_dir, membership, accepted_content_sha256, expected), index_receipt
    if manifest.exists():
        membership.validate(bundle, manifest)
        if not audit_path.exists():
            raise ValueError("manifest exists without complete original-content audit receipt")
        receipt = json.loads(audit_path.read_text(encoding="utf-8"))
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError("tar/content receipt differs; inspect before training")
        return receipt, index_receipt

    pending = seal_dir / "manifest.unverified.tsv"
    previous_failure = seal_dir / "content_mismatch.json"
    if previous_failure.exists() and pending.exists():
        failure = json.loads(previous_failure.read_text(encoding="utf-8"))
        unchanged = all(failure.get(k) == expected[k] for k in
                        ("archive", "index_sha256", "Pillow", "jpeg", "libjpeg_turbo"))
        if unchanged and sha256(pending) == failure["actual_manifest_sha256"]:
            raise ValueError("known original-content mismatch; see content_mismatch.json; training remains blocked")
    # A new audit attempt keeps its previous unverified output for inspection.
    if pending.exists():
        pending.rename(seal_dir / ("manifest.unverified." + uuid.uuid4().hex + ".tsv"))
    chunks = []
    for start in range(0, len(rows), 128):
        chunks.append([(r, *entries[r["path"]]) for r in rows[start:start + 128]])
    started = time.time()
    count = 0
    with pending.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, MANIFEST_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers, initializer=_audit_worker_init,
                                 initargs=(str(archive),)) as pool:
            for audited in pool.map(_audit_chunk, chunks):
                writer.writerows(audited)
                count += len(audited)
                if count % 1024 == 0 or count == len(rows):
                    print(f"tar content audit: {count}/{len(rows)} original JPEG/RGB identities", flush=True)
                    write_json(seal_dir / "audit_progress.json", {
                        "status": "auditing", "verified_images": count, "total_images": len(rows),
                        "elapsed_seconds": time.time() - started, "updated_at": time.time()})
        stream.flush()
        os.fsync(stream.fileno())
    if archive_identity(archive) != expected["archive"]:
        raise ValueError("archive changed during content audit")
    # This checks every byte/RGB identity and the exact frozen row order together.
    validation = validate_content_manifest(bundle, pending, seal_dir, membership, expected)
    (seal_dir / "classes.json").write_bytes((bundle / "classes.json").read_bytes())
    calibration = membership.calibration_paths(bundle, classes, rows)
    write_json(seal_dir / "calibration.json", calibration)
    pending.rename(manifest)
    receipt = {**expected, "manifest_sha256": sha256(manifest), "verified_at_unix": time.time(),
               "method": "decode selected original JPEG byte ranges; reproduce frozen full-manifest SHA256",
               "elapsed_seconds": time.time() - started, "validation": validation}
    write_json(audit_path, receipt)
    write_json(seal_dir / "audit_progress.json", {"status": "passed", "verified_images": count,
               "total_images": len(rows), "manifest_sha256": sha256(manifest), "updated_at": time.time()})
    return receipt, index_receipt


class TarOriginalImages:
    """Same views/checks as pinned OriginalImages; only the byte source changes."""

    def __init__(self, archive, index_path, rows, role, seed=2026, epoch=0):
        if role not in ("train", "dev", "calibration"):
            raise ValueError("TRAIN-side roles only")
        self.archive, self.rows = str(Path(archive).resolve()), rows
        self.entries = load_index(index_path)
        self.role, self.seed, self.epoch = role, seed, epoch
        self._stream, self._pid = None, None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_stream"], state["_pid"] = None, None
        return state

    def close(self):
        if getattr(self, "_stream", None) is not None:
            self._stream.close()
            self._stream = None

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        data = importlib.import_module("experiments.w80_e23.data")
        r = self.rows[index]
        if self._stream is None or self._pid != os.getpid():
            self.close()
            self._stream = open(self.archive, "rb")
            self._pid = os.getpid()
        offset, size = self.entries[data.safe_relative(r["path"])]
        self._stream.seek(offset)
        raw = self._stream.read(size)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != r["sha"]:
            raise ValueError("original tar image bytes changed: " + r["path"])
        with Image.open(io.BytesIO(raw)) as source:
            im = source.convert("RGB")
        if hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest() != r["rgb"]:
            raise ValueError("decoded RGB identity changed: " + r["path"])
        if self.role == "train":
            view_seed = int(data.digest([self.seed, self.epoch, r["path"]])[:16], 16) % 2**63
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(view_seed)
                i, j, h, w = T.RandomResizedCrop.get_params(im, (.08, 1.), (3/4, 4/3))
                im = F.resized_crop(im, i, j, h, w, (160, 160), T.InterpolationMode.BICUBIC, antialias=True)
                if torch.rand(()) < .5:
                    im = F.hflip(im)
        else:
            im = F.center_crop(F.resize(im, 182, T.InterpolationMode.BICUBIC, antialias=True), [160, 160])
        return F.to_tensor(im), int(r["label"])


def loader(local, rows, role, epoch=0, order=None, cursor=0, workers=0, prefetch=1):
    import torch
    from torch.utils.data import DataLoader
    execution = importlib.import_module("experiments.w80_e23.execution")
    dataset = TarOriginalImages(local["archive"], local["index_path"], rows, role, seed=2026, epoch=epoch)
    ids = (order if order is not None else list(range(len(rows))))[cursor:]
    options = dict(multiprocessing_context="spawn", prefetch_factor=prefetch,
                   worker_init_fn=execution.worker_init) if workers else {}
    return DataLoader(dataset, batch_size=32, sampler=ids, num_workers=workers,
                      pin_memory=False, drop_last=False, persistent_workers=False,
                      generator=torch.Generator().manual_seed(2026 + epoch), **options)
