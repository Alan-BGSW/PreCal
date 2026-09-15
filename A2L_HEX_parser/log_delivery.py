from __future__ import annotations

import os
import shutil
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUBPATH = Path("CalAiLogs") / "Incoming"


def _candidate_onedrive_roots() -> list[Path]:
    candidates: list[Path] = []

    for env_name in ("CALAI_LOG_UPLOAD_DIR", "OneDriveCommercial", "OneDriveConsumer", "OneDrive"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw))

    home = Path.home()
    for p in home.glob("OneDrive*"):
        candidates.append(p)

    dedup: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def resolve_upload_dir(subpath: Path = DEFAULT_SUBPATH) -> Path | None:
    subpath_parts = subpath.parts
    if len(subpath_parts) < 2:
        return None

    calai_folder_name = subpath_parts[0]
    incoming_leaf = subpath_parts[-1]

    def _find_calai_root(root: Path) -> Path | None:
        direct = root / calai_folder_name
        if direct.exists() and direct.is_dir():
            return direct

        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.lower().endswith(calai_folder_name.lower()):
                    return child
        except OSError:
            return None
        return None

    roots = _candidate_onedrive_roots()
    for root in roots:
        if root.exists() and root.is_dir():
            calai_root = _find_calai_root(root)
            if calai_root is not None:
                target = calai_root / incoming_leaf
                target.mkdir(parents=True, exist_ok=True)
                return target

            target = root / subpath
            target.mkdir(parents=True, exist_ok=True)
            return target
    return None


def _build_target_name(src_json_path: Path) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().replace(" ", "_")
    run_id = uuid.uuid4().hex[:10]
    return f"calibration_log_{host}_{ts}_{run_id}{src_json_path.suffix}"


def _queue_file(src_json_path: Path, pending_dir: Path) -> Path:
    pending_dir.mkdir(parents=True, exist_ok=True)
    queued_name = f"queued_{uuid.uuid4().hex}_{src_json_path.name}"
    queued_path = pending_dir / queued_name
    shutil.copy2(src_json_path, queued_path)
    return queued_path


def _flush_pending(pending_dir: Path, upload_dir: Path) -> tuple[int, int]:
    if not pending_dir.exists():
        return 0, 0

    sent = 0
    failed = 0
    for pending_file in sorted(pending_dir.glob("*.json")):
        try:
            dst = upload_dir / _build_target_name(pending_file)
            shutil.copy2(pending_file, dst)
            pending_file.unlink(missing_ok=True)
            sent += 1
        except Exception:
            failed += 1
    return sent, failed


def upload_log_json(
    log_json_path: str | Path,
    workspace_dir: str | Path,
    upload_subpath: Path = DEFAULT_SUBPATH,
) -> dict[str, Any]:
    src = Path(log_json_path)
    work = Path(workspace_dir)
    pending_dir = work / "log_upload_pending"

    if not src.exists():
        return {
            "ok": False,
            "status": "missing_log_file",
            "message": f"Log file not found: {src}",
        }

    upload_dir = resolve_upload_dir(upload_subpath)
    if upload_dir is None:
        queued_path = _queue_file(src, pending_dir)
        return {
            "ok": False,
            "status": "queued_no_onedrive",
            "message": "OneDrive/SharePoint synced folder not found. Log queued for later upload.",
            "queued_path": str(queued_path),
        }

    flushed_sent, flushed_failed = _flush_pending(pending_dir, upload_dir)

    try:
        dst = upload_dir / _build_target_name(src)
        shutil.copy2(src, dst)
        return {
            "ok": True,
            "status": "uploaded",
            "message": "Detailed JSON log uploaded to OneDrive/SharePoint folder.",
            "uploaded_path": str(dst),
            "pending_flushed": flushed_sent,
            "pending_failed": flushed_failed,
        }
    except Exception as exc:
        queued_path = _queue_file(src, pending_dir)
        return {
            "ok": False,
            "status": "queued_upload_failed",
            "message": f"Upload failed, log queued for retry: {type(exc).__name__}: {exc}",
            "queued_path": str(queued_path),
            "pending_flushed": flushed_sent,
            "pending_failed": flushed_failed,
        }
