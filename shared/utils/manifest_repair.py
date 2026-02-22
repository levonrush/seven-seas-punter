from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_timestamp_slug() -> str:
    """Generate a stable UTC timestamp suffix so manifest backups stay sortable and collision-resistant."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    """Load JSON from disk so manifest pruning can run against validated object payloads."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist JSON with deterministic formatting so manifest diffs remain readable and reviewable."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_bad_basenames(path: Path) -> set[str]:
    """Load bad basenames from a newline-delimited file so repair runs can target only known-corrupt members."""
    if not path.exists():
        raise FileNotFoundError(f"Bad-member list not found: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def backup_manifest(path: Path, timestamp: str | None = None) -> Path:
    """Create a timestamped backup before edits so manifest repair remains reversible."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    ts = timestamp or _utc_timestamp_slug()
    backup_path = path.with_name(f"{path.name}.bak_{ts}")
    shutil.copy2(path, backup_path)
    return backup_path


def prune_historic_manifest(manifest_path: Path, bad_basenames: set[str]) -> tuple[int, int]:
    """Remove bad members from historic manifest entries so downloader retries only corrupted payloads."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Historic manifest not found: {manifest_path}")
    payload = _read_json(manifest_path)
    files = payload.get("files") or {}
    before = len(files)
    payload["files"] = {
        key: value for key, value in files.items() if Path(str(key)).name not in bad_basenames
    }
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, payload)
    after = len(payload["files"])
    return before, after


def prune_ingest_manifest(manifest_path: Path, bad_basenames: set[str]) -> tuple[int, int]:
    """Remove bad members from ingest manifest so incremental ingest does not skip failed decompression members."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ingest manifest not found: {manifest_path}")
    payload = _read_json(manifest_path)
    members = payload.get("members") or {}
    before = len(members)
    payload["members"] = {
        key: value for key, value in members.items() if Path(str(key)).name not in bad_basenames
    }
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, payload)
    after = len(payload["members"])
    return before, after


def repair_manifests(
    *,
    bad_list_path: Path,
    historic_manifest_path: Path,
    ingest_manifest_path: Path,
    backup: bool = True,
) -> Dict[str, Any]:
    """Prune bad members from both manifests so download/ingest pipelines can automatically recover holes."""
    bad_basenames = load_bad_basenames(bad_list_path)
    summary: Dict[str, Any] = {
        "bad_basenames": len(bad_basenames),
        "bad_list_path": str(bad_list_path),
        "historic_manifest_path": str(historic_manifest_path),
        "ingest_manifest_path": str(ingest_manifest_path),
        "backup_paths": [],
    }
    if backup:
        timestamp = _utc_timestamp_slug()
        summary["backup_paths"] = [
            str(backup_manifest(historic_manifest_path, timestamp)),
            str(backup_manifest(ingest_manifest_path, timestamp)),
        ]
    historic_before, historic_after = prune_historic_manifest(historic_manifest_path, bad_basenames)
    ingest_before, ingest_after = prune_ingest_manifest(ingest_manifest_path, bad_basenames)
    summary["historic_removed"] = historic_before - historic_after
    summary["ingest_removed"] = ingest_before - ingest_after
    summary["historic_before"] = historic_before
    summary["historic_after"] = historic_after
    summary["ingest_before"] = ingest_before
    summary["ingest_after"] = ingest_after
    return summary
