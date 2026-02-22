import json
from pathlib import Path

from shared.utils.manifest_repair import repair_manifests


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON fixtures so manifest repair tests can exercise real file IO behavior."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_repair_manifests_prunes_bad_entries(tmp_path):
    bad_list = tmp_path / "bad.txt"
    bad_list.write_text("bad1.bz2\nbad2.bz2\n", encoding="utf-8")

    historic_manifest = tmp_path / "historic_manifest.json"
    _write_json(
        historic_manifest,
        {
            "version": 1,
            "files": {
                "/xds/path/good.bz2": {"basename": "good.bz2"},
                "/xds/path/bad1.bz2": {"basename": "bad1.bz2"},
                "/xds/path/bad2.bz2": {"basename": "bad2.bz2"},
            },
        },
    )

    ingest_manifest = tmp_path / "ingest_manifest.json"
    _write_json(
        ingest_manifest,
        {
            "version": 1,
            "members": {
                "good.bz2": {"ingested_at": "2026-01-01T00:00:00+00:00"},
                "bad1.bz2": {"ingested_at": "2026-01-01T00:00:00+00:00"},
            },
        },
    )

    summary = repair_manifests(
        bad_list_path=bad_list,
        historic_manifest_path=historic_manifest,
        ingest_manifest_path=ingest_manifest,
        backup=False,
    )

    updated_historic = json.loads(historic_manifest.read_text(encoding="utf-8"))
    updated_ingest = json.loads(ingest_manifest.read_text(encoding="utf-8"))

    assert summary["bad_basenames"] == 2
    assert summary["historic_removed"] == 2
    assert summary["ingest_removed"] == 1
    assert "/xds/path/good.bz2" in updated_historic["files"]
    assert "/xds/path/bad1.bz2" not in updated_historic["files"]
    assert "good.bz2" in updated_ingest["members"]
    assert "bad1.bz2" not in updated_ingest["members"]


def test_repair_manifests_creates_backups(tmp_path):
    bad_list = tmp_path / "bad.txt"
    bad_list.write_text("bad1.bz2\n", encoding="utf-8")

    historic_manifest = tmp_path / "historic_manifest.json"
    _write_json(historic_manifest, {"version": 1, "files": {"/xds/path/bad1.bz2": {"basename": "bad1.bz2"}}})

    ingest_manifest = tmp_path / "ingest_manifest.json"
    _write_json(ingest_manifest, {"version": 1, "members": {"bad1.bz2": {"ingested_at": "2026-01-01"}}})

    summary = repair_manifests(
        bad_list_path=bad_list,
        historic_manifest_path=historic_manifest,
        ingest_manifest_path=ingest_manifest,
        backup=True,
    )

    backup_paths = summary["backup_paths"]
    assert len(backup_paths) == 2
    assert all(Path(path).exists() for path in backup_paths)
