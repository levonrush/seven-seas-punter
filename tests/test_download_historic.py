import datetime as dt
from types import SimpleNamespace

import requests

from workflow.download_historic import (
    _extract_basket_size_mb,
    _load_manifest,
    _is_retryable_download_error,
    _normalize_market_types,
    _plan_sharded_windows,
    _resolve_workers,
)


def test_normalize_market_types_all_disables_filter():
    assert _normalize_market_types(["ALL"]) == []
    assert _normalize_market_types(["*", "WIN"]) == []


def test_normalize_market_types_normalizes_and_deduplicates():
    assert _normalize_market_types([" win ", "EXACTA", "Win", "QUINELLA"]) == [
        "WIN",
        "EXACTA",
        "QUINELLA",
    ]


def test_normalize_market_types_empty_values_return_empty_list():
    assert _normalize_market_types([]) == []


def test_extract_basket_size_mb_handles_bytes_payload():
    payload = {"totalSizeBytes": 10 * 1024 * 1024}
    assert _extract_basket_size_mb(payload) == 10.0


def test_extract_basket_size_mb_handles_mb_payload():
    payload = {"sizeMB": 875}
    assert _extract_basket_size_mb(payload) == 875


def test_plan_sharded_windows_splits_large_windows():
    class _DummyClient:
        """Return synthetic basket sizes so sharding logic can be validated without network calls."""

        def get_basket_size(self, payload):  # noqa: ANN001 - mirrors real client method
            span_days = payload["toDay"] - payload["fromDay"] + 1
            return {"sizeMB": 2400 if span_days > 8 else 600}

    args = SimpleNamespace(
        sport="Horse Racing",
        event_id=None,
        event_name=None,
        target_basket_mb=1000.0,
        max_shard_depth=4,
    )
    windows = _plan_sharded_windows(
        client=_DummyClient(),
        args=args,
        plan_name="Basic Plan",
        from_date=dt.date(2026, 1, 1),
        to_date=dt.date(2026, 1, 31),
        market_types=["WIN"],
        countries=["AU"],
        file_types=["M"],
    )
    assert len(windows) > 1
    assert all((window_to - window_from).days <= 8 for window_from, window_to in windows)


def test_retryable_download_error_flags_429_and_timeouts():
    error = requests.HTTPError("rate limited")
    error.response = SimpleNamespace(status_code=429)
    assert _is_retryable_download_error(error) is True
    assert _is_retryable_download_error(requests.ReadTimeout("timed out")) is True


def test_resolve_workers_uses_auto_preset_when_missing(monkeypatch):
    monkeypatch.setattr("workflow.download_historic.recommended_historic_workers", lambda: 6)
    workers, auto = _resolve_workers(None)
    assert workers == 6
    assert auto is True


def test_load_manifest_recovers_from_empty_file(tmp_path):
    manifest_path = tmp_path / "historic_manifest.json"
    manifest_path.write_text("", encoding="utf-8")

    manifest = _load_manifest(manifest_path)

    assert manifest["files"] == {}
    assert manifest["version"] == 1
    assert manifest["_basenames"] == set()


def test_load_manifest_recovers_from_invalid_json_and_backups(tmp_path):
    manifest_path = tmp_path / "historic_manifest.json"
    manifest_path.write_text("{", encoding="utf-8")

    manifest = _load_manifest(manifest_path)

    assert manifest["files"] == {}
    assert manifest["version"] == 1
    backup_files = list(tmp_path.glob("historic_manifest.json.bad_*"))
    assert len(backup_files) == 1
