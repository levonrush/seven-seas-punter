from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse timestamp-like values so provider exports map to one UTC timeline."""
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _to_int(value: Any) -> int | None:
    """Coerce numeric-like values to integers for stable storage typing."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """Coerce numeric-like values to floats for stable storage typing."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    """Normalize provider booleans so fallback rules can be applied consistently."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "1", "yes", "y"}:
        return True
    if token in {"false", "0", "no", "n"}:
        return False
    return None


def _class_index_from_label(label: Any) -> float | None:
    """Map provider class labels to a monotone numeric scale for progression features."""
    if label is None:
        return None
    text = str(label).strip().upper()
    if not text:
        return None
    if "GROUP 1" in text or "G1" in text:
        return 1.0
    if "GROUP 2" in text or "G2" in text:
        return 2.0
    if "GROUP 3" in text or "G3" in text:
        return 3.0
    if "LISTED" in text:
        return 4.0
    if "MAIDEN" in text:
        return 20.0
    if "CLASS" in text:
        for token in text.replace("/", " ").split():
            try:
                return 10.0 + float(token)
            except ValueError:
                continue
    if "BM" in text or "BENCHMARK" in text:
        for token in text.replace("-", " ").split():
            try:
                rating = float(token)
                # Lower numeric score means stronger class.
                return max(5.0, 25.0 - (rating / 10.0))
            except ValueError:
                continue
    return None


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    """Extract record lists from JSON payload variants produced by provider exports."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ["records", "data", "rows", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [payload]
    return []


def _known_row_keys() -> set[str]:
    """Return the canonical key set so unknown fields can be preserved in metadata."""
    return {
        "source",
        "market_id",
        "selection_id",
        "snapshot_time",
        "race_start_time",
        "seconds_to_start",
        "run_index",
        "runner_name",
        "horse_name",
        "jockey_name",
        "trainer_name",
        "track",
        "surface",
        "distance_m",
        "class_label",
        "class_index",
        "track_condition",
        "run_date",
        "run_finish_pos",
        "run_field_size",
        "run_distance_m",
        "run_surface",
        "run_track",
        "run_class_label",
        "run_class_index",
        "run_track_condition",
        "run_sectional_time",
        "run_speed_rating",
        "run_weight_value",
        "run_barrier",
        "run_jockey_name",
        "run_trainer_name",
        "run_won",
        "run_placed",
        "metadata",
    }


def _normalize_external_form_row(
    raw_row: dict[str, Any],
    *,
    source: str,
    fallback_snapshot_time: pd.Timestamp | None,
    default_cutoff_minutes: int,
    inferred_run_index: int | None,
) -> dict[str, Any] | None:
    """Normalize one external form run row so feature builders can rely on stable field names and types."""
    market_id = str(raw_row.get("market_id") or "").strip()
    selection_id = _to_int(raw_row.get("selection_id"))
    race_start_time = _parse_timestamp(raw_row.get("race_start_time") or raw_row.get("market_time"))
    if not market_id or selection_id is None or race_start_time is None:
        return None

    snapshot_time = _parse_timestamp(raw_row.get("snapshot_time")) or fallback_snapshot_time
    if snapshot_time is None:
        snapshot_time = race_start_time - pd.Timedelta(minutes=default_cutoff_minutes)
    seconds_to_start = _to_int(raw_row.get("seconds_to_start"))
    if seconds_to_start is None and snapshot_time is not None:
        seconds_to_start = int((race_start_time - snapshot_time).total_seconds())

    run_index = _to_int(raw_row.get("run_index") or raw_row.get("runNumber") or inferred_run_index)
    if run_index is None:
        run_index = 1

    class_label = raw_row.get("class_label") or raw_row.get("race_class")
    run_class_label = raw_row.get("run_class_label") or raw_row.get("run_race_class")
    run_finish_pos = _to_int(raw_row.get("run_finish_pos") or raw_row.get("finish_pos"))
    run_won = _to_bool(raw_row.get("run_won") or raw_row.get("won"))
    run_placed = _to_bool(raw_row.get("run_placed") or raw_row.get("placed"))
    if run_won is None and run_finish_pos is not None:
        run_won = run_finish_pos == 1
    if run_placed is None and run_finish_pos is not None:
        run_placed = run_finish_pos <= 3

    metadata = raw_row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key, value in raw_row.items():
        if key in _known_row_keys():
            continue
        if value is None:
            continue
        metadata[key] = value

    return {
        "source": source,
        "market_id": market_id,
        "selection_id": selection_id,
        "snapshot_time": snapshot_time,
        "race_start_time": race_start_time,
        "seconds_to_start": seconds_to_start,
        "run_index": run_index,
        "runner_name": raw_row.get("runner_name"),
        "horse_name": raw_row.get("horse_name"),
        "jockey_name": raw_row.get("jockey_name"),
        "trainer_name": raw_row.get("trainer_name"),
        "track": raw_row.get("track"),
        "surface": raw_row.get("surface"),
        "distance_m": _to_int(raw_row.get("distance_m")),
        "class_label": class_label,
        "class_index": _to_float(raw_row.get("class_index")) or _class_index_from_label(class_label),
        "track_condition": raw_row.get("track_condition"),
        "run_date": _parse_timestamp(raw_row.get("run_date") or raw_row.get("date")),
        "run_finish_pos": run_finish_pos,
        "run_field_size": _to_int(raw_row.get("run_field_size") or raw_row.get("field_size")),
        "run_distance_m": _to_int(raw_row.get("run_distance_m") or raw_row.get("distance")),
        "run_surface": raw_row.get("run_surface"),
        "run_track": raw_row.get("run_track"),
        "run_class_label": run_class_label,
        "run_class_index": _to_float(raw_row.get("run_class_index")) or _class_index_from_label(run_class_label),
        "run_track_condition": raw_row.get("run_track_condition"),
        "run_sectional_time": _to_float(raw_row.get("run_sectional_time") or raw_row.get("sectional_time")),
        "run_speed_rating": _to_float(raw_row.get("run_speed_rating") or raw_row.get("speed_rating")),
        "run_weight_value": _to_float(raw_row.get("run_weight_value") or raw_row.get("weight_value")),
        "run_barrier": _to_int(raw_row.get("run_barrier") or raw_row.get("barrier")),
        "run_jockey_name": raw_row.get("run_jockey_name") or raw_row.get("jockey"),
        "run_trainer_name": raw_row.get("run_trainer_name") or raw_row.get("trainer"),
        "run_won": run_won,
        "run_placed": run_placed,
        "metadata": metadata,
    }


def _expand_record_runs(record: dict[str, Any]) -> Iterable[tuple[dict[str, Any], int | None]]:
    """Yield one run-shaped row per record entry so nested provider payloads become tabular features."""
    runs = record.get("runs")
    if isinstance(runs, list) and runs:
        base = {key: value for key, value in record.items() if key != "runs"}
        for index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                continue
            merged = dict(base)
            merged.update(run)
            yield merged, index
        return
    yield record, None


def load_external_form_rows(
    input_path: str,
    *,
    source: str,
    default_cutoff_minutes: int = 10,
    default_snapshot_time: str | None = None,
) -> list[dict[str, Any]]:
    """Load and normalize provider export rows so external form runs can be stored and replayed at cutoff."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"External form input not found: {path}")

    records: list[dict[str, Any]]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    elif suffix == ".parquet":
        records = pd.read_parquet(path).to_dict(orient="records")
    elif suffix in {".jsonl", ".ndjson"}:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = _extract_records(payload)
    else:
        raise ValueError(f"Unsupported external form file type: {path.suffix}")

    fallback_snapshot_time = _parse_timestamp(default_snapshot_time)
    normalized_rows: list[dict[str, Any]] = []
    for record in records:
        for run_row, inferred_run_index in _expand_record_runs(record):
            normalized = _normalize_external_form_row(
                run_row,
                source=source,
                fallback_snapshot_time=fallback_snapshot_time,
                default_cutoff_minutes=default_cutoff_minutes,
                inferred_run_index=inferred_run_index,
            )
            if normalized is not None:
                normalized_rows.append(normalized)
    return normalized_rows

