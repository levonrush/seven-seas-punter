import json

from shared.form.ingest import load_external_form_rows


def test_load_external_form_rows_flattens_nested_runs_and_derives_defaults(tmp_path):
    payload = [
        {
            "market_id": "1.555",
            "selection_id": 42,
            "race_start_time": "2024-01-01T05:00:00Z",
            "runner_name": "Horse Prime",
            "track": "Randwick",
            "surface": "TURF",
            "distance_m": 1200,
            "runs": [
                {
                    "date": "2023-12-20T05:00:00Z",
                    "finish_pos": 1,
                    "distance": 1200,
                    "run_surface": "TURF",
                    "run_track": "Randwick",
                    "sectional_time": 34.1,
                    "speed_rating": 93.0,
                },
                {
                    "date": "2023-12-01T05:00:00Z",
                    "finish_pos": 3,
                    "distance": 1200,
                    "run_surface": "TURF",
                    "run_track": "Randwick",
                    "sectional_time": 34.8,
                    "speed_rating": 90.0,
                },
            ],
        }
    ]
    input_path = tmp_path / "form.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_external_form_rows(
        str(input_path),
        source="punting_form",
        default_cutoff_minutes=10,
    )
    assert len(rows) == 2
    first = rows[0]
    assert first["source"] == "punting_form"
    assert first["market_id"] == "1.555"
    assert first["selection_id"] == 42
    assert first["run_index"] == 1
    assert first["seconds_to_start"] == 600
    assert first["run_finish_pos"] == 1
    assert first["run_won"] is True
    assert first["run_placed"] is True
    assert first["run_speed_rating"] == 93.0


def test_load_external_form_rows_supports_jsonl_rows(tmp_path):
    rows = [
        {
            "market_id": "1.777",
            "selection_id": 9,
            "race_start_time": "2024-02-01T02:00:00Z",
            "snapshot_time": "2024-02-01T01:50:00Z",
            "run_index": 1,
            "run_date": "2024-01-20T02:00:00Z",
            "run_finish_pos": 4,
            "run_won": False,
            "run_placed": False,
        }
    ]
    input_path = tmp_path / "form.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    parsed = load_external_form_rows(str(input_path), source="provider_x")
    assert len(parsed) == 1
    assert parsed[0]["source"] == "provider_x"
    assert parsed[0]["seconds_to_start"] == 600
    assert parsed[0]["run_won"] is False
