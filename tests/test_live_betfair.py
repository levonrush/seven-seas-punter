import datetime as dt

import pandas as pd
import pytest

from shared.live.betfair_live import (
    DEFAULT_LIVE_CONFIG,
    apply_safety_gates,
    build_market_filter,
    run_live_iteration,
    run_live_loop,
)


def _base_config() -> dict:
    """Return a minimal mutable config fixture so tests can isolate specific safety behaviors."""
    config = {
        **DEFAULT_LIVE_CONFIG,
        "model": dict(DEFAULT_LIVE_CONFIG["model"]),
        "markets": dict(DEFAULT_LIVE_CONFIG["markets"]),
        "strategy": dict(DEFAULT_LIVE_CONFIG["strategy"]),
        "safety": dict(DEFAULT_LIVE_CONFIG["safety"]),
        "state": dict(DEFAULT_LIVE_CONFIG["state"]),
    }
    return config


def test_apply_safety_gates_dry_run_marks_approved_orders_as_simulated():
    config = _base_config()
    config["dry_run"] = True
    config["strategy"]["stake_per_bet"] = 2.0
    config["safety"]["max_stake_per_market"] = 10.0
    config["safety"]["max_daily_exposure"] = 20.0
    candidates = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "selection_id": 100,
                "price": 3.5,
                "stake": 2.0,
                "p_hat": 0.35,
                "edge_pct": 0.2,
                "expected_value": 0.05,
                "minutes_to_start": 20.0,
            }
        ]
    )
    exposure_state = {"date": "2026-02-12", "total_stake": 0.0, "market_stake": {}}

    decisions = apply_safety_gates(candidates, config, exposure_state)

    assert len(decisions) == 1
    assert decisions[0]["approved"] is True
    assert decisions[0]["action"] == "DRY_RUN"
    assert exposure_state["total_stake"] == 2.0


def test_apply_safety_gates_rejects_market_cap_breach():
    config = _base_config()
    config["dry_run"] = True
    config["strategy"]["stake_per_bet"] = 3.0
    config["safety"]["max_stake_per_market"] = 5.0
    config["safety"]["max_daily_exposure"] = 50.0
    candidates = pd.DataFrame(
        [
            {
                "market_id": "1.22",
                "selection_id": 100,
                "price": 4.0,
                "stake": 3.0,
                "p_hat": 0.30,
                "edge_pct": 0.2,
                "expected_value": 0.05,
                "minutes_to_start": 30.0,
            },
            {
                "market_id": "1.22",
                "selection_id": 101,
                "price": 4.5,
                "stake": 3.0,
                "p_hat": 0.25,
                "edge_pct": 0.2,
                "expected_value": 0.05,
                "minutes_to_start": 30.0,
            },
        ]
    )
    exposure_state = {"date": "2026-02-12", "total_stake": 0.0, "market_stake": {}}

    decisions = apply_safety_gates(candidates, config, exposure_state)

    assert decisions[0]["approved"] is True
    assert decisions[1]["approved"] is False
    assert decisions[1]["reason"] == "market_cap_exceeded"
    assert exposure_state["total_stake"] == 3.0


class _MockClient:
    """Provide deterministic market and book payloads so live iteration can be tested offline."""

    def __init__(self) -> None:
        self.place_orders_calls = 0
        self.dry_run = True

    def list_market_catalogue(self, **kwargs):  # noqa: ANN003 - test stub mirrors runtime kwargs
        market_id = "1.123"
        return [
            {
                "market_id": market_id,
                "venue": "Testville",
                "race_start_time": dt.datetime(2026, 2, 12, 12, 30, tzinfo=dt.timezone.utc),
                "race_name": "Test Race",
                "market_type": "WIN",
                "country_code": "AU",
                "event_type": "horse_racing",
                "runners": [
                    {"market_id": market_id, "selection_id": 100, "runner_name": "Runner A", "stall_draw": 1},
                    {"market_id": market_id, "selection_id": 101, "runner_name": "Runner B", "stall_draw": 2},
                ],
            }
        ]

    def fetch_market_books(self, market_ids):  # noqa: ANN001 - test stub accepts list-like
        snapshot_time = dt.datetime(2026, 2, 12, 12, 0, tzinfo=dt.timezone.utc)
        race_start = dt.datetime(2026, 2, 12, 12, 30, tzinfo=dt.timezone.utc)
        return [
            {
                "market_id": market_ids[0],
                "selection_id": 100,
                "snapshot_time": snapshot_time,
                "seconds_to_start": 1800,
                "best_back_price": 3.2,
                "best_back_size": 100.0,
                "best_lay_price": 3.25,
                "best_lay_size": 100.0,
                "last_traded_price": 3.2,
                "total_matched": 1000.0,
                "runner_status": "ACTIVE",
                "venue": "Testville",
                "race_start_time": race_start,
                "race_name": "Test Race",
            },
            {
                "market_id": market_ids[0],
                "selection_id": 101,
                "snapshot_time": snapshot_time,
                "seconds_to_start": 1800,
                "best_back_price": 4.0,
                "best_back_size": 100.0,
                "best_lay_price": 4.1,
                "best_lay_size": 100.0,
                "last_traded_price": 4.0,
                "total_matched": 900.0,
                "runner_status": "ACTIVE",
                "venue": "Testville",
                "race_start_time": race_start,
                "race_name": "Test Race",
            },
        ]

    def place_orders(self, **kwargs):  # noqa: ANN003 - test stub mirrors runtime kwargs
        self.place_orders_calls += 1
        return {"status": "SUCCESS", "instruction_reports": [{"status": "SUCCESS"}]}


def test_run_live_iteration_dry_run_with_mock_client(monkeypatch):
    config = _base_config()
    config["dry_run"] = True
    config["strategy"]["min_ev"] = -1.0
    config["strategy"]["min_edge"] = -1.0
    config["safety"]["ignore_within_minutes"] = 1
    client = _MockClient()
    runtime_state = {
        "snapshot_history": [],
        "exposure_state": {"date": "2026-02-12", "total_stake": 0.0, "market_stake": {}},
    }

    monkeypatch.setattr(
        "shared.live.betfair_live.predict_probabilities",
        lambda model, calibrator, feature_df: pd.Series([0.35, 0.25], index=feature_df.index),
    )

    decisions = run_live_iteration(
        client=client,
        model=object(),
        calibrator=None,
        config=config,
        runtime_state=runtime_state,
        now_utc=dt.datetime(2026, 2, 12, 12, 0, tzinfo=dt.timezone.utc),
    )

    assert decisions
    assert decisions[0]["action"] == "DRY_RUN"
    assert decisions[0]["execution_status"] == "SIMULATED"
    assert client.place_orders_calls == 0


def test_run_live_loop_refuses_live_mode_when_auth_is_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "live.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dry_run: false",
                "max_iterations: 1",
                "poll_interval_seconds: 1",
                "model:",
                "  cutoff_minutes: 10",
                "  model_path: artifacts/model_cutoff_10.joblib",
                "  calibrator_path: artifacts/calibrator_cutoff_10.joblib",
            ]
        ),
        encoding="utf-8",
    )

    class _AuthFailClient:
        def __init__(self):
            self.dry_run = True

    monkeypatch.setattr("shared.live.betfair_live.BetfairClient", _AuthFailClient)

    with pytest.raises(RuntimeError):
        run_live_loop(str(config_path))


def test_build_market_filter_omits_market_type_codes_when_all():
    config = _base_config()
    config["markets"]["market_type_codes"] = ["ALL"]
    payload = build_market_filter(
        config=config,
        now_utc=dt.datetime(2026, 2, 12, 12, 0, tzinfo=dt.timezone.utc),
    )
    assert "market_type_codes" not in payload
