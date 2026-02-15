import argparse
import sys
import types
from pathlib import Path

# `workflow.cli` imports storage modules that import duckdb. For CLI wiring tests we only
# need parser/dispatch behavior, so we stub duckdb when the dependency is unavailable.
sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(DuckDBPyConnection=object, connect=lambda *args, **kwargs: None),
)

from workflow.cli import _build_live_overrides, cmd_live


def _live_args(**kwargs) -> argparse.Namespace:
    """Build a complete argparse namespace for cmd_live tests with sensible defaults."""
    defaults = {
        "config": "config/live.yaml",
        "write_config": None,
        "force": False,
        "dry_run": None,
        "once": False,
        "max_iterations": None,
        "poll_interval_seconds": None,
        "min_edge": None,
        "max_stake_per_market": None,
        "max_daily_exposure": None,
        "ignore_within_minutes": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_live_overrides_once_takes_precedence():
    args = _live_args(
        dry_run=False,
        once=True,
        max_iterations=9,
        poll_interval_seconds=2.5,
        min_edge=0.15,
        max_stake_per_market=4.0,
        max_daily_exposure=20.0,
        ignore_within_minutes=6.0,
    )

    overrides = _build_live_overrides(args)

    assert overrides["dry_run"] is False
    assert overrides["max_iterations"] == 1
    assert overrides["poll_interval_seconds"] == 2.5
    assert overrides["strategy"]["min_edge"] == 0.15
    assert overrides["safety"]["max_stake_per_market"] == 4.0
    assert overrides["safety"]["max_daily_exposure"] == 20.0
    assert overrides["safety"]["ignore_within_minutes"] == 6.0


def test_cmd_live_write_config_short_circuits_run_loop(monkeypatch, tmp_path):
    captured = {"write_path": None, "run_called": False}

    def fake_write(path: str, overwrite: bool = False) -> Path:
        captured["write_path"] = path
        return Path(path)

    def fake_run(config_path: str, overrides=None) -> None:  # noqa: ANN001 - mirror runtime signature
        captured["run_called"] = True

    monkeypatch.setattr("shared.live.betfair_live.write_live_config_template", fake_write)
    monkeypatch.setattr("shared.live.betfair_live.run_live_loop", fake_run)

    args = _live_args(write_config=str(tmp_path / "live.yaml"))
    cmd_live(args)

    assert captured["write_path"] == str(tmp_path / "live.yaml")
    assert captured["run_called"] is False


def test_cmd_live_passes_overrides_to_run_loop(monkeypatch):
    captured = {"config_path": None, "overrides": None}

    def fake_run(config_path: str, overrides=None) -> None:  # noqa: ANN001 - mirror runtime signature
        captured["config_path"] = config_path
        captured["overrides"] = overrides

    monkeypatch.setattr("shared.live.betfair_live.run_live_loop", fake_run)

    args = _live_args(
        config="config/live.yaml",
        dry_run=True,
        once=True,
        max_stake_per_market=3.0,
    )
    cmd_live(args)

    assert captured["config_path"] == "config/live.yaml"
    assert captured["overrides"]["dry_run"] is True
    assert captured["overrides"]["max_iterations"] == 1
    assert captured["overrides"]["safety"]["max_stake_per_market"] == 3.0
