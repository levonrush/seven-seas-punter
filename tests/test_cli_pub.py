import argparse
import datetime as dt
import sys
import types

# `workflow.cli` imports storage modules that import duckdb. For CLI wiring tests we only
# need parser/dispatch behavior, so we stub duckdb when the dependency is unavailable.
sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(DuckDBPyConnection=object, connect=lambda *args, **kwargs: None),
)

from workflow.cli import build_parser, cmd_pub


def _pub_args(**kwargs) -> argparse.Namespace:
    """Build a full argparse namespace for pub command tests with stable defaults."""
    defaults = {
        "command": "pub",
        "cutoff_minutes": 10,
        "dry_run": False,
        "market_types": "ALL",
        "output": None,
        "min_prob": None,
        "budget": None,
        "allocation_method": "fractional_kelly",
        "kelly_fraction": 0.25,
        "max_bet_pct": 0.2,
        "execution_domain": None,
        "tab_translation_model": None,
        "tab_odds_quantile": None,
        "tab_fallback_haircut": None,
        "tab_fallback_spread": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cmd_pub_sets_default_output_and_calls_score(monkeypatch):
    captured = {"output": None, "called": 0, "execution_domain": None, "tab_odds_quantile": None}

    def fake_cmd_score(args):  # noqa: ANN001 - mirror runtime signature
        captured["called"] += 1
        captured["output"] = args.output
        captured["execution_domain"] = args.execution_domain
        captured["tab_odds_quantile"] = args.tab_odds_quantile

    monkeypatch.setattr("workflow.cli.cmd_score", fake_cmd_score)

    args = _pub_args(output=None)
    cmd_pub(args)

    expected = f"artifacts/pub_sheet_{dt.date.today().isoformat()}.csv"
    assert captured["called"] == 1
    assert captured["output"] == expected
    assert captured["execution_domain"] == "tab"
    assert captured["tab_odds_quantile"] == 0.10


def test_parser_accepts_pub_alias_sheet():
    parser = build_parser()
    args = parser.parse_args(["sheet"])
    assert args.command == "sheet"
    assert args.func is cmd_pub
    assert args.market_types == "ALL"
    assert args.execution_domain == "tab"


def test_parser_accepts_pub_budget_options():
    parser = build_parser()
    args = parser.parse_args(
        ["pub", "--budget", "50", "--allocation-method", "equal", "--max-bet-pct", "0.3"]
    )
    assert args.command == "pub"
    assert args.budget == 50.0
    assert args.allocation_method == "equal"
    assert args.max_bet_pct == 0.3


def test_parser_accepts_pub_tab_domain_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "pub",
            "--execution-domain",
            "tab",
            "--tab-translation-model",
            "artifacts/tab_translation.joblib",
            "--tab-odds-quantile",
            "0.2",
            "--tab-fallback-haircut",
            "0.12",
            "--tab-fallback-spread",
            "0.15",
        ]
    )
    assert args.execution_domain == "tab"
    assert args.tab_translation_model == "artifacts/tab_translation.joblib"
    assert args.tab_odds_quantile == 0.2
    assert args.tab_fallback_haircut == 0.12
    assert args.tab_fallback_spread == 0.15
