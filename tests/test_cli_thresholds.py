import sys
import types

import pandas as pd

sys.modules.setdefault(
    "duckdb",
    types.SimpleNamespace(DuckDBPyConnection=object, connect=lambda *args, **kwargs: None),
)

from workflow.cli import (
    _apply_probability_thresholds,
    _apply_score_strategy_filters,
    _resolve_strategy_filter_values,
)


def test_apply_probability_thresholds_uses_bucket_specific_values_with_global_fallback():
    frame = pd.DataFrame(
        [
            {"market_type": "WIN", "p_hat": 0.30},
            {"market_type": "PLACE", "p_hat": 0.39},
            {"market_type": None, "p_hat": 0.36},
        ]
    )
    filtered = _apply_probability_thresholds(
        frame=frame,
        label="Test",
        global_min_prob=0.35,
        bucket_min_probs={"WIN": 0.25, "PLACE": 0.40},
    )
    assert len(filtered) == 2
    assert filtered["p_hat"].tolist() == [0.30, 0.36]


def test_apply_score_strategy_filters_matches_backtest_style_constraints():
    frame = pd.DataFrame(
        [
            {"p_hat": 0.60, "decision_price": 2.0, "ev": 0.14, "spread_t10": 0.2},
            {"p_hat": 0.35, "decision_price": 3.0, "ev": 0.01, "spread_t10": 0.2},
            {"p_hat": 0.70, "decision_price": 2.0, "ev": 0.33, "spread_t10": 1.5},
        ]
    )
    filtered = _apply_score_strategy_filters(
        frame=frame,
        cutoff_minutes=10,
        min_ev=0.02,
        min_edge=0.1,
        max_price=200.0,
        max_edge_multiplier=5.0,
        max_spread=1.0,
        apply_probability_safety=False,
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["p_hat"] == 0.60


def test_rescue_strategy_defaults_apply_when_filters_not_explicitly_set():
    args = types.SimpleNamespace(
        min_ev=None,
        min_edge=None,
        max_price=None,
        max_edge_mult=None,
        rescue_guards=True,
    )
    min_ev, min_edge, max_price, max_edge_mult = _resolve_strategy_filter_values(args, "Test")
    assert min_ev == 0.02
    assert min_edge == 0.01
    assert max_price == 30.0
    assert max_edge_mult == 1.2
