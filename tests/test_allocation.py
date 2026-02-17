import pandas as pd

from shared.backtest.allocation import allocate_stakes_from_budget, summarize_budget_usage


def _sample_candidates() -> pd.DataFrame:
    """Build a stable candidate frame for stake-allocation unit tests."""
    return pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 101, "p_hat": 0.35, "price": 4.0, "ev": 0.12},
            {"market_id": "1.2", "selection_id": 102, "p_hat": 0.28, "price": 5.0, "ev": 0.08},
            {"market_id": "1.3", "selection_id": 103, "p_hat": 0.20, "price": 7.0, "ev": 0.03},
        ]
    )


def test_fractional_kelly_allocation_respects_budget_and_caps():
    frame = _sample_candidates()
    allocated = allocate_stakes_from_budget(
        frame,
        budget=100.0,
        commission=0.05,
        method="fractional_kelly",
        kelly_fraction=0.25,
        max_bet_pct=0.2,
    )
    assert "suggested_stake" in allocated.columns
    assert allocated["suggested_stake"].sum() <= 100.0 + 1e-6
    assert allocated["suggested_stake"].max() <= 20.0 + 1e-6
    assert allocated["suggested_stake"].iloc[0] >= allocated["suggested_stake"].iloc[1]


def test_equal_allocation_splits_budget_evenly():
    frame = _sample_candidates()
    allocated = allocate_stakes_from_budget(
        frame,
        budget=90.0,
        method="equal",
        max_bet_pct=1.0,
    )
    assert all(value == 30.0 for value in allocated["suggested_stake"].tolist())


def test_summarize_budget_usage_reports_remaining_cash():
    frame = pd.DataFrame({"suggested_stake": [10.0, 15.5]})
    used, remaining = summarize_budget_usage(frame, budget=40.0)
    assert used == 25.5
    assert remaining == 14.5

