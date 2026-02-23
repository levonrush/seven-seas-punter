import pandas as pd

from shared.backtest.engine import compute_expected_value, run_backtest


def test_expected_value_computation():
    ev = compute_expected_value(prob=0.25, price=5.0, commission=0.05)
    # Expected profit: 0.25 * 4 * 0.95 - 0.75 = 0.20
    assert round(ev, 4) == 0.2


def test_backtest_metrics():
    feature_df = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "selection_id": 1,
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "back_price_t10": 3.0,
                "spread_t10": 0.1,
                "win_target": 1,
            },
            {
                "market_id": "1.1",
                "selection_id": 2,
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "back_price_t10": 4.0,
                "spread_t10": 0.2,
                "win_target": 0,
            },
        ]
    )
    probs = pd.Series([0.5, 0.2])
    bets, metrics = run_backtest(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        commission=0.05,
        min_ev=-1.0,  # allow both bets
        min_edge=-1.0,
        max_spread=0.5,
        max_price=10_000.0,
        max_edge_multiplier=1_000.0,
        stake=1.0,
    )
    assert metrics["bets"] == 2
    assert round(metrics["roi"], 2) > 0
    assert "max_drawdown" in metrics


def test_backtest_skips_nan_probabilities():
    feature_df = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "selection_id": 1,
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "back_price_t10": 5.0,
                "spread_t10": 0.1,
                "win_target": 0,
            }
        ]
    )
    probs = pd.Series([float("nan")])
    bets, metrics = run_backtest(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        min_ev=-1.0,
        min_edge=-1.0,
        max_spread=1.0,
        max_price=1000.0,
        max_edge_multiplier=1000.0,
    )
    assert bets.empty
    assert metrics == {}
