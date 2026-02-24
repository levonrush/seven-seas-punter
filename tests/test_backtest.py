import pandas as pd

from shared.backtest.engine import (
    compute_expected_value,
    run_backtest,
    sanitize_probability_for_decision,
)


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


def test_backtest_market_net_commission_matches_market_level_settlement():
    """Ensure market-net commission applies once on positive net market winnings."""
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

    _, metrics = run_backtest(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        commission=0.05,
        min_ev=-1.0,
        min_edge=-1.0,
        max_spread=0.5,
        max_price=10_000.0,
        max_edge_multiplier=1_000.0,
        stake=1.0,
        commission_mode="market_net",
        apply_probability_safety=False,
    )
    assert metrics["gross_profit"] == 1.0
    assert round(metrics["commission_paid"], 6) == 0.05
    assert round(metrics["profit"], 6) == 0.95


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


def test_probability_safety_caps_longshot_and_avoids_binary_extremes():
    """Ensure rescue probability safety prevents certainty predictions on extreme longshots."""
    safe = sanitize_probability_for_decision(prob=1.0, price=300.0)
    assert 0.0 < safe < 1.0
    assert safe <= 0.10


def test_backtest_supports_optional_settlement_adjustments():
    """Ensure reduction and dead-heat style fields can adjust gross settlement when provided."""
    feature_df = pd.DataFrame(
        [
            {
                "market_id": "1.2",
                "selection_id": 7,
                "race_start_time": pd.Timestamp("2024-01-01T00:20:00Z"),
                "back_price_t10": 5.0,
                "spread_t10": 0.1,
                "win_target": 1,
                "reduction_factor": 0.2,
                "dead_heat_divisor": 2,
            }
        ]
    )
    probs = pd.Series([0.5])
    bets, metrics = run_backtest(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        commission=0.05,
        min_ev=-1.0,
        min_edge=-1.0,
        max_spread=1.0,
        max_price=1000.0,
        max_edge_multiplier=1000.0,
        stake=1.0,
        commission_mode="market_net",
        apply_probability_safety=False,
    )
    # Adjusted price = 1 + (5-1)*(1-0.2) = 4.2; dead-heat half-share => gross 1.6.
    assert len(bets) == 1
    assert round(metrics["gross_profit"], 6) == 1.6
