from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from shared.backtest.engine import run_backtest
from shared.utils.progress import log

GRID_PROFILES: Dict[str, Dict[str, List[float]]] = {
    "small": {
        "min_ev": [0.01, 0.02],
        "min_edge": [0.05, 0.1],
        "max_price": [20.0, 50.0, 100.0],
        "max_edge_mult": [3.0, 5.0],
        "max_spread": [0.5, 1.0],
    },
    "medium": {
        "min_ev": [0.0, 0.01, 0.02],
        "min_edge": [0.0, 0.05, 0.1],
        "max_price": [10.0, 20.0, 50.0, 100.0],
        "max_edge_mult": [2.0, 3.0, 5.0],
        "max_spread": [0.5, 1.0],
    },
    "large": {
        "min_ev": [0.0, 0.01, 0.02, 0.05],
        "min_edge": [0.0, 0.05, 0.1, 0.2],
        "max_price": [10.0, 20.0, 50.0, 100.0, 200.0],
        "max_edge_mult": [2.0, 3.0, 5.0, 10.0],
        "max_spread": [0.2, 0.5, 1.0],
    },
}


def build_strategy_grid(profile: str = "small") -> List[Dict[str, float]]:
    """Return a deterministic grid of strategy parameters for tuning."""
    if profile not in GRID_PROFILES:
        raise ValueError(f"Unknown grid profile: {profile}")
    grid = GRID_PROFILES[profile]
    keys = ["min_ev", "min_edge", "max_price", "max_edge_mult", "max_spread"]
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _extract_score(metrics: Dict[str, float], objective: str) -> float:
    """Return a scalar score from metrics for ranking strategy candidates."""
    if objective not in metrics:
        raise ValueError(f"Unknown strategy objective: {objective}")
    return float(metrics[objective])


def tune_strategy(
    feature_df: pd.DataFrame,
    probs: pd.Series,
    cutoff_minutes: int,
    commission: float = 0.05,
    grid_profile: str = "small",
    objective: str = "expected_roi",
    min_hit_rate: float = 0.05,
    min_bets: int = 200,
    stake: float = 1.0,
    log_every: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a grid of bet filters and return ranked results plus trade-off summary."""
    if feature_df.empty or probs.empty:
        log("Strategy tuning: empty features or probabilities; skipping.")
        return pd.DataFrame(), pd.DataFrame()

    grid = build_strategy_grid(grid_profile)
    results: List[Dict[str, float]] = []
    total = len(grid)
    log(f"Strategy tuning: evaluating {total} parameter sets (grid={grid_profile}).")
    for idx, params in enumerate(grid, start=1):
        _, metrics = run_backtest(
            feature_df=feature_df,
            probs=probs,
            cutoff_minutes=cutoff_minutes,
            commission=commission,
            min_ev=params["min_ev"],
            min_edge=params["min_edge"],
            max_spread=params["max_spread"],
            max_price=params["max_price"],
            max_edge_multiplier=params["max_edge_mult"],
            stake=stake,
            quiet=True,
        )
        if not metrics:
            continue
        if metrics.get("bets", 0) < min_bets:
            continue
        if metrics.get("hit_rate", 0.0) < min_hit_rate:
            continue
        score = _extract_score(metrics, objective)
        results.append({**params, **metrics, "score": score})
        if log_every and idx % log_every == 0:
            log(f"Strategy tuning: checked {idx}/{total} configs.")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        log("Strategy tuning: no parameter sets passed constraints.")
        return results_df, pd.DataFrame()

    results_df = results_df.sort_values("score", ascending=False).reset_index(drop=True)
    tradeoff = build_tradeoff_table(results_df, objective=objective)
    return results_df, tradeoff


def build_tradeoff_table(
    results_df: pd.DataFrame, objective: str = "expected_roi"
) -> pd.DataFrame:
    """Build a hit-rate trade-off summary from tuned strategy results."""
    if results_df.empty:
        return pd.DataFrame()
    bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.01]
    labels = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    trade = results_df.copy()
    trade["hit_rate_bin"] = pd.cut(trade["hit_rate"], bins=bins, labels=labels, right=False)
    trade = trade.sort_values(objective, ascending=False)
    trade = trade.groupby("hit_rate_bin").head(1)
    columns = [
        "hit_rate_bin",
        "min_ev",
        "min_edge",
        "max_price",
        "max_edge_mult",
        "max_spread",
        "bets",
        "hit_rate",
        "roi",
        "expected_roi",
        "profit",
        "expected_profit",
    ]
    columns = [col for col in columns if col in trade.columns]
    return trade[columns].reset_index(drop=True)
