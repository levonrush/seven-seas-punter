from __future__ import annotations

import pandas as pd

from shared.utils.progress import log


def compute_expected_value(prob: float, price: float, commission: float = 0.05) -> float:
    """Return unit-stake expected value given win probability, price, and commission."""
    return prob * (price - 1) * (1 - commission) - (1 - prob)


def _select_price(feature_row: pd.Series, cutoff_minutes: int) -> float | None:
    """Choose the price column matching the cutoff, falling back to nearest earlier snapshot."""
    candidate = feature_row.get(f"back_price_t{cutoff_minutes}")
    if pd.notnull(candidate):
        return candidate
    for offset in [60, 30, 10, 5, 2, 1]:
        val = feature_row.get(f"back_price_t{offset}")
        if pd.notnull(val):
            return val
    return None


def run_backtest(
    feature_df: pd.DataFrame,
    probs: pd.Series,
    cutoff_minutes: int,
    commission: float = 0.05,
    min_ev: float = 0.02,
    max_spread: float = 1.0,
    stake: float = 1.0,
    max_bets_per_day: int | None = None,
    max_exposure_per_race: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a simple value strategy with deterministic filters and risk controls."""
    if feature_df.empty or probs.empty:
        log("Backtest: empty features or probabilities; skipping.")
        return pd.DataFrame(), {}

    df = feature_df.copy()
    df["p_hat"] = probs
    price_col = f"back_price_t{cutoff_minutes}"
    spread_col = f"spread_t{cutoff_minutes}"

    bets = []
    for idx, row in df.iterrows():
        price = _select_price(row, cutoff_minutes)
        spread = row.get(spread_col)
        prob = row["p_hat"]
        if price is None or prob is None:
            continue
        ev = compute_expected_value(prob, price, commission)
        if ev < min_ev:
            continue
        if spread is not None and spread > max_spread:
            continue
        race_date = pd.to_datetime(row["race_start_time"]).date()
        bets.append(
            {
                "market_id": row["market_id"],
                "selection_id": row["selection_id"],
                "race_date": race_date,
                "price": price,
                "prob": prob,
                "expected_value": ev,
                "stake": stake,
                "race_start_time": row["race_start_time"],
                "win_flag": row.get("win_target"),
            }
        )

    bet_df = pd.DataFrame(bets)
    if bet_df.empty:
        log("Backtest: no bets passed filters.")
        return bet_df, {}

    # Apply risk controls
    bet_df = bet_df.sort_values(["race_date", "expected_value"], ascending=[True, False])
    if max_bets_per_day:
        bet_df["day_rank"] = bet_df.groupby("race_date").cumcount()
        bet_df = bet_df[bet_df["day_rank"] < max_bets_per_day].drop(columns=["day_rank"])
    if max_exposure_per_race:
        exposure = bet_df.groupby("market_id")["stake"].transform("sum")
        bet_df = bet_df[exposure <= max_exposure_per_race]

    # Outcome metrics
    bet_df["bet_time"] = pd.to_datetime(bet_df["race_start_time"]) - pd.to_timedelta(cutoff_minutes, unit="m")
    bet_df["bet_type"] = "BACK"
    bet_df["commission_rate"] = commission
    stake_sum = bet_df["stake"].sum()
    bet_df["profit"] = bet_df.apply(
        lambda r: r["stake"] * (r["price"] - 1) * (1 - commission) if r.get("win_flag") else -r["stake"],
        axis=1,
    )
    bet_df["result_profit"] = bet_df["profit"]
    profit = bet_df["profit"].sum()
    wins = (bet_df["win_flag"] == 1).sum() if "win_flag" in bet_df else 0
    hit_rate = wins / len(bet_df) if len(bet_df) else 0
    roi = profit / stake_sum if stake_sum else 0

    # Drawdown
    bet_df = bet_df.sort_values(["race_start_time", "market_id", "selection_id"])
    bet_df["cum_profit"] = bet_df["profit"].cumsum()
    cum_max = bet_df["cum_profit"].cummax()
    drawdowns = bet_df["cum_profit"] - cum_max
    max_drawdown = drawdowns.min() if not drawdowns.empty else 0

    # Beat later price proxy
    later_col = "back_price_t1"
    if later_col in feature_df.columns and price_col in feature_df.columns:
        joined = df[["market_id", "selection_id", price_col, later_col]].rename(
            columns={price_col: "price_cutoff", later_col: "price_late"}
        )
        bet_df = bet_df.merge(joined, on=["market_id", "selection_id"], how="left")
        bet_df["price_improvement"] = bet_df["price_cutoff"] - bet_df["price_late"]
        price_improvement = bet_df["price_improvement"].mean(skipna=True)
    else:
        price_improvement = None

    metrics = {
        "bets": len(bet_df),
        "profit": float(profit),
        "turnover": float(stake_sum),
        "roi": float(roi),
        "hit_rate": float(hit_rate),
        "max_drawdown": float(max_drawdown),
        "avg_price_improvement": float(price_improvement) if price_improvement is not None else None,
    }

    log(f"Backtest: candidates={len(df)}, bets={len(bet_df)}, roi={metrics['roi']:.4f}")
    return bet_df, metrics
