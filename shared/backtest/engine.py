from __future__ import annotations

import math

import pandas as pd

from shared.utils.progress import log
from shared.utils.bet_explain import annotate_preview_frame


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
    min_ev: float | None = None,
    min_edge: float | None = None,
    max_spread: float | None = None,
    max_price: float | None = None,
    max_edge_multiplier: float | None = None,
    min_prob: float | None = None,
    stake: float = 1.0,
    max_bets_per_day: int | None = None,
    max_exposure_per_race: float | None = None,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a simple value strategy with optional filters and risk controls."""
    if feature_df.empty or probs.empty:
        if not quiet:
            log("Backtest: empty features or probabilities; skipping.")
        return pd.DataFrame(), {}

    df = feature_df.copy()
    if "win_target" in df.columns:
        missing_results = df["win_target"].isna().sum()
        if missing_results and not quiet:
            log(f"Backtest: skipping {missing_results} rows without results.")
        df = df[df["win_target"].notna()].copy()
    if df.empty:
        if not quiet:
            log("Backtest: no labeled rows available after filtering; skipping.")
        return pd.DataFrame(), {}
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
        if not isinstance(prob, (int, float)) or not math.isfinite(float(prob)):
            continue
        if max_price is not None and price > max_price:
            continue
        ev = compute_expected_value(prob, price, commission)
        if min_ev is not None and ev < min_ev:
            continue
        if min_prob is not None and prob < min_prob:
            continue
        implied_prob = 1 / price if price else None
        if implied_prob:
            edge_pct = (prob - implied_prob) / implied_prob
            edge_multiplier = prob / implied_prob
            if min_edge is not None and edge_pct < min_edge:
                continue
            if max_edge_multiplier is not None and edge_multiplier > max_edge_multiplier:
                continue
        else:
            continue
        if max_spread is not None and spread is not None and spread > max_spread:
            continue
        race_date = pd.to_datetime(row["race_start_time"]).date()
        bets.append(
            {
                "market_id": row["market_id"],
                "selection_id": row["selection_id"],
                "race_date": race_date,
                "price": price,
                "prob": prob,
                "edge_pct": edge_pct if implied_prob else None,
                "edge_multiplier": edge_multiplier if implied_prob else None,
                "expected_value": ev,
                "stake": stake,
                "race_start_time": row["race_start_time"],
                "win_flag": row.get("win_target"),
            }
        )

    bet_df = pd.DataFrame(bets)
    if bet_df.empty:
        if not quiet:
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
    bet_df["expected_profit"] = bet_df["expected_value"] * bet_df["stake"]
    stake_sum = bet_df["stake"].sum()
    bet_df["profit"] = bet_df.apply(
        lambda r: r["stake"] * (r["price"] - 1) * (1 - commission) if r.get("win_flag") else -r["stake"],
        axis=1,
    )
    bet_df["result_profit"] = bet_df["profit"]
    profit = bet_df["profit"].sum()
    expected_profit = bet_df["expected_profit"].sum()
    wins = (bet_df["win_flag"] == 1).sum() if "win_flag" in bet_df else 0
    hit_rate = wins / len(bet_df) if len(bet_df) else 0
    roi = profit / stake_sum if stake_sum else 0
    expected_roi = expected_profit / stake_sum if stake_sum else 0
    avg_pred = float(bet_df["prob"].mean()) if "prob" in bet_df else 0.0
    avg_win_rate = float(bet_df["win_flag"].mean()) if "win_flag" in bet_df else 0.0
    prob_gap = avg_pred - avg_win_rate

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
        "expected_profit": float(expected_profit),
        "turnover": float(stake_sum),
        "roi": float(roi),
        "expected_roi": float(expected_roi),
        "hit_rate": float(hit_rate),
        "avg_pred": float(avg_pred),
        "avg_win_rate": float(avg_win_rate),
        "prob_gap": float(prob_gap),
        "max_drawdown": float(max_drawdown),
        "avg_price_improvement": float(price_improvement) if price_improvement is not None else None,
    }

    if not quiet:
        log(
            "Backtest: candidates="
            f"{len(df)}, bets={len(bet_df)}, roi={metrics['roi']:.4f}, expected_roi={metrics['expected_roi']:.4f}"
        )
    return bet_df, metrics


def build_bet_preview(
    bet_df: pd.DataFrame,
    runners: pd.DataFrame | None = None,
    markets: pd.DataFrame | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Return a concise bet preview table for sanity checking model selections."""
    if bet_df.empty or "expected_value" not in bet_df.columns:
        return pd.DataFrame()

    preview = bet_df.copy()
    if runners is not None and not runners.empty:
        preview = preview.merge(
            runners[["market_id", "selection_id", "runner_name"]],
            on=["market_id", "selection_id"],
            how="left",
        )
    if markets is not None and not markets.empty:
        preview = preview.merge(
            markets[["market_id", "venue", "race_start_time", "market_type"]],
            on="market_id",
            how="left",
            suffixes=("", "_market"),
        )
        if "race_start_time_market" in preview.columns:
            preview["race_start_time"] = preview["race_start_time"].fillna(preview["race_start_time_market"])
            preview = preview.drop(columns=["race_start_time_market"])

    if "runner_name" not in preview.columns:
        preview["runner_name"] = None
    preview["selection"] = preview["runner_name"].fillna(preview["selection_id"].astype(str))

    preview = annotate_preview_frame(
        preview,
        selection_col="selection",
        market_type_col="market_type",
        bet_type_col="bet_type",
    )
    preview = preview.sort_values("expected_value", ascending=False).head(limit)
    columns = [
        "race_start_time",
        "venue",
        "market_type",
        "market_type_label",
        "selection",
        "selection_label",
        "runner_number",
        "selection_kind",
        "selection_notes",
        "price",
        "edge_pct",
        "expected_value",
        "stake",
        "bet_type",
        "bet_type_label",
        "bet_guidance",
        "result_profit",
        "market_id",
    ]
    columns = [col for col in columns if col in preview.columns]
    return preview[columns]
