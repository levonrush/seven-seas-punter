from __future__ import annotations

import pandas as pd

from shared.backtest.engine import compute_expected_value
from shared.utils.bet_explain import annotate_preview_frame


def _select_price(feature_row: pd.Series, cutoff_minutes: int) -> float | None:
    """Pick the price at the cutoff, falling back to earlier snapshots when missing."""
    candidate = feature_row.get(f"back_price_t{cutoff_minutes}")
    if pd.notnull(candidate):
        return float(candidate)
    for offset in [60, 30, 10, 5, 2, 1]:
        val = feature_row.get(f"back_price_t{offset}")
        if pd.notnull(val):
            return float(val)
    return None


def _coalesce_market_columns(preview: pd.DataFrame) -> pd.DataFrame:
    """Prefer feature-time market metadata and fill gaps from merged market table columns."""
    merged = preview.copy()
    for col in ["venue", "race_start_time", "market_type"]:
        market_col = f"{col}_market"
        if market_col not in merged.columns:
            continue
        merged[col] = merged[col].combine_first(merged[market_col])
        merged = merged.drop(columns=[market_col])
    return merged


def build_prediction_preview(
    feature_df: pd.DataFrame,
    probs: pd.Series,
    cutoff_minutes: int,
    runners: pd.DataFrame | None = None,
    markets: pd.DataFrame | None = None,
    limit: int = 20,
    min_ev: float | None = None,
    min_edge: float | None = None,
    max_price: float | None = None,
    max_edge_multiplier: float | None = None,
    per_market_limit: int | None = 1,
    commission: float = 0.05,
    min_prob: float | None = None,
) -> pd.DataFrame:
    """Create a human-readable preview of top predictions for sanity checking."""
    if feature_df.empty or probs.empty:
        return pd.DataFrame()

    preview = feature_df.copy()
    preview["p_hat"] = probs.reindex(preview.index).values
    preview = preview.dropna(subset=["p_hat"])
    preview["price"] = preview.apply(lambda r: _select_price(r, cutoff_minutes), axis=1)
    preview = preview.dropna(subset=["price"])
    preview = preview[preview["price"] > 0]
    if preview.empty:
        return pd.DataFrame()
    if min_prob is not None:
        preview = preview[preview["p_hat"] >= min_prob]
        if preview.empty:
            return pd.DataFrame()
    preview["implied_prob"] = 1.0 / preview["price"]
    preview["expected_value"] = preview.apply(
        lambda r: compute_expected_value(r["p_hat"], r["price"], commission=commission), axis=1
    )
    preview["edge_pct"] = (preview["p_hat"] - preview["implied_prob"]) / preview["implied_prob"]
    preview["edge_multiplier"] = preview["p_hat"] / preview["implied_prob"]

    if min_ev is not None:
        preview = preview[preview["expected_value"] >= min_ev]
    if min_edge is not None:
        preview = preview[preview["edge_pct"] >= min_edge]
    if max_price is not None:
        preview = preview[preview["price"] <= max_price]
    if max_edge_multiplier is not None:
        preview = preview[preview["edge_multiplier"] <= max_edge_multiplier]
    if preview.empty:
        return pd.DataFrame()

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
        preview = _coalesce_market_columns(preview)

    if "runner_name" in preview.columns:
        preview["selection"] = preview["runner_name"].fillna(preview["selection_id"].astype(str))
    else:
        preview["selection"] = preview["selection_id"].astype(str)
    preview = preview.sort_values("expected_value", ascending=False)
    if per_market_limit:
        preview = preview.groupby("market_id").head(per_market_limit)
    preview = preview.sort_values("expected_value", ascending=False).head(limit)
    preview = annotate_preview_frame(
        preview,
        selection_col="selection",
        market_type_col="market_type",
        default_bet_type="BACK",
    )
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
        "bet_type",
        "bet_type_label",
        "bet_guidance",
        "price",
        "p_hat",
        "implied_prob",
        "edge_pct",
        "expected_value",
        "market_id",
    ]
    columns = [col for col in columns if col in preview.columns]
    return preview[columns]
