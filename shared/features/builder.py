from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from shared.utils.progress import log

SNAPSHOT_OFFSETS_MIN = [60, 30, 10, 5, 2, 1]
MOVEMENT_PAIRS = [(60, 30), (30, 10), (10, 5), (5, 2), (2, 1)]
OFFSET_FEATURES = [
    "back_price",
    "lay_price",
    "last_traded_price",
    "back_size",
    "lay_size",
    "volume",
    "implied_prob",
    "spread",
    "mid_price",
    "spread_pct",
    "size_imbalance",
    "liquidity",
    "liquidity_weighted_price",
    "log_back_price",
    "back_tick",
    "spread_ticks",
]
TICK_LADDER = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2),
    (10.0, 20.0, 0.5),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
]
TICK_INDEX_OFFSETS = []
_tick_index_cursor = 0
for lower, upper, step in TICK_LADDER:
    TICK_INDEX_OFFSETS.append((lower, upper, step, _tick_index_cursor))
    _tick_index_cursor += int(round((upper - lower) / step))


def _safe_log(value: float | None) -> float | None:
    """Return log(value) when valid, otherwise None."""
    if value is None or value <= 0:
        return None
    return float(np.log(value))


def _tick_size(price: float | None) -> float | None:
    """Return Betfair tick size so odds moves can be compared across price ranges."""
    if price is None or price <= 0:
        return None
    for lower, upper, step in TICK_LADDER:
        if price <= upper:
            return step
    return None


def _tick_index(price: float | None) -> int | None:
    """Return a tick index along the Betfair ladder for tick-normalized deltas."""
    if price is None or price <= 0:
        return None
    for lower, upper, step, offset in TICK_INDEX_OFFSETS:
        if price <= upper:
            return int(round(offset + (price - lower) / step))
    return None


def _select_snapshot_value(df: pd.DataFrame, offset_min: int, column: str):
    """Pick the latest snapshot at or after the requested offset (still before jump)."""
    eligible = df[df["seconds_to_start"] >= offset_min * 60]
    if eligible.empty:
        return None
    row = eligible.sort_values("seconds_to_start").iloc[0]
    return row.get(column)


def _valid_offsets(cutoff_minutes: int) -> set[int]:
    """Return offsets that are not later than the chosen cutoff."""
    return {offset for offset in SNAPSHOT_OFFSETS_MIN if offset >= cutoff_minutes}


def split_features_by_race_time(
    feature_df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature frame into train/test sets using race_start_time and an ISO split date."""
    if feature_df.empty:
        return feature_df.copy(), feature_df.copy()
    split_ts = pd.to_datetime(split_date, utc=True)
    race_times = pd.to_datetime(feature_df["race_start_time"], utc=True, errors="coerce")
    train_mask = race_times < split_ts
    return feature_df[train_mask].copy(), feature_df[~train_mask].copy()


def _filter_snapshots_to_markets(snapshots: pd.DataFrame, market_ids: set[str] | None) -> pd.DataFrame:
    """Limit snapshots to selected market ids so live scoring does not rebuild full-history features."""
    if snapshots.empty or not market_ids:
        return snapshots
    if "market_id" not in snapshots.columns:
        return snapshots.iloc[0:0].copy()
    return snapshots[snapshots["market_id"].astype(str).isin(market_ids)].copy()


def build_features_from_store(
    store,
    cutoff_minutes: int,
    market_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Construct leakage-safe features using only snapshots strictly before the cutoff."""
    market_id_filter = {str(value) for value in (market_ids or []) if str(value).strip()}
    if hasattr(store, "load_snapshots_for_cutoff"):
        try:
            snapshots = store.load_snapshots_for_cutoff(
                cutoff_minutes,
                market_ids=market_id_filter or None,
            )
        except TypeError:
            # Backwards-compatible fallback for older store shims used in tests.
            snapshots = store.load_snapshots_for_cutoff(cutoff_minutes)
            snapshots = _filter_snapshots_to_markets(snapshots, market_id_filter)
    else:
        snapshots = store.load_snapshots()
        snapshots = _filter_snapshots_to_markets(snapshots, market_id_filter)
    if snapshots.empty:
        log("Features: no snapshots found; returning empty frame.")
        return pd.DataFrame()
    log(f"Features: loaded {len(snapshots)} snapshots.")
    if not hasattr(store, "load_snapshots_for_cutoff"):
        snapshots = snapshots.dropna(subset=["snapshot_time", "race_start_time"])
        snapshots = snapshots[snapshots["snapshot_time"] < snapshots["race_start_time"]]
        snapshots = snapshots[snapshots["seconds_to_start"] >= cutoff_minutes * 60].copy()
        snapshots = _filter_snapshots_to_markets(snapshots, market_id_filter)
    if snapshots.empty:
        log(f"Features: no snapshots at/after cutoff T-{cutoff_minutes}; returning empty frame.")
        return pd.DataFrame()
    log(f"Features: {len(snapshots)} snapshots after cutoff filter (T-{cutoff_minutes}).")
    if hasattr(store, "load_markets") and "market_type" not in snapshots.columns:
        markets = store.load_markets()
        if not markets.empty and "market_type" in markets.columns:
            snapshots = snapshots.merge(
                markets[["market_id", "market_type"]].drop_duplicates(subset=["market_id"]),
                on="market_id",
                how="left",
            )

    valid_offsets = _valid_offsets(cutoff_minutes)
    features = []
    for (market_id, selection_id), runner_df in snapshots.groupby(["market_id", "selection_id"], sort=False):
        runner_df_sorted = runner_df.sort_values("seconds_to_start")
        offset_rows: dict[int, pd.Series | None] = {}
        for offset in valid_offsets:
            eligible = runner_df_sorted[runner_df_sorted["seconds_to_start"] >= offset * 60]
            offset_rows[offset] = eligible.iloc[0] if not eligible.empty else None
        row = {
            "market_id": market_id,
            "selection_id": selection_id,
            "feature_time_cutoff": cutoff_minutes,
            "race_start_time": runner_df["race_start_time"].iloc[0],
            "venue": runner_df["venue"].iloc[0],
            "race_name": runner_df["race_name"].iloc[0],
            "market_type": runner_df["market_type"].iloc[0] if "market_type" in runner_df else None,
        }
        for offset in SNAPSHOT_OFFSETS_MIN:
            if offset not in valid_offsets:
                for name in OFFSET_FEATURES:
                    row[f"{name}_t{offset}"] = None
                continue
            snapshot_row = offset_rows.get(offset)
            price = snapshot_row.get("best_back_price") if snapshot_row is not None else None
            lay_price = snapshot_row.get("best_lay_price") if snapshot_row is not None else None
            last_traded = snapshot_row.get("last_traded_price") if snapshot_row is not None else None
            back_size = snapshot_row.get("best_back_size") if snapshot_row is not None else None
            lay_size = snapshot_row.get("best_lay_size") if snapshot_row is not None else None
            total_matched = snapshot_row.get("total_matched") if snapshot_row is not None else None
            prob = 1.0 / price if price and price > 0 else None
            spread = (lay_price - price) if (price and lay_price) else None
            mid_price = (price + lay_price) / 2.0 if (price and lay_price) else None
            spread_pct = (spread / mid_price) if (spread is not None and mid_price) else None
            size_imbalance = None
            if back_size is not None and lay_size is not None:
                size_total = back_size + lay_size
                if size_total > 0:
                    size_imbalance = (back_size - lay_size) / size_total
            liquidity = None
            if back_size is not None or lay_size is not None:
                liquidity = (back_size or 0.0) + (lay_size or 0.0)
            liquidity_weighted_price = None
            if back_size is not None and lay_size is not None and price and lay_price:
                weighted_total = back_size + lay_size
                if weighted_total > 0:
                    liquidity_weighted_price = (
                        (price * back_size) + (lay_price * lay_size)
                    ) / weighted_total
            tick_size = _tick_size(price)
            back_tick = _tick_index(price)
            spread_ticks = (spread / tick_size) if (spread is not None and tick_size) else None
            row[f"back_price_t{offset}"] = price
            row[f"lay_price_t{offset}"] = lay_price
            row[f"last_traded_price_t{offset}"] = last_traded
            row[f"back_size_t{offset}"] = back_size
            row[f"lay_size_t{offset}"] = lay_size
            row[f"volume_t{offset}"] = total_matched
            row[f"implied_prob_t{offset}"] = prob
            row[f"spread_t{offset}"] = spread
            row[f"mid_price_t{offset}"] = mid_price
            row[f"spread_pct_t{offset}"] = spread_pct
            row[f"size_imbalance_t{offset}"] = size_imbalance
            row[f"liquidity_t{offset}"] = liquidity
            row[f"liquidity_weighted_price_t{offset}"] = liquidity_weighted_price
            row[f"log_back_price_t{offset}"] = _safe_log(price)
            row[f"back_tick_t{offset}"] = back_tick
            row[f"spread_ticks_t{offset}"] = spread_ticks

        # Movement features between offsets
        for earlier, later in MOVEMENT_PAIRS:
            if earlier not in valid_offsets or later not in valid_offsets:
                continue
            p_earlier = row.get(f"back_price_t{earlier}")
            p_later = row.get(f"back_price_t{later}")
            if p_earlier is not None and p_later is not None:
                row[f"back_delta_{earlier}_{later}"] = p_later - p_earlier
                row[f"back_slope_{earlier}_{later}"] = (p_later - p_earlier) / (earlier - later)
                row[f"back_ratio_{earlier}_{later}"] = p_later / p_earlier if p_earlier > 0 else None
                log_earlier = _safe_log(p_earlier)
                log_later = _safe_log(p_later)
                if log_earlier is not None and log_later is not None:
                    row[f"back_log_delta_{earlier}_{later}"] = log_later - log_earlier
            tick_earlier = row.get(f"back_tick_t{earlier}")
            tick_later = row.get(f"back_tick_t{later}")
            if tick_earlier is not None and tick_later is not None:
                tick_delta = tick_later - tick_earlier
                row[f"back_tick_delta_{earlier}_{later}"] = tick_delta
                row[f"back_tick_velocity_{earlier}_{later}"] = tick_delta / (earlier - later)

            spread_earlier = row.get(f"spread_pct_t{earlier}")
            spread_later = row.get(f"spread_pct_t{later}")
            if spread_earlier is not None and spread_later is not None:
                row[f"spread_pct_delta_{earlier}_{later}"] = spread_later - spread_earlier

            imbalance_earlier = row.get(f"size_imbalance_t{earlier}")
            imbalance_later = row.get(f"size_imbalance_t{later}")
            if imbalance_earlier is not None and imbalance_later is not None:
                row[f"size_imbalance_delta_{earlier}_{later}"] = imbalance_later - imbalance_earlier

            v_earlier = row.get(f"volume_t{earlier}")
            v_later = row.get(f"volume_t{later}")
            if v_earlier is not None and v_later is not None:
                row[f"volume_delta_{earlier}_{later}"] = v_later - v_earlier
                row[f"volume_rate_{earlier}_{later}"] = (v_later - v_earlier) / (earlier - later)
            lw_earlier = row.get(f"liquidity_weighted_price_t{earlier}")
            lw_later = row.get(f"liquidity_weighted_price_t{later}")
            if lw_earlier is not None and lw_later is not None:
                row[f"liquidity_weighted_price_delta_{earlier}_{later}"] = lw_later - lw_earlier
                row[f"liquidity_weighted_price_slope_{earlier}_{later}"] = (
                    lw_later - lw_earlier
                ) / (earlier - later)

        features.append(row)

    feature_df = pd.DataFrame(features)
    if feature_df.empty:
        log("Features: no feature rows produced.")
        return feature_df

    offsets = sorted(valid_offsets, reverse=True)
    back_cols = [f"back_price_t{offset}" for offset in offsets if f"back_price_t{offset}" in feature_df]
    log_cols = [f"log_back_price_t{offset}" for offset in offsets if f"log_back_price_t{offset}" in feature_df]
    spread_cols = [f"spread_pct_t{offset}" for offset in offsets if f"spread_pct_t{offset}" in feature_df]
    if back_cols:
        feature_df["back_price_min"] = feature_df[back_cols].min(axis=1)
        feature_df["back_price_max"] = feature_df[back_cols].max(axis=1)
        feature_df["back_price_range"] = feature_df["back_price_max"] - feature_df["back_price_min"]
        feature_df["back_price_std"] = feature_df[back_cols].std(axis=1)
    if log_cols:
        feature_df["log_back_price_std"] = feature_df[log_cols].std(axis=1)
    if spread_cols:
        feature_df["spread_pct_range"] = feature_df[spread_cols].max(axis=1) - feature_df[spread_cols].min(
            axis=1
        )

    feature_df["field_size"] = feature_df.groupby("market_id")["selection_id"].transform("count")
    feature_df["field_size_log"] = np.log1p(feature_df["field_size"])

    # Rank and market tightness
    for offset in SNAPSHOT_OFFSETS_MIN:
        if offset not in valid_offsets:
            continue
        col = f"implied_prob_t{offset}"
        if col in feature_df:
            feature_df[f"rank_t{offset}"] = feature_df.groupby("market_id")[col].rank(
                ascending=False, method="min"
            )
            market_tightness = feature_df.groupby("market_id")[col].transform("sum")
            feature_df[f"market_tightness_t{offset}"] = market_tightness
            feature_df[f"overround_t{offset}"] = market_tightness
            feature_df[f"implied_prob_share_t{offset}"] = feature_df[col] / market_tightness.replace(
                0, np.nan
            )

        back_size_col = f"back_size_t{offset}"
        lay_size_col = f"lay_size_t{offset}"
        liquidity_col = f"liquidity_t{offset}"
        if back_size_col in feature_df:
            market_back_size = feature_df.groupby("market_id")[back_size_col].transform("sum")
            feature_df[f"market_back_size_t{offset}"] = market_back_size
            feature_df[f"back_size_share_t{offset}"] = feature_df[back_size_col] / market_back_size.replace(
                0, np.nan
            )
        if lay_size_col in feature_df:
            market_lay_size = feature_df.groupby("market_id")[lay_size_col].transform("sum")
            feature_df[f"market_lay_size_t{offset}"] = market_lay_size
            feature_df[f"lay_size_share_t{offset}"] = feature_df[lay_size_col] / market_lay_size.replace(
                0, np.nan
            )
        if liquidity_col in feature_df:
            market_liquidity = feature_df.groupby("market_id")[liquidity_col].transform("sum")
            feature_df[f"market_liquidity_t{offset}"] = market_liquidity
            feature_df[f"liquidity_share_t{offset}"] = feature_df[liquidity_col] / market_liquidity.replace(
                0, np.nan
            )

    for earlier, later in MOVEMENT_PAIRS:
        rank_earlier = f"rank_t{earlier}"
        rank_later = f"rank_t{later}"
        if rank_earlier in feature_df and rank_later in feature_df:
            feature_df[f"rank_delta_{earlier}_{later}"] = feature_df[rank_later] - feature_df[rank_earlier]

    results = store.load_results()
    if not results.empty:
        feature_df = feature_df.merge(
            results[["market_id", "selection_id", "win_flag"]],
            on=["market_id", "selection_id"],
            how="left",
        )
        feature_df = feature_df.rename(columns={"win_flag": "win_target"})
    else:
        feature_df["win_target"] = None

    log(f"Features: built {len(feature_df)} rows.")
    return feature_df
