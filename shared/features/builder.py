from __future__ import annotations

import pandas as pd

from shared.utils.progress import log

SNAPSHOT_OFFSETS_MIN = [60, 30, 10, 5, 2, 1]


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


def build_features_from_store(store, cutoff_minutes: int) -> pd.DataFrame:
    """Construct leakage-safe features using only snapshots strictly before the cutoff."""
    snapshots = store.load_snapshots()
    if snapshots.empty:
        log("Features: no snapshots found; returning empty frame.")
        return pd.DataFrame()
    log(f"Features: loaded {len(snapshots)} snapshots.")
    snapshots = snapshots.dropna(subset=["snapshot_time", "race_start_time"])
    snapshots = snapshots[snapshots["snapshot_time"] < snapshots["race_start_time"]]
    snapshots = snapshots[snapshots["seconds_to_start"] >= cutoff_minutes * 60].copy()
    if snapshots.empty:
        log(f"Features: no snapshots at/after cutoff T-{cutoff_minutes}; returning empty frame.")
        return pd.DataFrame()
    log(f"Features: {len(snapshots)} snapshots after cutoff filter (T-{cutoff_minutes}).")

    valid_offsets = _valid_offsets(cutoff_minutes)
    features = []
    for (market_id, selection_id), runner_df in snapshots.groupby(["market_id", "selection_id"]):
        row = {
            "market_id": market_id,
            "selection_id": selection_id,
            "feature_time_cutoff": cutoff_minutes,
            "race_start_time": runner_df["race_start_time"].iloc[0],
            "venue": runner_df["venue"].iloc[0],
            "race_name": runner_df["race_name"].iloc[0],
        }
        for offset in SNAPSHOT_OFFSETS_MIN:
            if offset not in valid_offsets:
                row[f"back_price_t{offset}"] = None
                row[f"implied_prob_t{offset}"] = None
                row[f"spread_t{offset}"] = None
                row[f"volume_t{offset}"] = None
                continue
            price = _select_snapshot_value(runner_df, offset, "best_back_price")
            lay_price = _select_snapshot_value(runner_df, offset, "best_lay_price")
            total_matched = _select_snapshot_value(runner_df, offset, "total_matched")
            prob = 1.0 / price if price and price > 0 else None
            spread = (lay_price - price) if (price and lay_price) else None
            row[f"back_price_t{offset}"] = price
            row[f"implied_prob_t{offset}"] = prob
            row[f"spread_t{offset}"] = spread
            row[f"volume_t{offset}"] = total_matched

        # Movement features between offsets
        for earlier, later in [(60, 30), (30, 10), (10, 5), (5, 2)]:
            if earlier not in valid_offsets or later not in valid_offsets:
                continue
            p_earlier = row.get(f"back_price_t{earlier}")
            p_later = row.get(f"back_price_t{later}")
            if p_earlier is not None and p_later is not None:
                row[f"back_delta_{earlier}_{later}"] = p_later - p_earlier
                row[f"back_slope_{earlier}_{later}"] = (p_later - p_earlier) / (earlier - later)

        for earlier, later in [(60, 30), (30, 10), (10, 5), (5, 2)]:
            if earlier not in valid_offsets or later not in valid_offsets:
                continue
            v_earlier = row.get(f"volume_t{earlier}")
            v_later = row.get(f"volume_t{later}")
            if v_earlier is not None and v_later is not None:
                row[f"volume_rate_{earlier}_{later}"] = (v_later - v_earlier) / (earlier - later)

        features.append(row)

    feature_df = pd.DataFrame(features)
    if feature_df.empty:
        log("Features: no feature rows produced.")
        return feature_df

    # Rank and market tightness
    for offset in SNAPSHOT_OFFSETS_MIN:
        if offset not in valid_offsets:
            continue
        col = f"implied_prob_t{offset}"
        if col in feature_df:
            feature_df[f"rank_t{offset}"] = feature_df.groupby("market_id")[col].rank(
                ascending=False, method="min"
            )
            feature_df[f"market_tightness_t{offset}"] = feature_df.groupby("market_id")[col].transform(
                "sum"
            )

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
