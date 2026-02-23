from __future__ import annotations

import re
from collections import defaultdict, deque
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


def _normalize_entity_name(value: object) -> str | None:
    """Canonicalize horse/jockey/trainer names so rolling form joins stay stable across feed formatting drift."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non-scalar objects are treated as invalid identifiers for entity matching.
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = re.sub(r"[^\w\s]", " ", text.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    tokens = [token for token in cleaned.split(" ") if token]
    removable_suffixes = {"NZ", "IRE", "JNR", "SNR"}
    while tokens and tokens[-1] in removable_suffixes:
        tokens = tokens[:-1]
    normalized = " ".join(tokens).strip()
    return normalized or None


def _softmax_probabilities(values: np.ndarray, scale: float) -> np.ndarray:
    """Convert rating values to race-level probabilities for Elo updates and competitiveness metrics."""
    if values.size == 0:
        return np.array([], dtype=float)
    safe_scale = max(float(scale), 1e-6)
    shifted = (values - np.max(values)) / safe_scale
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(values.shape[0], 1.0 / values.shape[0], dtype=float)
    return weights / total


def _attach_runner_lookup_features(
    feature_df: pd.DataFrame,
    store,
    market_ids: set[str] | None,
) -> pd.DataFrame:
    """Attach runner names/draws from dimensions so identity-based form features can be computed consistently."""
    if feature_df.empty or not hasattr(store, "load_runners"):
        return feature_df
    runners = store.load_runners()
    if runners.empty:
        return feature_df
    if market_ids:
        runners = runners[runners["market_id"].astype(str).isin(market_ids)]
    if runners.empty:
        return feature_df
    join_cols = [col for col in ["market_id", "selection_id", "runner_name", "stall_draw"] if col in runners.columns]
    if len(join_cols) < 2:
        return feature_df
    runners = runners[join_cols].drop_duplicates(subset=["market_id", "selection_id"], keep="last")
    return feature_df.merge(runners, on=["market_id", "selection_id"], how="left")


def _attach_runner_metadata_features(
    feature_df: pd.DataFrame,
    store,
    cutoff_minutes: int,
    market_ids: set[str] | None,
) -> pd.DataFrame:
    """Join point-in-time runner metadata and explicit missingness flags so optional coverage drift is learnable."""
    if feature_df.empty or not hasattr(store, "load_runner_metadata_for_cutoff"):
        return feature_df
    try:
        metadata = store.load_runner_metadata_for_cutoff(
            cutoff_minutes=cutoff_minutes,
            market_ids=market_ids or None,
        )
    except TypeError:
        metadata = store.load_runner_metadata_for_cutoff(cutoff_minutes)
        if market_ids and "market_id" in metadata.columns:
            metadata = metadata[metadata["market_id"].astype(str).isin(market_ids)]
    if metadata.empty:
        return feature_df

    columns = [
        "market_id",
        "selection_id",
        "runner_name",
        "jockey_name",
        "trainer_name",
        "age",
        "official_rating",
        "adjusted_rating",
        "days_since_last_run",
        "weight_value",
        "jockey_claim",
        "stall_draw",
        "form_string",
    ]
    available_columns = [column for column in columns if column in metadata.columns]
    metadata = metadata[available_columns].drop_duplicates(subset=["market_id", "selection_id"], keep="last")
    rename_map = {
        "runner_name": "runner_name_meta",
        "age": "meta_age",
        "official_rating": "meta_official_rating",
        "adjusted_rating": "meta_adjusted_rating",
        "days_since_last_run": "meta_days_since_last_run",
        "weight_value": "meta_weight_value",
        "jockey_claim": "meta_jockey_claim",
        "stall_draw": "meta_stall_draw",
        "form_string": "meta_form_string",
    }
    metadata = metadata.rename(columns=rename_map)
    enriched = feature_df.merge(metadata, on=["market_id", "selection_id"], how="left")

    if "runner_name" in enriched.columns and "runner_name_meta" in enriched.columns:
        enriched["runner_name"] = enriched["runner_name"].fillna(enriched["runner_name_meta"])
    elif "runner_name_meta" in enriched.columns and "runner_name" not in enriched.columns:
        enriched["runner_name"] = enriched["runner_name_meta"]

    numeric_cols = [
        "meta_age",
        "meta_official_rating",
        "meta_adjusted_rating",
        "meta_days_since_last_run",
        "meta_weight_value",
        "meta_jockey_claim",
        "meta_stall_draw",
    ]
    for column in numeric_cols:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    missingness_cols = [
        "jockey_name",
        "trainer_name",
        "meta_age",
        "meta_official_rating",
        "meta_adjusted_rating",
        "meta_days_since_last_run",
        "meta_weight_value",
        "meta_jockey_claim",
        "meta_stall_draw",
        "meta_form_string",
    ]
    available_missingness_cols = [column for column in missingness_cols if column in enriched.columns]
    for column in available_missingness_cols:
        enriched[f"missing_{column}"] = enriched[column].isna().astype(int)
    if available_missingness_cols:
        enriched["metadata_missing_count"] = enriched[
            [f"missing_{column}" for column in available_missingness_cols]
        ].sum(axis=1)
        enriched["metadata_any_available"] = (
            enriched["metadata_missing_count"] < len(available_missingness_cols)
        ).astype(int)
    return enriched


def _add_pre_race_elo_features(
    feature_df: pd.DataFrame,
    horse_key_col: str = "horse_entity_key",
    base_rating: float = 1500.0,
    k_factor: float = 24.0,
    rating_scale: float = 200.0,
) -> pd.DataFrame:
    """Build leakage-safe horse ratings and field-relative strength features from only prior race outcomes."""
    if feature_df.empty or horse_key_col not in feature_df.columns:
        return feature_df
    enriched = feature_df.copy()
    race_times = pd.to_datetime(enriched["race_start_time"], utc=True, errors="coerce")
    ordered = enriched.assign(_race_time=race_times).sort_values(
        by=["_race_time", "market_id", "selection_id"],
        kind="mergesort",
    )
    ratings: dict[str, float] = {}

    for (_, market_id), race_rows in ordered.groupby(["_race_time", "market_id"], sort=False):
        race_indices = race_rows.index
        horse_keys = race_rows[horse_key_col].tolist()
        pre_ratings = np.array([ratings.get(key, base_rating) for key in horse_keys], dtype=float)
        implied_probs = _softmax_probabilities(pre_ratings, rating_scale)
        if pre_ratings.size == 0:
            continue

        top_rating = float(np.max(pre_ratings))
        topk = min(3, pre_ratings.size)
        topk_mean = float(np.mean(np.sort(pre_ratings)[-topk:]))
        ranks = pd.Series(pre_ratings).rank(method="min", ascending=False).to_numpy(dtype=float)
        if pre_ratings.size > 1:
            percentile = 1.0 - ((ranks - 1.0) / (pre_ratings.size - 1.0))
            entropy_denominator = float(np.log(pre_ratings.size))
            entropy = -np.sum(implied_probs * np.log(np.clip(implied_probs, 1e-12, 1.0)))
            entropy_norm = float(entropy / entropy_denominator) if entropy_denominator > 0 else 0.0
        else:
            percentile = np.ones_like(pre_ratings)
            entropy_norm = 0.0

        enriched.loc[race_indices, "horse_elo_rating_pre"] = pre_ratings
        enriched.loc[race_indices, "horse_elo_prob_pre"] = implied_probs
        enriched.loc[race_indices, "horse_elo_rank_pre"] = ranks
        enriched.loc[race_indices, "horse_elo_percentile_pre"] = percentile
        enriched.loc[race_indices, "horse_elo_gap_to_top_pre"] = top_rating - pre_ratings
        enriched.loc[race_indices, "horse_elo_gap_to_top3_mean_pre"] = topk_mean - pre_ratings
        enriched.loc[race_indices, "field_elo_entropy_pre"] = entropy_norm
        enriched.loc[race_indices, "field_elo_top3_mean_pre"] = topk_mean

        if "win_target" not in enriched.columns:
            continue
        race_outcomes = pd.to_numeric(enriched.loc[race_indices, "win_target"], errors="coerce")
        if race_outcomes.isna().any():
            continue
        updates = race_outcomes.to_numpy(dtype=float)
        for key, rating, prob, outcome in zip(horse_keys, pre_ratings, implied_probs, updates):
            ratings[key] = float(rating + (k_factor * (outcome - prob)))

    return enriched.drop(columns=["_race_time"], errors="ignore")


def _add_pre_race_entity_rolling_rates(
    feature_df: pd.DataFrame,
    entity_col: str,
    output_prefix: str,
    window_days: int = 365,
    min_history: int = 20,
) -> pd.DataFrame:
    """Compute conservative time-windowed entity win rates from prior races only to avoid lookahead leakage."""
    if feature_df.empty or entity_col not in feature_df.columns or "win_target" not in feature_df.columns:
        return feature_df
    enriched = feature_df.copy()
    race_times = pd.to_datetime(enriched["race_start_time"], utc=True, errors="coerce")
    ordered = enriched.assign(_race_time=race_times).sort_values(
        by=["_race_time", "market_id", "selection_id"],
        kind="mergesort",
    )
    history: dict[str, deque[tuple[pd.Timestamp, int]]] = defaultdict(deque)
    lookback = pd.Timedelta(days=window_days)

    rides_col = f"{output_prefix}_prior_rides_{window_days}d"
    wins_col = f"{output_prefix}_prior_wins_{window_days}d"
    rate_col = f"{output_prefix}_win_rate_{window_days}d"
    confident_col = f"{output_prefix}_confident_history"

    for (_, market_id), race_rows in ordered.groupby(["_race_time", "market_id"], sort=False):
        race_indices = race_rows.index
        race_time = race_rows["_race_time"].iloc[0]
        if pd.isna(race_time):
            continue
        stale_before = race_time - lookback

        for idx, key in zip(race_indices, race_rows[entity_col]):
            if key is None or (isinstance(key, str) and not key.strip()) or pd.isna(key):
                enriched.at[idx, rides_col] = 0
                enriched.at[idx, wins_col] = 0
                enriched.at[idx, rate_col] = np.nan
                enriched.at[idx, confident_col] = 0
                continue
            queue = history[str(key)]
            while queue and queue[0][0] < stale_before:
                queue.popleft()
            rides = len(queue)
            wins = int(sum(win_flag for _, win_flag in queue))
            enriched.at[idx, rides_col] = rides
            enriched.at[idx, wins_col] = wins
            enriched.at[idx, confident_col] = int(rides >= min_history)
            enriched.at[idx, rate_col] = (wins / rides) if rides >= min_history and rides > 0 else np.nan

        outcomes = pd.to_numeric(enriched.loc[race_indices, "win_target"], errors="coerce")
        if outcomes.isna().any():
            continue
        for key, outcome in zip(race_rows[entity_col], outcomes):
            if key is None or (isinstance(key, str) and not key.strip()) or pd.isna(key):
                continue
            history[str(key)].append((race_time, int(float(outcome) > 0.5)))

    return enriched.drop(columns=["_race_time"], errors="ignore")


def _add_form_intelligence_features(
    feature_df: pd.DataFrame,
    results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add leakage-safe form features (Elo, PL hierarchical, and entity rates) on top of market signals."""
    if feature_df.empty:
        return feature_df
    enriched = feature_df.copy()
    if "runner_name" in enriched.columns:
        enriched["horse_entity_key"] = enriched["runner_name"].map(_normalize_entity_name)
    else:
        enriched["horse_entity_key"] = None
    selection_tokens = pd.to_numeric(enriched["selection_id"], errors="coerce").astype("Int64").astype(str)
    fallback_horse_key = "selection_" + selection_tokens.replace("<NA>", "unknown")
    enriched["horse_entity_key"] = enriched["horse_entity_key"].fillna(fallback_horse_key)
    if "jockey_name" in enriched.columns:
        enriched["jockey_entity_key"] = enriched["jockey_name"].map(_normalize_entity_name)
    else:
        enriched["jockey_entity_key"] = None
    if "trainer_name" in enriched.columns:
        enriched["trainer_entity_key"] = enriched["trainer_name"].map(_normalize_entity_name)
    else:
        enriched["trainer_entity_key"] = None

    enriched = _add_pre_race_elo_features(enriched, horse_key_col="horse_entity_key")
    enriched = _add_pre_race_pl_hierarchical_features(enriched, results_df=results_df)
    enriched = _add_pre_race_entity_rolling_rates(
        enriched,
        entity_col="jockey_entity_key",
        output_prefix="jockey",
        window_days=365,
        min_history=20,
    )
    enriched = _add_pre_race_entity_rolling_rates(
        enriched,
        entity_col="trainer_entity_key",
        output_prefix="trainer",
        window_days=365,
        min_history=20,
    )
    return enriched


def _distance_bin(distance_m: object) -> str | None:
    """Bucket distance into coarse bins so sparse external form histories remain learnable."""
    value = pd.to_numeric(distance_m, errors="coerce")
    if pd.isna(value):
        return None
    if value < 1400:
        return "sprint"
    if value < 1800:
        return "mile"
    if value < 2200:
        return "middle"
    return "staying"


def _normalized_token(value: object) -> str | None:
    """Normalize categorical context values (surface/track) to stable comparison tokens."""
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    return token


def _safe_rate(numerator: float, denominator: float, min_count: int = 1) -> float | None:
    """Return ratio when denominator is large enough, otherwise None to avoid false confidence."""
    if denominator < min_count or denominator <= 0:
        return None
    return float(numerator / denominator)


def _series_or_empty(frame: pd.DataFrame, column: str, dtype: str = "object") -> pd.Series:
    """Return a frame column when present, otherwise an empty aligned Series."""
    if column in frame.columns:
        return frame[column]
    return pd.Series(index=frame.index, dtype=dtype)


def _selection_token_series(values: pd.Series) -> pd.Series:
    """Normalize selection ids into stable string tokens so result joins stay robust across numeric dtypes."""
    tokens = pd.to_numeric(values, errors="coerce").astype("Int64").astype(str)
    return tokens.replace("<NA>", "unknown")


def _pl_topk_probabilities(weights: np.ndarray, top_k: int = 3) -> np.ndarray:
    """Return exact Plackett-Luce top-k inclusion probabilities for each runner (k<=3)."""
    if weights.size == 0:
        return np.array([], dtype=float)
    k = int(max(1, min(top_k, 3)))
    n = int(weights.size)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return np.full(n, min(k / n, 1.0), dtype=float)

    probs = np.zeros(n, dtype=float)
    win_probs = weights / total_weight
    probs += win_probs
    if k == 1 or n == 1:
        return probs

    for first in range(n):
        w_first = float(weights[first])
        p_first = w_first / total_weight
        remaining_after_first = total_weight - w_first
        if remaining_after_first <= 0:
            continue
        for second in range(n):
            if second == first:
                continue
            probs[second] += p_first * (float(weights[second]) / remaining_after_first)
    if k == 2 or n <= 2:
        return probs

    for first in range(n):
        w_first = float(weights[first])
        p_first = w_first / total_weight
        remaining_after_first = total_weight - w_first
        if remaining_after_first <= 0:
            continue
        for second in range(n):
            if second == first:
                continue
            w_second = float(weights[second])
            p_second = w_second / remaining_after_first
            remaining_after_second = remaining_after_first - w_second
            if remaining_after_second <= 0:
                continue
            prefix_prob = p_first * p_second
            for third in range(n):
                if third == first or third == second:
                    continue
                probs[third] += prefix_prob * (float(weights[third]) / remaining_after_second)
    return probs


def _extract_market_finish_orders(results_df: pd.DataFrame | None) -> dict[str, list[str]]:
    """Build ordered finish lists per market from place positions with winner fallback."""
    if results_df is None or results_df.empty or "market_id" not in results_df.columns:
        return {}
    frame = results_df.copy()
    frame["market_id"] = frame["market_id"].astype(str)
    frame["selection_token"] = _selection_token_series(frame["selection_id"])
    place_series = pd.to_numeric(_series_or_empty(frame, "place_position", dtype="float64"), errors="coerce")
    win_series = pd.to_numeric(_series_or_empty(frame, "win_flag", dtype="float64"), errors="coerce").fillna(0.0)
    frame["_place_position_numeric"] = place_series
    frame["_win_numeric"] = win_series
    finish_orders: dict[str, list[str]] = {}

    for market_id, market_rows in frame.groupby("market_id", sort=False):
        ordered: list[str] = []
        place_rows = market_rows[market_rows["_place_position_numeric"].notna()].copy()
        if not place_rows.empty:
            place_rows = place_rows.sort_values(
                ["_place_position_numeric", "selection_token"],
                ascending=[True, True],
                kind="mergesort",
            )
            for token in place_rows["selection_token"]:
                if token not in ordered:
                    ordered.append(token)
        winner_rows = market_rows[market_rows["_win_numeric"] > 0.5]
        for token in winner_rows["selection_token"]:
            if token not in ordered:
                ordered.append(token)
        if ordered:
            finish_orders[str(market_id)] = ordered
    return finish_orders


def _add_pre_race_pl_hierarchical_features(
    feature_df: pd.DataFrame,
    results_df: pd.DataFrame | None,
    top_m: int = 3,
    base_rating: float = 1500.0,
    rating_scale: float = 180.0,
    learning_rate: float = 18.0,
    l2_shrinkage: float = 0.01,
) -> pd.DataFrame:
    """Add Plackett-Luce top-m and hierarchical component ratings using only prior-race outcomes."""
    if feature_df.empty:
        return feature_df
    enriched = feature_df.copy()
    race_times = pd.to_datetime(enriched["race_start_time"], utc=True, errors="coerce")
    ordered = enriched.assign(_race_time=race_times).sort_values(
        by=["_race_time", "market_id", "selection_id"],
        kind="mergesort",
    )

    finish_orders = _extract_market_finish_orders(results_df)
    horse_component: dict[str, float] = {}
    surface_component: dict[str, float] = {}
    distance_component: dict[str, float] = {}
    track_component: dict[str, float] = {}
    jockey_component: dict[str, float] = {}
    trainer_component: dict[str, float] = {}
    component_weights = {
        "horse": 1.0,
        "surface": 0.35,
        "distance": 0.25,
        "track": 0.20,
        "jockey": 0.30,
        "trainer": 0.30,
    }

    def _runner_context_keys(race_rows: pd.DataFrame) -> dict[str, list[str | None]]:
        """Derive stable context keys for hierarchical component updates."""
        horse_keys = race_rows.get("horse_entity_key")
        if horse_keys is None:
            horse_keys = _selection_token_series(race_rows["selection_id"]).map(lambda value: f"selection_{value}")
        else:
            horse_keys = horse_keys.fillna(_selection_token_series(race_rows["selection_id"]).map(lambda value: f"selection_{value}"))
        surface_keys = _series_or_empty(race_rows, "surface").map(_normalized_token).tolist()
        distance_keys = _series_or_empty(race_rows, "distance_m", dtype="float64").map(_distance_bin).tolist()
        track_keys = _series_or_empty(race_rows, "venue").map(_normalized_token).tolist()
        jockey_keys = _series_or_empty(race_rows, "jockey_entity_key").map(_normalize_entity_name).tolist()
        trainer_keys = _series_or_empty(race_rows, "trainer_entity_key").map(_normalize_entity_name).tolist()
        return {
            "horse": horse_keys.tolist(),
            "surface": surface_keys,
            "distance": distance_keys,
            "track": track_keys,
            "jockey": jockey_keys,
            "trainer": trainer_keys,
        }

    def _effective_rating_for_index(index: int, keys: dict[str, list[str | None]]) -> float:
        """Compute effective runner rating from hierarchical components."""
        value = float(base_rating)
        horse_key = keys["horse"][index]
        value += float(horse_component.get(horse_key, 0.0))
        surface_key = keys["surface"][index]
        if surface_key is not None:
            value += float(surface_component.get(surface_key, 0.0))
        distance_key = keys["distance"][index]
        if distance_key is not None:
            value += float(distance_component.get(distance_key, 0.0))
        track_key = keys["track"][index]
        if track_key is not None:
            value += float(track_component.get(track_key, 0.0))
        jockey_key = keys["jockey"][index]
        if jockey_key is not None:
            value += float(jockey_component.get(jockey_key, 0.0))
        trainer_key = keys["trainer"][index]
        if trainer_key is not None:
            value += float(trainer_component.get(trainer_key, 0.0))
        return value

    for (_, market_id), race_rows in ordered.groupby(["_race_time", "market_id"], sort=False):
        race_indices = race_rows.index
        if len(race_indices) == 0:
            continue
        context_keys = _runner_context_keys(race_rows)
        pre_ratings = np.array(
            [_effective_rating_for_index(i, context_keys) for i in range(len(race_rows))],
            dtype=float,
        )
        win_probs = _softmax_probabilities(pre_ratings, rating_scale)
        pl_weights = np.exp((pre_ratings - np.max(pre_ratings)) / max(rating_scale, 1e-6))
        top2_probs = _pl_topk_probabilities(pl_weights, top_k=2)
        top3_probs = _pl_topk_probabilities(pl_weights, top_k=3)
        ranks = pd.Series(pre_ratings).rank(method="min", ascending=False).to_numpy(dtype=float)
        if len(race_rows) > 1:
            percentiles = 1.0 - ((ranks - 1.0) / (len(race_rows) - 1.0))
            entropy = -np.sum(win_probs * np.log(np.clip(win_probs, 1e-12, 1.0)))
            entropy_norm = float(entropy / np.log(len(race_rows)))
        else:
            percentiles = np.ones_like(pre_ratings)
            entropy_norm = 0.0

        horse_pre = np.array([float(horse_component.get(key, 0.0)) for key in context_keys["horse"]], dtype=float)
        surface_pre = np.array(
            [float(surface_component.get(key, 0.0)) if key is not None else 0.0 for key in context_keys["surface"]],
            dtype=float,
        )
        distance_pre = np.array(
            [float(distance_component.get(key, 0.0)) if key is not None else 0.0 for key in context_keys["distance"]],
            dtype=float,
        )
        track_pre = np.array(
            [float(track_component.get(key, 0.0)) if key is not None else 0.0 for key in context_keys["track"]],
            dtype=float,
        )
        jockey_pre = np.array(
            [float(jockey_component.get(key, 0.0)) if key is not None else 0.0 for key in context_keys["jockey"]],
            dtype=float,
        )
        trainer_pre = np.array(
            [float(trainer_component.get(key, 0.0)) if key is not None else 0.0 for key in context_keys["trainer"]],
            dtype=float,
        )

        enriched.loc[race_indices, "pl_eff_rating_pre"] = pre_ratings
        enriched.loc[race_indices, "pl_eff_prob_win_pre"] = win_probs
        enriched.loc[race_indices, "pl_eff_prob_top2_pre"] = top2_probs
        enriched.loc[race_indices, "pl_eff_prob_top3_pre"] = top3_probs
        enriched.loc[race_indices, "pl_eff_rank_pre"] = ranks
        enriched.loc[race_indices, "pl_eff_percentile_pre"] = percentiles
        enriched.loc[race_indices, "pl_eff_gap_to_top_pre"] = float(np.max(pre_ratings)) - pre_ratings
        enriched.loc[race_indices, "pl_field_entropy_pre"] = entropy_norm
        enriched.loc[race_indices, "pl_component_horse_pre"] = horse_pre
        enriched.loc[race_indices, "pl_component_jockey_pre"] = jockey_pre
        enriched.loc[race_indices, "pl_component_trainer_pre"] = trainer_pre
        enriched.loc[race_indices, "pl_component_context_pre"] = surface_pre + distance_pre + track_pre

        race_selection_tokens = _selection_token_series(race_rows["selection_id"])
        observed_tokens = finish_orders.get(str(market_id), [])
        observed_indices: list[int] = []
        for token in observed_tokens:
            matches = np.where(race_selection_tokens.to_numpy() == token)[0]
            if matches.size == 0:
                continue
            candidate_index = int(matches[0])
            if candidate_index not in observed_indices:
                observed_indices.append(candidate_index)
        if not observed_indices:
            continue

        delta_horse: dict[str, float] = defaultdict(float)
        delta_surface: dict[str, float] = defaultdict(float)
        delta_distance: dict[str, float] = defaultdict(float)
        delta_track: dict[str, float] = defaultdict(float)
        delta_jockey: dict[str, float] = defaultdict(float)
        delta_trainer: dict[str, float] = defaultdict(float)
        touched_horse: set[str] = set()
        touched_surface: set[str] = set()
        touched_distance: set[str] = set()
        touched_track: set[str] = set()
        touched_jockey: set[str] = set()
        touched_trainer: set[str] = set()

        remaining = set(range(len(race_rows)))
        for winner_idx in observed_indices[: max(1, top_m)]:
            candidates = sorted(remaining)
            if winner_idx not in remaining or not candidates:
                continue
            candidate_ratings = np.array(
                [_effective_rating_for_index(index, context_keys) for index in candidates],
                dtype=float,
            )
            candidate_probs = _softmax_probabilities(candidate_ratings, rating_scale)
            for local_idx, runner_idx in enumerate(candidates):
                target = 1.0 if runner_idx == winner_idx else 0.0
                gradient = target - float(candidate_probs[local_idx])
                step_delta = learning_rate * gradient
                horse_key = context_keys["horse"][runner_idx]
                if horse_key is not None:
                    delta_horse[horse_key] += step_delta * component_weights["horse"]
                    touched_horse.add(horse_key)
                surface_key = context_keys["surface"][runner_idx]
                if surface_key is not None:
                    delta_surface[surface_key] += step_delta * component_weights["surface"]
                    touched_surface.add(surface_key)
                distance_key = context_keys["distance"][runner_idx]
                if distance_key is not None:
                    delta_distance[distance_key] += step_delta * component_weights["distance"]
                    touched_distance.add(distance_key)
                track_key = context_keys["track"][runner_idx]
                if track_key is not None:
                    delta_track[track_key] += step_delta * component_weights["track"]
                    touched_track.add(track_key)
                jockey_key = context_keys["jockey"][runner_idx]
                if jockey_key is not None:
                    delta_jockey[jockey_key] += step_delta * component_weights["jockey"]
                    touched_jockey.add(jockey_key)
                trainer_key = context_keys["trainer"][runner_idx]
                if trainer_key is not None:
                    delta_trainer[trainer_key] += step_delta * component_weights["trainer"]
                    touched_trainer.add(trainer_key)
            remaining.remove(winner_idx)
            if not remaining:
                break

        for key in touched_horse:
            horse_component[key] = (horse_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_horse.get(key, 0.0)
        for key in touched_surface:
            surface_component[key] = (surface_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_surface.get(
                key, 0.0
            )
        for key in touched_distance:
            distance_component[key] = (distance_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_distance.get(
                key, 0.0
            )
        for key in touched_track:
            track_component[key] = (track_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_track.get(key, 0.0)
        for key in touched_jockey:
            jockey_component[key] = (jockey_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_jockey.get(
                key, 0.0
            )
        for key in touched_trainer:
            trainer_component[key] = (trainer_component.get(key, 0.0) * (1.0 - l2_shrinkage)) + delta_trainer.get(
                key, 0.0
            )

    return enriched.drop(columns=["_race_time"], errors="ignore")


def _compute_external_form_runner_features(
    runner_form_rows: pd.DataFrame,
    race_start_time: pd.Timestamp | None,
    current_distance_m: float | None,
    current_surface: str | None,
    current_track: str | None,
    current_class_index: float | None,
    current_jockey_name: object,
    current_trainer_name: object,
) -> dict[str, float | int | None]:
    """Compute last-N external form features with strict pre-race filtering to avoid lookahead leakage."""
    if runner_form_rows.empty:
        return {"ext_form_available": 0}
    runs = runner_form_rows.copy()
    runs["run_date"] = pd.to_datetime(runs.get("run_date"), utc=True, errors="coerce")
    if race_start_time is not None:
        runs = runs[(runs["run_date"].isna()) | (runs["run_date"] < race_start_time)].copy()
    if runs.empty:
        return {"ext_form_available": 0}

    runs["run_index"] = pd.to_numeric(runs.get("run_index"), errors="coerce")
    if runs["run_index"].notna().any():
        runs = runs.sort_values(["run_index", "run_date"], ascending=[True, False], na_position="last")
    else:
        runs = runs.sort_values(["run_date"], ascending=[False], na_position="last")
    runs_last10 = runs.head(10).copy()
    if runs_last10.empty:
        return {"ext_form_available": 0}

    finish_pos = pd.to_numeric(_series_or_empty(runs_last10, "run_finish_pos", dtype="float64"), errors="coerce")
    won_series = _series_or_empty(runs_last10, "run_won")
    if won_series.empty:
        won = (finish_pos == 1).astype(float)
    else:
        won = pd.to_numeric(won_series.astype(object), errors="coerce")
        won = won.where(~won.isna(), (finish_pos == 1).astype(float)).fillna(0.0)

    placed_series = _series_or_empty(runs_last10, "run_placed")
    if placed_series.empty:
        placed = (finish_pos <= 3).astype(float)
    else:
        placed = pd.to_numeric(placed_series.astype(object), errors="coerce")
        placed = placed.where(~placed.isna(), (finish_pos <= 3).astype(float)).fillna(0.0)

    starts = float(len(runs_last10))
    wins = float(won.sum())
    places = float(placed.sum())
    feature_row: dict[str, float | int | None] = {
        "ext_form_available": 1,
        "ext_form_runs_last10": int(starts),
        "ext_form_wins_last10": int(wins),
        "ext_form_places_last10": int(places),
        "ext_form_win_rate_last10": _safe_rate(wins, starts),
        "ext_form_place_rate_last10": _safe_rate(places, starts),
    }

    run_dates = runs_last10["run_date"]
    if race_start_time is not None and run_dates.notna().any():
        day_deltas = ((race_start_time - run_dates).dt.total_seconds() / 86_400.0).clip(lower=0)
        valid_mask = day_deltas.notna()
        if valid_mask.any():
            day_values = day_deltas[valid_mask]
            feature_row["ext_form_days_since_last_run"] = float(day_values.min())
            weights = np.exp(-np.log(2) * (day_values / 60.0))
            won_values = won.loc[day_values.index].to_numpy(dtype=float)
            placed_values = placed.loc[day_values.index].to_numpy(dtype=float)
            denom = float(weights.sum())
            if denom > 0:
                feature_row["ext_form_ewinrate_60d"] = float(np.dot(weights, won_values) / denom)
                feature_row["ext_form_eplacerate_60d"] = float(np.dot(weights, placed_values) / denom)

    current_distance_bin = _distance_bin(current_distance_m)
    run_distance_bins = _series_or_empty(runs_last10, "run_distance_m", dtype="float64").map(_distance_bin)
    if current_distance_bin is not None and not run_distance_bins.empty:
        distance_mask = run_distance_bins == current_distance_bin
        distance_starts = float(distance_mask.sum())
        distance_wins = float(won.loc[distance_mask.index][distance_mask].sum()) if distance_starts else 0.0
        feature_row["ext_form_distance_starts_last10"] = int(distance_starts)
        feature_row["ext_form_distance_win_rate_last10"] = _safe_rate(distance_wins, distance_starts, min_count=2)

    current_surface_token = _normalized_token(current_surface)
    run_surface_tokens = _series_or_empty(runs_last10, "run_surface").map(_normalized_token)
    if current_surface_token is not None and not run_surface_tokens.empty:
        surface_mask = run_surface_tokens == current_surface_token
        surface_starts = float(surface_mask.sum())
        surface_wins = float(won.loc[surface_mask.index][surface_mask].sum()) if surface_starts else 0.0
        feature_row["ext_form_surface_starts_last10"] = int(surface_starts)
        feature_row["ext_form_surface_win_rate_last10"] = _safe_rate(surface_wins, surface_starts, min_count=2)

    current_track_token = _normalized_token(current_track)
    run_track_tokens = _series_or_empty(runs_last10, "run_track").map(_normalized_token)
    if current_track_token is not None and not run_track_tokens.empty:
        track_mask = run_track_tokens == current_track_token
        track_starts = float(track_mask.sum())
        track_wins = float(won.loc[track_mask.index][track_mask].sum()) if track_starts else 0.0
        feature_row["ext_form_track_starts_last10"] = int(track_starts)
        feature_row["ext_form_track_win_rate_last10"] = _safe_rate(track_wins, track_starts, min_count=2)

    run_class_index = pd.to_numeric(_series_or_empty(runs_last10, "run_class_index", dtype="float64"), errors="coerce")
    recent_class = run_class_index.dropna()
    if current_class_index is not None and not recent_class.empty:
        latest_class = float(recent_class.iloc[0])
        feature_row["ext_form_class_delta_last"] = float(current_class_index - latest_class)
    if recent_class.head(3).notna().any():
        feature_row["ext_form_class_mean_last3"] = float(recent_class.head(3).mean())

    run_sectional = pd.to_numeric(_series_or_empty(runs_last10, "run_sectional_time", dtype="float64"), errors="coerce")
    run_speed = pd.to_numeric(_series_or_empty(runs_last10, "run_speed_rating", dtype="float64"), errors="coerce")
    recent_sectional = run_sectional.dropna()
    recent_speed = run_speed.dropna()
    if not recent_sectional.empty:
        feature_row["ext_form_last_sectional"] = float(recent_sectional.iloc[0])
        feature_row["ext_form_mean_sectional_last3"] = float(run_sectional.head(3).mean())
    if not recent_speed.empty:
        feature_row["ext_form_last_speed_rating"] = float(recent_speed.iloc[0])
        feature_row["ext_form_mean_speed_rating_last3"] = float(run_speed.head(3).mean())

    current_jockey = _normalize_entity_name(current_jockey_name)
    current_trainer = _normalize_entity_name(current_trainer_name)
    run_jockey = _series_or_empty(runs_last10, "run_jockey_name").map(_normalize_entity_name)
    run_trainer = _series_or_empty(runs_last10, "run_trainer_name").map(_normalize_entity_name)
    if current_jockey is not None and not run_jockey.empty:
        horse_jockey_mask = run_jockey == current_jockey
        starts_hj = float(horse_jockey_mask.sum())
        wins_hj = float(won.loc[horse_jockey_mask.index][horse_jockey_mask].sum()) if starts_hj else 0.0
        feature_row["ext_form_horse_jockey_starts_last10"] = int(starts_hj)
        feature_row["ext_form_horse_jockey_win_rate_last10"] = _safe_rate(wins_hj, starts_hj, min_count=2)
    if current_trainer is not None and not run_trainer.empty:
        horse_trainer_mask = run_trainer == current_trainer
        starts_ht = float(horse_trainer_mask.sum())
        wins_ht = float(won.loc[horse_trainer_mask.index][horse_trainer_mask].sum()) if starts_ht else 0.0
        feature_row["ext_form_horse_trainer_starts_last10"] = int(starts_ht)
        feature_row["ext_form_horse_trainer_win_rate_last10"] = _safe_rate(wins_ht, starts_ht, min_count=2)
    if current_jockey is not None and current_trainer is not None and not run_jockey.empty and not run_trainer.empty:
        jockey_trainer_mask = (run_jockey == current_jockey) & (run_trainer == current_trainer)
        starts_jt = float(jockey_trainer_mask.sum())
        wins_jt = float(won.loc[jockey_trainer_mask.index][jockey_trainer_mask].sum()) if starts_jt else 0.0
        feature_row["ext_form_jockey_trainer_starts_last10"] = int(starts_jt)
        feature_row["ext_form_jockey_trainer_win_rate_last10"] = _safe_rate(wins_jt, starts_jt, min_count=2)

    return feature_row


def _attach_external_form_features(
    feature_df: pd.DataFrame,
    store,
    cutoff_minutes: int,
    market_ids: set[str] | None,
) -> pd.DataFrame:
    """Join external provider history and compute leakage-safe form splits from pre-race snapshots only."""
    if feature_df.empty or not hasattr(store, "load_external_runner_form_for_cutoff"):
        return feature_df
    try:
        form_runs = store.load_external_runner_form_for_cutoff(
            cutoff_minutes=cutoff_minutes,
            market_ids=market_ids or None,
        )
    except TypeError:
        form_runs = store.load_external_runner_form_for_cutoff(cutoff_minutes)
        if market_ids and "market_id" in form_runs.columns:
            form_runs = form_runs[form_runs["market_id"].astype(str).isin(market_ids)]
    if form_runs.empty:
        return feature_df

    enriched = feature_df.copy()
    indexed_features = enriched.set_index(["market_id", "selection_id"], drop=False)
    form_feature_rows: list[dict[str, float | int | None]] = []
    for (market_id, selection_id), rows in form_runs.groupby(["market_id", "selection_id"], sort=False):
        key = (market_id, selection_id)
        if key not in indexed_features.index:
            continue
        runner_row = indexed_features.loc[key]
        if isinstance(runner_row, pd.DataFrame):
            runner_row = runner_row.iloc[0]
        race_start_time = pd.to_datetime(runner_row.get("race_start_time"), utc=True, errors="coerce")
        current_distance = pd.to_numeric(runner_row.get("distance_m"), errors="coerce")
        if pd.isna(current_distance):
            if "distance_m" in rows.columns:
                current_distance_series = pd.to_numeric(rows["distance_m"], errors="coerce").dropna()
                current_distance = (
                    float(current_distance_series.iloc[0]) if not current_distance_series.empty else None
                )
            else:
                current_distance = None
        else:
            current_distance = float(current_distance)
        current_surface = runner_row.get("surface")
        if current_surface is None and "surface" in rows.columns and rows["surface"].notna().any():
            current_surface = rows["surface"].dropna().iloc[0]
        current_track = runner_row.get("venue") or runner_row.get("track")
        if current_track is None and "track" in rows.columns and rows["track"].notna().any():
            current_track = rows["track"].dropna().iloc[0]
        current_class_index = pd.to_numeric(runner_row.get("class_index"), errors="coerce")
        if pd.isna(current_class_index):
            if "class_index" in rows.columns:
                class_values = pd.to_numeric(rows["class_index"], errors="coerce").dropna()
                current_class_index = float(class_values.iloc[0]) if not class_values.empty else None
            else:
                current_class_index = None
        else:
            current_class_index = float(current_class_index)

        feature_values = _compute_external_form_runner_features(
            rows,
            race_start_time=race_start_time if not pd.isna(race_start_time) else None,
            current_distance_m=current_distance,
            current_surface=current_surface,
            current_track=current_track,
            current_class_index=current_class_index,
            current_jockey_name=runner_row.get("jockey_name"),
            current_trainer_name=runner_row.get("trainer_name"),
        )
        feature_values["market_id"] = market_id
        feature_values["selection_id"] = selection_id
        form_feature_rows.append(feature_values)

    if not form_feature_rows:
        return enriched
    form_features = pd.DataFrame(form_feature_rows)
    return enriched.merge(form_features, on=["market_id", "selection_id"], how="left")


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

    feature_df = _attach_runner_lookup_features(feature_df, store=store, market_ids=market_id_filter or None)
    feature_df = _attach_runner_metadata_features(
        feature_df,
        store=store,
        cutoff_minutes=cutoff_minutes,
        market_ids=market_id_filter or None,
    )
    feature_df = _attach_external_form_features(
        feature_df,
        store=store,
        cutoff_minutes=cutoff_minutes,
        market_ids=market_id_filter or None,
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

    feature_df = _add_form_intelligence_features(feature_df, results_df=results)

    log(f"Features: built {len(feature_df)} rows.")
    return feature_df
