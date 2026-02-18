from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from shared.utils.progress import log

FALLBACK_PRICE_FLOOR = 1.01
FALLBACK_PRICE_CAP = 1000.0


def quantile_to_label(quantile: float) -> str:
    """Convert a quantile to a stable label so output columns stay easy to read and join."""
    bounded = min(0.99, max(0.01, float(quantile)))
    return f"q{int(round(bounded * 100)):02d}"


def quantile_to_column(quantile: float, prefix: str = "tab_price") -> str:
    """Build a deterministic quantile column name so callers can request prices by quantile."""
    return f"{prefix}_{quantile_to_label(quantile)}"


def _safe_price_series(values: Iterable[float], index: pd.Index) -> pd.Series:
    """Clamp predicted odds into a valid decimal range so downstream EV math cannot explode."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.repeat(float(arr), len(index))
    else:
        arr = arr.reshape(-1)
    if arr.size == 1 and len(index) > 1:
        arr = np.repeat(arr.item(), len(index))
    elif arr.size != len(index):
        padded = np.full(len(index), np.nan, dtype=float)
        count = min(arr.size, len(index))
        padded[:count] = arr[:count]
        arr = padded
    clipped = np.clip(arr, FALLBACK_PRICE_FLOOR, FALLBACK_PRICE_CAP)
    clipped = np.where(np.isfinite(clipped), clipped, np.nan)
    return pd.Series(clipped, index=index, dtype=float)


def _select_betfair_price_series(feature_df: pd.DataFrame, cutoff_minutes: int) -> pd.Series:
    """Pick the live Betfair price with fallbacks so translation remains resilient to sparse offsets."""
    if feature_df.empty:
        return pd.Series(dtype=float)
    preferred = [cutoff_minutes, 60, 30, 10, 5, 2, 1]
    offsets = []
    for offset in preferred:
        if offset not in offsets:
            offsets.append(offset)

    prices = pd.Series(np.nan, index=feature_df.index, dtype=float)
    for offset in offsets:
        col = f"back_price_t{offset}"
        if col not in feature_df.columns:
            continue
        candidate = pd.to_numeric(feature_df[col], errors="coerce")
        prices = prices.where(prices.notna(), candidate)
    prices = prices.where(prices > 0)
    return prices


def _build_model_features(feature_df: pd.DataFrame, bundle: dict[str, Any] | None) -> pd.DataFrame:
    """Align feature columns for TAB translation models so missing runtime fields do not break scoring."""
    if feature_df.empty:
        return pd.DataFrame(index=feature_df.index)
    if bundle is None:
        return feature_df.select_dtypes(include=[np.number]).fillna(0.0)

    feature_columns = bundle.get("feature_columns")
    if feature_columns:
        frame = feature_df.reindex(columns=list(feature_columns), fill_value=0.0)
        return frame.fillna(0.0)
    return feature_df.select_dtypes(include=[np.number]).fillna(0.0)


def _parse_quantile_key(raw_key: Any) -> float | None:
    """Normalize user/model quantile keys to floats so model bundles can use flexible key formats."""
    if isinstance(raw_key, (int, float)) and not isinstance(raw_key, bool):
        value = float(raw_key)
    elif isinstance(raw_key, str):
        value_text = raw_key.strip().lower()
        if value_text.startswith("q"):
            value_text = value_text[1:]
        try:
            value = float(value_text)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value > 1.0:
        value = value / 100.0
    if value <= 0.0 or value >= 1.0:
        return None
    return value


def _extract_quantile_models(bundle: dict[str, Any]) -> dict[float, Any]:
    """Read quantile models from a bundle so the scorer can use direct quantile predictions when available."""
    raw_map = bundle.get("quantile_models") or bundle.get("models") or {}
    if not isinstance(raw_map, dict):
        return {}
    models: dict[float, Any] = {}
    for raw_key, model in raw_map.items():
        quantile = _parse_quantile_key(raw_key)
        if quantile is None or model is None:
            continue
        models[quantile] = model
    return models


def _predict_with_model(model: Any, model_features: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Run a translation model prediction and coerce output into safe decimal odds values."""
    if not hasattr(model, "predict"):
        raise ValueError("TAB translation model does not expose a predict method.")
    raw = model.predict(model_features)
    return _safe_price_series(raw, index=index)


def _fallback_quantile_prices(
    betfair_price: pd.Series,
    quantiles: list[float],
    haircut: float,
    spread: float,
) -> pd.DataFrame:
    """Build conservative TAB odds bounds from Betfair price when no trained translator is available."""
    bounded_haircut = min(0.95, max(0.0, float(haircut)))
    bounded_spread = min(0.95, max(0.0, float(spread)))
    median = _safe_price_series(betfair_price * (1.0 - bounded_haircut), betfair_price.index)
    output = pd.DataFrame(index=betfair_price.index)
    for quantile in quantiles:
        if quantile < 0.5:
            scale = (0.5 - quantile) / 0.4
            prices = median * (1.0 - bounded_spread * scale)
        elif quantile > 0.5:
            scale = (quantile - 0.5) / 0.4
            prices = median * (1.0 + bounded_spread * scale)
        else:
            prices = median
        output[quantile_to_column(quantile)] = _safe_price_series(prices, betfair_price.index)
    output["tab_price_source"] = "fallback_haircut"
    return output


def estimate_tab_odds_quantiles(
    feature_df: pd.DataFrame,
    cutoff_minutes: int,
    quantiles: Iterable[float],
    model_path: str | None = None,
    fallback_haircut: float = 0.08,
    fallback_spread: float = 0.10,
) -> pd.DataFrame:
    """Estimate executable TAB odds quantiles from Betfair features for conservative decisioning in pub mode."""
    quantile_values = sorted({min(0.99, max(0.01, float(q))) for q in quantiles})
    if feature_df.empty:
        cols = [quantile_to_column(q) for q in quantile_values] + ["tab_price_source"]
        return pd.DataFrame(columns=cols)

    betfair_price = _select_betfair_price_series(feature_df, cutoff_minutes=cutoff_minutes)
    result = _fallback_quantile_prices(
        betfair_price=betfair_price,
        quantiles=quantile_values,
        haircut=fallback_haircut,
        spread=fallback_spread,
    )
    if not model_path:
        return result

    path = Path(model_path)
    if not path.exists():
        log(f"TAB translation: model not found at {path}; using fallback haircut prices.")
        return result

    try:
        bundle = joblib.load(path)
    except Exception as exc:  # pragma: no cover - depends on runtime model file
        log(f"TAB translation: failed to load {path} ({exc}); using fallback haircut prices.")
        return result

    quantile_models: dict[float, Any] = {}
    direct_model = None
    if isinstance(bundle, dict):
        quantile_models = _extract_quantile_models(bundle)
        direct_model = bundle.get("model")
    elif hasattr(bundle, "predict"):
        direct_model = bundle

    model_features = _build_model_features(feature_df, bundle if isinstance(bundle, dict) else None)
    if quantile_models:
        predicted: dict[float, pd.Series] = {}
        for quantile, model in quantile_models.items():
            try:
                predicted[quantile] = _predict_with_model(model, model_features, feature_df.index)
            except Exception as exc:  # pragma: no cover - model runtime variability
                log(f"TAB translation: quantile model q={quantile:.2f} failed ({exc}); using fallback.")
        if predicted:
            known = sorted(predicted.keys())
            for quantile in quantile_values:
                column = quantile_to_column(quantile)
                if quantile in predicted:
                    result[column] = predicted[quantile]
                    continue
                if len(known) >= 2:
                    table = np.column_stack([predicted[k].to_numpy(dtype=float) for k in known])
                    interp = np.array(
                        [np.interp(quantile, known, row) for row in table],
                        dtype=float,
                    )
                    result[column] = _safe_price_series(interp, feature_df.index)
            result["tab_price_source"] = "model_quantiles"
            return result

    if direct_model is not None:
        try:
            median = _predict_with_model(direct_model, model_features, feature_df.index)
            median_column = quantile_to_column(0.5)
            result[median_column] = median
            for quantile in quantile_values:
                column = quantile_to_column(quantile)
                if quantile == 0.5:
                    continue
                if quantile < 0.5:
                    scale = (0.5 - quantile) / 0.4
                    prices = median * (1.0 - max(0.0, fallback_spread) * scale)
                else:
                    scale = (quantile - 0.5) / 0.4
                    prices = median * (1.0 + max(0.0, fallback_spread) * scale)
                result[column] = _safe_price_series(prices, feature_df.index)
            result["tab_price_source"] = "model_median"
            return result
        except Exception as exc:  # pragma: no cover - model runtime variability
            log(f"TAB translation: median model failed ({exc}); using fallback haircut prices.")
            return result

    log("TAB translation: bundle contained no usable model; using fallback haircut prices.")
    return result
