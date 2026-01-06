from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, cohen_kappa_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from shared.utils.progress import log
from shared.model.market_type import bucket_market_type

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


DEFAULT_CV_FOLDS = 5
DEFAULT_CV_GAP_DAYS = 1
DEFAULT_CV_STRATEGY = "expanding"
DEFAULT_MARKET_TYPE_MIN_ROWS = 5000


class ProbabilityCalibrator:
    """Calibrate raw model probabilities using isotonic or Platt scaling."""

    def __init__(self, method: str, calibrator) -> None:
        """Store the calibration method and fitted calibrator."""
        self.method = method
        self.calibrator = calibrator

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""
        raw = np.asarray(raw_probs, dtype=float).reshape(-1)
        if self.method == "isotonic":
            return self.calibrator.predict(raw)
        if self.method == "platt":
            return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        raise ValueError(f"Unknown calibration method: {self.method}")


def _require_lightgbm():
    """Ensure LightGBM is installed so training can use the required model."""
    if lgb is None:
        raise RuntimeError("LightGBM is required for training. Install it via environment.yml.")
    return lgb


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns suitable for model input, excluding identifiers and targets."""
    drop_cols = {
        "market_id",
        "selection_id",
        "win_target",
        "race_start_time",
        "race_name",
        "venue",
        "feature_time_cutoff",
    }
    return [c for c in df.columns if c not in drop_cols and df[c].dtype != "object"]


def _bucket_market_series(df: pd.DataFrame) -> pd.Series:
    """Return a bucket label per row to route market-type specific models."""
    if "market_type" not in df.columns:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index)
    return df["market_type"].apply(bucket_market_type)


def _apply_plackett_luce(probs: pd.Series, market_ids: pd.Series) -> pd.Series:
    """Normalize probabilities per market using a Plackett-Luce style transform."""
    if probs.empty:
        return probs
    clipped = probs.clip(lower=1e-9, upper=1 - 1e-9)
    weights = clipped / (1 - clipped)
    grouped = weights.groupby(market_ids).transform("sum")
    normalized = weights / grouped.replace(0, np.nan)
    return normalized.fillna(probs)


def _calibrate_probs(raw_probs: np.ndarray, calibrator, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated probabilities given raw model output and optional calibrator."""
    if calibrator is None:
        return raw_probs
    if isinstance(calibrator, ProbabilityCalibrator):
        return calibrator.calibrate(raw_probs)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(X)[:, 1]
    if hasattr(calibrator, "predict"):
        return calibrator.predict(raw_probs)
    return raw_probs


def _predict_with_model(model, calibrator, X: pd.DataFrame) -> np.ndarray:
    """Predict calibrated probabilities for a feature matrix using one model."""
    raw_probs = model.predict_proba(X)[:, 1]
    return _calibrate_probs(raw_probs, calibrator, X)


def _build_time_folds(
    df: pd.DataFrame, folds: int, gap_days: int, strategy: str
) -> list[tuple[pd.Index, pd.Index]]:
    """Create rolling time-based train/test folds using market-level race_start_time."""
    if "market_id" not in df.columns or "race_start_time" not in df.columns:
        return []

    market_times = df[["market_id", "race_start_time"]].drop_duplicates().copy()
    market_times["race_start_time"] = pd.to_datetime(
        market_times["race_start_time"], utc=True, errors="coerce"
    )
    market_times = market_times.dropna(subset=["race_start_time"]).sort_values("race_start_time")
    times = market_times["race_start_time"].unique()
    if len(times) < folds + 2:
        return []

    if strategy != "expanding":
        log(f"Training: unsupported CV strategy '{strategy}', defaulting to expanding.")
    cut_idx = np.linspace(0, len(times), folds + 2, dtype=int)
    fold_indices: list[tuple[pd.Index, pd.Index]] = []
    gap = pd.Timedelta(days=gap_days)
    for i in range(1, folds + 1):
        train_end = times[cut_idx[i] - 1]
        test_end = times[cut_idx[i + 1] - 1]
        test_start = train_end + gap
        if test_start > test_end:
            continue
        train_mask = market_times["race_start_time"] < train_end
        test_mask = (market_times["race_start_time"] >= test_start) & (
            market_times["race_start_time"] <= test_end
        )
        train_markets = market_times.loc[train_mask, "market_id"]
        test_markets = market_times.loc[test_mask, "market_id"]
        if train_markets.empty or test_markets.empty:
            continue
        train_idx = df.index[df["market_id"].isin(train_markets)]
        test_idx = df.index[df["market_id"].isin(test_markets)]
        if train_idx.empty or test_idx.empty:
            continue
        fold_indices.append((train_idx, test_idx))
    return fold_indices


def _find_best_kappa_threshold(
    probs: np.ndarray, y: pd.Series
) -> tuple[Optional[float], Optional[float]]:
    """Find the probability cutoff that maximizes Cohen's kappa."""
    if probs.size == 0 or y.empty:
        return None, None
    quantiles = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(probs, quantiles))
    if thresholds.size == 0:
        return None, None
    best_threshold = float(thresholds[0])
    best_kappa = -1.0
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        kappa = cohen_kappa_score(y, preds)
        if np.isnan(kappa):
            continue
        if kappa > best_kappa + 1e-9 or (
            abs(kappa - best_kappa) <= 1e-9 and threshold < best_threshold
        ):
            best_kappa = float(kappa)
            best_threshold = float(threshold)
    return best_threshold, best_kappa


def _build_model_from_params(best_params: Optional[dict]):
    """Instantiate a model using tuned params when possible."""
    lgb_module = _require_lightgbm()
    if best_params:
        return lgb_module.LGBMClassifier(**best_params)
    return lgb_module.LGBMClassifier(random_state=42, verbosity=-1)


def _compute_oof_raw_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    time_folds: list[tuple[pd.Index, pd.Index]],
    best_params: Optional[dict],
) -> pd.Series:
    """Generate out-of-fold raw probabilities for calibration and preview."""
    oof = pd.Series(index=X.index, dtype=float)
    for train_idx, valid_idx in time_folds:
        X_tr = X.loc[train_idx]
        y_tr = y.loc[train_idx]
        X_val = X.loc[valid_idx]
        y_val = y.loc[valid_idx]
        if y_tr.nunique() < 2 or y_val.nunique() < 2:
            continue
        model = _build_model_from_params(best_params)
        model.fit(X_tr, y_tr)
        oof.loc[valid_idx] = model.predict_proba(X_val)[:, 1]
    return oof


def _bayes_optimize_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    time_folds: Optional[list[tuple[pd.Index, pd.Index]]] = None,
    n_trials: int = 20,
) -> tuple[Optional[dict], Optional[float]]:
    """Run a small Bayesian hyperparameter search for LightGBM; returns best params and score."""
    _require_lightgbm()
    if optuna is None:
        log("Training: Optuna not available; using LightGBM default params.")
        return None, None

    log(f"Training: Optuna tuning with {n_trials} trials.")
    if time_folds:
        log(f"Training: rolling time CV with {len(time_folds)} folds.")
    else:
        log("Training: rolling time CV unavailable; using single split.")
        X_train, X_valid, y_train, y_valid, _ = _split_train_calib(
            X, y, groups=groups, test_size=0.2, random_state=42, label="Optuna"
        )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(
            random_state=42,
            **params,
        )
        if time_folds:
            fold_losses = []
            for train_idx, valid_idx in time_folds:
                X_tr = X.loc[train_idx]
                y_tr = y.loc[train_idx]
                X_val = X.loc[valid_idx]
                y_val = y.loc[valid_idx]
                if y_tr.nunique() < 2 or y_val.nunique() < 2:
                    continue
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
                preds = model.predict_proba(X_val)[:, 1]
                fold_losses.append(log_loss(y_val, preds))
            if not fold_losses:
                return float("inf")
            return float(np.mean(fold_losses))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        preds = model.predict_proba(X_valid)[:, 1]
        return log_loss(y_valid, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_score = study.best_value
    # Ensure deterministic training
    best_params["objective"] = "binary"
    best_params["random_state"] = 42
    best_params["verbosity"] = -1
    log(f"Training: best Optuna score={best_score:.5f}")
    return best_params, best_score


def _train_single_model(
    df: pd.DataFrame,
    cutoff_minutes: int,
    cv_folds: int,
    cv_gap_days: int,
    cv_strategy: str,
    label: str,
) -> tuple[Optional[object], Optional[object], Dict[str, float], Optional[pd.Series]]:
    """Train one LightGBM model + calibrator for a specific data subset."""
    if df.empty:
        log(f"Training{label}: no labeled rows; skipping.")
        return None, None, {}, None

    X = df[_feature_columns(df)].fillna(0.0)
    y = df["win_target"].astype(int)
    groups = df["market_id"] if "market_id" in df.columns else None
    log(f"Training{label}: {len(df)} rows, {X.shape[1]} features, cutoff T-{cutoff_minutes}.")

    time_folds = _build_time_folds(df, cv_folds, cv_gap_days, cv_strategy)
    if not time_folds:
        log(f"Training{label}: time CV folds unavailable; Optuna will use a single split.")

    tuned_params, tuning_score = _bayes_optimize_lightgbm(
        X,
        y,
        groups=groups,
        time_folds=time_folds,
        n_trials=25,
    )
    best_params = tuned_params or {}
    if tuned_params is None:
        best_params["objective"] = "binary"
        best_params["random_state"] = 42
        best_params["verbosity"] = -1
    if tuned_params is not None and lgb is not None:
        base_model = lgb.LGBMClassifier(**best_params)
        base_model.fit(X, y)
        model_label = "LightGBM (Optuna tuned)"
        log(f"Training{label}: using LightGBM (Optuna tuned).")
    else:
        base_model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
        base_model.fit(X, y)
        model_label = "LightGBM (default params)"
        log(f"Training{label}: using LightGBM default params (Optuna unavailable).")

    calibrator = None
    oof_predictions = None
    if time_folds:
        oof_raw = _compute_oof_raw_predictions(X, y, time_folds, best_params)
        oof_mask = oof_raw.notna()
        if oof_mask.any():
            calibrator, calibrated = _fit_probability_calibrator(
                oof_raw[oof_mask].to_numpy(), y[oof_mask]
            )
            oof_predictions = pd.Series(calibrated, index=oof_raw[oof_mask].index)
            eval_y = y[oof_mask]
            probs = calibrated
        else:
            log(f"Training{label}: OOF predictions empty; using in-sample probabilities.")
            probs = base_model.predict_proba(X)[:, 1]
            oof_predictions = pd.Series(probs, index=X.index)
            eval_y = y
            calibrator = None
    else:
        log(f"Training{label}: OOF folds unavailable; using group split for calibration.")
        X_train, X_calib, y_train, y_calib, _ = _split_train_calib(
            X, y, groups=groups, test_size=0.2, random_state=42, label=f"Training{label}"
        )
        raw_calib = base_model.predict_proba(X_calib)[:, 1]
        calibrator, calibrated = _fit_probability_calibrator(raw_calib, y_calib)
        oof_predictions = pd.Series(calibrated, index=X_calib.index)
        eval_y = y_calib
        probs = calibrated

    kappa_threshold = None
    kappa_score = None
    if probs is not None and eval_y is not None and len(eval_y):
        kappa_threshold, kappa_score = _find_best_kappa_threshold(np.asarray(probs), eval_y)
        if kappa_threshold is not None and kappa_score is not None:
            log(
                f"Training{label}: best kappa="
                f"{kappa_score:.4f} at threshold={kappa_threshold:.3f}"
            )

    best_params_clean = {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (best_params or {}).items()
    }
    metrics = {
        "log_loss": float(log_loss(eval_y, probs)),
        "brier": float(brier_score_loss(eval_y, probs)),
        "tuning_score": float(tuning_score) if tuning_score is not None else None,
        "model_label": model_label,
        "best_params": best_params_clean,
        "calibration_method": calibrator.method if isinstance(calibrator, ProbabilityCalibrator) else None,
        "train_rows": int(len(X)),
        "calib_rows": int(len(oof_predictions)) if oof_predictions is not None else 0,
        "train_markets": int(groups.nunique()) if groups is not None else None,
        "cv_folds": int(len(time_folds)) if time_folds else 0,
        "cv_gap_days": int(cv_gap_days),
        "cv_strategy": cv_strategy,
        "kappa_threshold": float(kappa_threshold) if kappa_threshold is not None else None,
        "kappa_score": float(kappa_score) if kappa_score is not None else None,
    }
    return base_model, calibrator, metrics, oof_predictions


def _train_market_type_models(
    df: pd.DataFrame,
    cutoff_minutes: int,
    cv_folds: int,
    cv_gap_days: int,
    cv_strategy: str,
    min_rows: int,
) -> tuple[dict, dict, Dict[str, float], pd.Series]:
    """Train per-market-type models with a global fallback and return combined metrics."""
    df = df.copy()
    df["market_type_bucket"] = _bucket_market_series(df)
    buckets = df["market_type_bucket"].value_counts()
    eligible = {bucket for bucket, count in buckets.items() if count >= min_rows}
    skipped = {bucket: int(count) for bucket, count in buckets.items() if count < min_rows}
    if skipped:
        log(f"Training: skipping buckets with <{min_rows} rows: {skipped}")

    models: dict[str, object] = {}
    calibrators: dict[str, object] = {}
    metrics_by_bucket: dict[str, Dict[str, float]] = {}
    oof_predictions = pd.Series(index=df.index, dtype=float)

    for bucket in sorted(eligible):
        bucket_df = df[df["market_type_bucket"] == bucket]
        label = f"[{bucket}]"
        model, calibrator, metrics, oof = _train_single_model(
            bucket_df,
            cutoff_minutes=cutoff_minutes,
            cv_folds=cv_folds,
            cv_gap_days=cv_gap_days,
            cv_strategy=cv_strategy,
            label=label,
        )
        if model is None:
            continue
        models[bucket] = model
        calibrators[bucket] = calibrator
        metrics_by_bucket[bucket] = metrics
        if oof is not None:
            oof_predictions.loc[oof.index] = oof

    fallback_model = None
    fallback_calibrator = None
    fallback_metrics: Dict[str, float] = {}
    needs_fallback = len(models) == 0 or len(eligible) < len(buckets)
    if needs_fallback:
        log("Training: fitting fallback model for uncovered market types.")
        fallback_model, fallback_calibrator, fallback_metrics, fallback_oof = _train_single_model(
            df,
            cutoff_minutes=cutoff_minutes,
            cv_folds=cv_folds,
            cv_gap_days=cv_gap_days,
            cv_strategy=cv_strategy,
            label="[ALL]",
        )
        if fallback_oof is not None:
            oof_predictions = oof_predictions.fillna(fallback_oof)

    market_type_bucket = df["market_type_bucket"]
    if "market_id" in df.columns:
        win_mask = market_type_bucket == "WIN"
        if win_mask.any():
            oof_predictions.loc[win_mask] = _apply_plackett_luce(
                oof_predictions.loc[win_mask], df.loc[win_mask, "market_id"]
            )

    oof_predictions = oof_predictions.dropna()
    metrics: Dict[str, float] = {}
    if not oof_predictions.empty:
        eval_index = oof_predictions.index.intersection(df.index)
        eval_probs = oof_predictions.loc[eval_index]
        eval_y = df.loc[eval_index, "win_target"].astype(int)
        metrics = {
            "log_loss": float(log_loss(eval_y, eval_probs)),
            "brier": float(brier_score_loss(eval_y, eval_probs)),
            "train_rows": int(len(df)),
            "train_markets": int(df["market_id"].nunique()) if "market_id" in df.columns else None,
            "calib_rows": int(len(eval_index)),
        }
        kappa_threshold, kappa_score = _find_best_kappa_threshold(
            np.asarray(eval_probs), eval_y
        )
        metrics["kappa_threshold"] = float(kappa_threshold) if kappa_threshold is not None else None
        metrics["kappa_score"] = float(kappa_score) if kappa_score is not None else None

    metrics["model_label"] = "LightGBM (market-type models)"
    metrics["market_type_models"] = metrics_by_bucket
    metrics["market_type_fallback"] = fallback_metrics
    metrics["market_type_min_rows"] = int(min_rows)

    model_bundle = {"models": models, "fallback": fallback_model}
    calibrator_bundle = {"calibrators": calibrators, "fallback": fallback_calibrator}
    return model_bundle, calibrator_bundle, metrics, oof_predictions


def _split_train_calib(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    test_size: float,
    random_state: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    """Split data into train/calib sets while avoiding market leakage when possible."""
    if groups is None or groups.nunique() < 2:
        log(f"{label}: group split unavailable; using stratified random split.")
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_calib, y_train, y_calib, None

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, calib_idx = next(splitter.split(X, y, groups))
    y_train = y.iloc[train_idx]
    y_calib = y.iloc[calib_idx]
    if y_train.nunique() < 2 or y_calib.nunique() < 2:
        log(f"{label}: group split produced single-class set; using stratified random split.")
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_calib, y_train, y_calib, None

    X_train = X.iloc[train_idx]
    X_calib = X.iloc[calib_idx]
    groups_train = groups.iloc[train_idx]
    return X_train, X_calib, y_train, y_calib, groups_train


def _fit_probability_calibrator(
    raw_probs: np.ndarray, y: pd.Series
) -> tuple[ProbabilityCalibrator, np.ndarray]:
    """Fit a calibrator on raw probabilities and return calibrated predictions."""
    try:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw_probs, y)
        calibrated = isotonic.predict(raw_probs)
        return ProbabilityCalibrator("isotonic", isotonic), calibrated
    except Exception as exc:  # pragma: no cover - rare fallback
        log(f"Training: isotonic calibration failed ({exc}); falling back to Platt scaling.")
        platt = LogisticRegression(max_iter=200, solver="lbfgs")
        platt.fit(raw_probs.reshape(-1, 1), y)
        calibrated = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return ProbabilityCalibrator("platt", platt), calibrated


def train_and_calibrate(
    features_df: pd.DataFrame,
    cutoff_minutes: int,
    artifacts_dir: str = "artifacts",
    store=None,
    split_date: Optional[str] = None,
    cv_folds: int = DEFAULT_CV_FOLDS,
    cv_gap_days: int = DEFAULT_CV_GAP_DAYS,
    cv_strategy: str = DEFAULT_CV_STRATEGY,
    run_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, float], Optional[pd.Series]]:
    """Train a tuned LightGBM model with rolling CV and return out-of-fold calibrated predictions."""
    df = features_df.dropna(subset=["win_target"])
    if df.empty:
        log("Training: no labeled rows; skipping model training.")
        return None, None, {}, None

    _require_lightgbm()
    use_market_type_models = "market_type" in df.columns
    if use_market_type_models:
        log("Training: market_type column found; using market-type specific models.")
        model_obj, calibrator_obj, metrics, oof_predictions = _train_market_type_models(
            df,
            cutoff_minutes=cutoff_minutes,
            cv_folds=cv_folds,
            cv_gap_days=cv_gap_days,
            cv_strategy=cv_strategy,
            min_rows=DEFAULT_MARKET_TYPE_MIN_ROWS,
        )
        metrics.setdefault("tuning_score", None)
        metrics.setdefault("best_params", None)
        metrics.setdefault("calibration_method", "bucketed")
    else:
        model_obj, calibrator_obj, metrics, oof_predictions = _train_single_model(
            df,
            cutoff_minutes=cutoff_minutes,
            cv_folds=cv_folds,
            cv_gap_days=cv_gap_days,
            cv_strategy=cv_strategy,
            label="",
        )

    metrics["cutoff_minutes"] = float(cutoff_minutes)
    metrics["split_date"] = split_date
    metrics["cv_folds"] = int(metrics.get("cv_folds") or cv_folds)
    metrics["cv_gap_days"] = int(cv_gap_days)
    metrics["cv_strategy"] = cv_strategy

    artifacts_path = Path(os.getenv("ARTIFACTS_DIR", artifacts_dir))
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_path / f"model_cutoff_{cutoff_minutes}.joblib"
    calibrator_path = artifacts_path / f"calibrator_cutoff_{cutoff_minutes}.joblib"
    joblib.dump(model_obj, model_path)
    joblib.dump(calibrator_obj, calibrator_path)
    log(f"Training: saved model to {model_path}")
    log(f"Training: saved calibrator to {calibrator_path}")

    if store:
        model_label = metrics.get("model_label", "LightGBM")
        if isinstance(calibrator_obj, dict):
            calibration_note = "bucketed"
        else:
            calibration_note = (
                calibrator_obj.method
                if isinstance(calibrator_obj, ProbabilityCalibrator)
                else "uncalibrated"
            )
        store.record_model_run(
            model_path=str(model_path),
            calibrator_path=str(calibrator_path),
            cutoff_minutes=cutoff_minutes,
            metrics=metrics,
            notes=f"{model_label} + {calibration_note} calibration",
            run_id=run_id,
        )

    return str(model_path), str(calibrator_path), metrics, oof_predictions


def load_model_and_calibrator(model_path: str, calibrator_path: Optional[str]):
    """Load persisted model artefacts for scoring and backtesting."""
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path) if calibrator_path else None
    return model, calibrator


def _predict_probabilities_by_market_type(
    model_bundle: dict,
    calibrator_bundle: dict | None,
    feature_df: pd.DataFrame,
    apply_plackett_luce: bool,
) -> pd.Series:
    """Route predictions through market-type specific models when available."""
    if feature_df.empty:
        return pd.Series(dtype=float)
    bucket_series = _bucket_market_series(feature_df)
    model_map = model_bundle.get("models", {})
    fallback_model = model_bundle.get("fallback") or next(iter(model_map.values()), None)
    if fallback_model is None:
        return pd.Series(dtype=float)

    calibrator_map = {}
    fallback_calibrator = None
    if isinstance(calibrator_bundle, dict):
        calibrator_map = calibrator_bundle.get("calibrators", {})
        fallback_calibrator = calibrator_bundle.get("fallback")
    else:
        fallback_calibrator = calibrator_bundle

    probs = pd.Series(index=feature_df.index, dtype=float)
    for bucket, indices in bucket_series.groupby(bucket_series).groups.items():
        model = model_map.get(bucket, fallback_model)
        calibrator = calibrator_map.get(bucket, fallback_calibrator)
        X = feature_df.loc[indices, _feature_columns(feature_df)].fillna(0.0)
        probs.loc[indices] = _predict_with_model(model, calibrator, X)

    if apply_plackett_luce and "market_id" in feature_df.columns:
        win_mask = bucket_series == "WIN"
        if win_mask.any():
            probs.loc[win_mask] = _apply_plackett_luce(
                probs.loc[win_mask], feature_df.loc[win_mask, "market_id"]
            )
    return probs


def predict_probabilities(
    model, calibrator, feature_df: pd.DataFrame, apply_plackett_luce: bool = True
) -> pd.Series:
    """Generate calibrated probabilities for the provided feature frame."""
    if feature_df.empty:
        return pd.Series(dtype=float)
    if isinstance(model, dict) and "models" in model:
        return _predict_probabilities_by_market_type(
            model, calibrator, feature_df, apply_plackett_luce=apply_plackett_luce
        )

    X = feature_df[_feature_columns(feature_df)].fillna(0.0)
    probs = _predict_with_model(model, calibrator, X)
    probs = pd.Series(probs, index=feature_df.index)
    if apply_plackett_luce and "market_id" in feature_df.columns:
        bucket_series = _bucket_market_series(feature_df)
        win_mask = bucket_series == "WIN"
        if win_mask.any():
            probs.loc[win_mask] = _apply_plackett_luce(
                probs.loc[win_mask], feature_df.loc[win_mask, "market_id"]
            )
    return probs
