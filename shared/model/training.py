from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from shared.utils.progress import log

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


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


def _bayes_optimize_lightgbm(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 20
) -> tuple[Optional[dict], Optional[float]]:
    """Run a small Bayesian hyperparameter search for LightGBM; returns best params and score."""
    if lgb is None or optuna is None:
        log("Training: LightGBM/Optuna not available; skipping tuning.")
        return None, None

    log(f"Training: Optuna tuning with {n_trials} trials.")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
        }
        model = lgb.LGBMClassifier(
            random_state=42,
            **params,
        )
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
    log(f"Training: best Optuna score={best_score:.5f}")
    return best_params, best_score


def train_and_calibrate(
    features_df: pd.DataFrame,
    cutoff_minutes: int,
    artifacts_dir: str = "artifacts",
    store=None,
) -> Tuple[Optional[str], Optional[str], Dict[str, float]]:
    """Train a tuned LightGBM model (Bayesian search) when available, with calibrated probabilities."""
    df = features_df.dropna(subset=["win_target"])
    if df.empty:
        log("Training: no labeled rows; skipping model training.")
        return None, None, {}

    X = df[_feature_columns(df)].fillna(0.0)
    y = df["win_target"].astype(int)
    log(f"Training: {len(df)} rows, {X.shape[1]} features, cutoff T-{cutoff_minutes}.")
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_params, tuning_score = _bayes_optimize_lightgbm(X_train, y_train, n_trials=25)
    if best_params and lgb is not None:
        base_model = lgb.LGBMClassifier(**best_params)
        base_model.fit(X_train, y_train)
        model_label = "LightGBM (Optuna tuned)"
        log("Training: using LightGBM (Optuna tuned).")
    else:
        base_model = HistGradientBoostingClassifier(random_state=42)
        base_model.fit(X_train, y_train)
        model_label = "HistGradientBoosting fallback"
        log("Training: using HistGradientBoosting fallback.")

    calibrator = CalibratedClassifierCV(base_estimator=base_model, method="isotonic", cv="prefit")
    calibrator.fit(X_calib, y_calib)

    probs = calibrator.predict_proba(X_calib)[:, 1]
    best_params_clean = {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (best_params or {}).items()
    }
    metrics = {
        "log_loss": float(log_loss(y_calib, probs)),
        "brier": float(brier_score_loss(y_calib, probs)),
        "cutoff_minutes": float(cutoff_minutes),
        "tuning_score": float(tuning_score) if tuning_score is not None else None,
        "model_label": model_label,
        "best_params": best_params_clean,
    }

    artifacts_path = Path(os.getenv("ARTIFACTS_DIR", artifacts_dir))
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_path / f"model_cutoff_{cutoff_minutes}.joblib"
    calibrator_path = artifacts_path / f"calibrator_cutoff_{cutoff_minutes}.joblib"
    joblib.dump(base_model, model_path)
    joblib.dump(calibrator, calibrator_path)
    log(f"Training: saved model to {model_path}")
    log(f"Training: saved calibrator to {calibrator_path}")

    if store:
        store.record_model_run(
            model_path=str(model_path),
            calibrator_path=str(calibrator_path),
            cutoff_minutes=cutoff_minutes,
            metrics=metrics,
            notes=f"{model_label} + isotonic calibration",
        )

    return str(model_path), str(calibrator_path), metrics


def load_model_and_calibrator(model_path: str, calibrator_path: Optional[str]):
    """Load persisted model artefacts for scoring and backtesting."""
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path) if calibrator_path else None
    return model, calibrator


def predict_probabilities(model, calibrator, feature_df: pd.DataFrame) -> pd.Series:
    """Generate calibrated probabilities for the provided feature frame."""
    if feature_df.empty:
        return pd.Series(dtype=float)
    X = feature_df[_feature_columns(feature_df)].fillna(0.0)
    if calibrator:
        return pd.Series(calibrator.predict_proba(X)[:, 1], index=feature_df.index)
    return pd.Series(model.predict_proba(X)[:, 1], index=feature_df.index)
