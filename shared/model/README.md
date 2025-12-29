# Modeling

This module trains LightGBM classifiers and calibrates probabilities for betting decisions.

## Training flow
- Build features at a chosen cutoff (e.g., T-10).
- Train LightGBM (Optuna tuning when available).
- Use rolling time-based CV (expanding windows) for tuning and OOF predictions.
- Calibrate probabilities with isotonic regression (fallback to Platt if needed).
- Tune a probability cutoff that maximizes Cohen's kappa on OOF predictions.
- When `market_type` is available, train separate models per market bucket (WIN/PLACE/EXOTIC/OTHER)
  with a global fallback if a bucket is too small.

## Out-of-fold (OOF) predictions
OOF predictions are generated from the time folds and used for:
- Calibration
- Training metrics
- Preview tables (so you are not looking at in-sample fits)

## Probability cutoff
- The tuned cutoff is stored as `kappa_threshold` in model run metrics.
- Backtests and scoring can use that threshold to decide whether to bet.

## Market normalization
- WIN market probabilities are normalized per race with a Plackett-Luce style transform
  so runner probabilities sum to 1 within each market.
- This normalization is applied after calibration during scoring/backtesting.

## Split date vs CV
- `--split-date` creates a holdout period (train on races before the date).
- Rolling CV happens **inside** the training period. It does not replace a forward holdout.
- If the post-split window is tiny, backtests will have few or zero bets.

## Artefacts
Models and calibrators are saved in `artifacts/` and tracked in DuckDB `model_runs`.
