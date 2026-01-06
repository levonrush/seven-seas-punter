# Artifacts

This folder is generated output from training, backtesting, and scoring runs.
It is safe to delete and re-create at any time.

Typical contents:
- `model_cutoff_*.joblib`: trained LightGBM model bundles.
- `calibrator_cutoff_*.joblib`: probability calibrators.
- `pipeline_score_cutoff_*.csv`: scored opportunities from `punter pipeline` or `punter score-today`.
- `strategy_tuning_cutoff_*.csv`: grid search results for strategy params.
- `strategy_best_cutoff_*.json`: best strategy params chosen by the tuner.

How to generate:
- Train + backtest + score: `punter pipeline --cutoff-minutes 10`
- Train only: `punter train --cutoff-minutes 10`
- Backtest only: `punter backtest --cutoff-minutes 10`
- Score today: `punter score-today --cutoff-minutes 10`

Notes:
- Files in `artifacts/` are ignored by git by default.
- If you want to archive results, copy them elsewhere or commit explicitly.
