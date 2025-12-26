# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Added automatic DuckDB schema recovery with backup on failure.
- Added elapsed-time logging across workflow steps and shared modules.
- Defaulted ingest filters to AU horse racing WIN markets to reduce data volume.
- Made ingest filtering optional (default off), with a `--filter-au-win` flag.
- Added logo asset and playful Punters Club intro to README.
- Improved elapsed-time logs to show seconds/minutes/hours as needed.
- Added ingest short-circuit when snapshots already exist, with `--force-ingest` to override.
- Added CSV fallback when parquet engine is missing during feature export.
- Added CSV fallback for CLI feature export when parquet engine is missing.
- Added “blind coding” banter to README.
- Made model calibration compatible with scikit-learn versions that expect `estimator` instead of `base_estimator`.
- Quieted LightGBM training output by default.
- Adjusted calibration fallback to avoid `cv='prefit'` in newer scikit-learn versions.
- Added expected profit/ROI tracking in backtests and a `punter report` summary command.
- Enforced strict cutoff offsets in feature building to prevent using post-cutoff signals.
- Added split-date support for train/backtest/pipeline commands to keep evaluations out-of-sample.
- Switched calibration to explicit holdout isotonic/Platt scaling with group-aware splits to reduce market leakage.
- Fixed test fixtures for expected value math and optional results handling.
