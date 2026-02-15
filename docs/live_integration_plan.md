# Live Integration Plan

## Repo Recon Findings

### 1) Where historical Betfair data enters the pipeline
- `workflow/download_historic.py` downloads historic files from Betfair Historic API.
- `workflow/ingest_archive.py` ingests archive members into DuckDB tables.
- `workflow/download_data.py` captures market snapshots/results directly via `shared/betfair/client.py` for date-based runs.
- Data lands in `DuckDBStore` tables defined in `shared/storage/schemas.sql`: `markets`, `runners`, `snapshots`, `results`.

### 2) How features are generated
- `shared/features/builder.py:build_features_from_store(store, cutoff_minutes)` is the canonical feature builder.
- It expects snapshot-level inputs with at least:
  - `market_id`, `selection_id`, `snapshot_time`, `seconds_to_start`
  - `best_back_price`, `best_back_size`, `best_lay_price`, `best_lay_size`
  - `last_traded_price`, `total_matched`, `race_start_time`, `venue`, `race_name`
- It builds offset features at `T-60/30/10/5/2/1` plus movement features and market-relative ranks/shares.

### 3) Model inference interface (exact contract)
- Model loading: `shared/model/training.py:load_model_and_calibrator(model_path, calibrator_path)`.
- Inference: `shared/model/training.py:predict_probabilities(model, calibrator, feature_df) -> pd.Series`.
- Returned probabilities are aligned to `feature_df.index`.
- Saved model bundles include `feature_columns`; prediction code auto-aligns missing columns and drops extras, filling missing values with `0.0`.

### 4) How bets are currently selected
- Backtest selection logic exists in `shared/backtest/engine.py:run_backtest(...)`:
  - Computes EV from `p_hat` and price.
  - Applies filters (`min_ev`, `min_edge`, `max_price`, `max_spread`, `min_prob`, etc.).
  - Applies optional risk controls (`max_bets_per_day`, `max_exposure_per_race`).
- Live scoring exists in `workflow/score_today.py` and `workflow/cli.py cmd_score`, but no order placement is implemented yet.

### 5) Existing config patterns
- Runtime config is mostly CLI args (`argparse`) plus environment variables.
- Secrets already come from env in Betfair clients (`BETFAIR_*`).
- JSON manifests are used for incremental ingestion/download bookkeeping.

## Smallest Workable Live Integration
- Keep workflow entrypoint thin in `workflow/live_betfair.py`.
- Put reusable live logic in `shared/live/betfair_live.py`.
- Polling mode is the default using `listMarketCatalogue` + `listMarketBook`.
- Add optional stream mode later; do not block initial delivery on streaming.

## Known Schema Mismatch and Minimal Fix
- Historical training relies on multiple pre-jump offsets (`T-60..T-1`) while live polling has only "now".
- Minimal fix: maintain an in-memory per-market snapshot cache during runtime and materialize synthetic offset snapshots based on current `seconds_to_start`, then call existing `build_features_from_store` through a temporary store adapter.
- This preserves the existing model interface without retraining immediately.
