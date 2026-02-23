# Storage (DuckDB)

DuckDB is used as the single local store for markets, runners, snapshots, results, bets, and model runs.

## Files
- `schemas.sql`: table DDL and migrations.
- `duckdb_store.py`: thin wrapper for inserts/updates and common queries.

## Tables
- `markets`, `runners`, `snapshots`, `results`
- `runner_metadata_snapshots` (point-in-time Betfair runner metadata captured pre-race for form features)
- `external_runner_form_runs` (point-in-time licensed provider run histories for last-N/sectional/split features)
- `bets` (simulated backtest bets)
- `model_runs` (model metadata and metrics)

`DuckDBStore.load_runner_metadata_completeness()` provides per-day metadata coverage monitoring for key fields (jockey/trainer/ratings/weight/draw/form).
`DuckDBStore.load_external_runner_form_completeness()` provides per-day coverage monitoring for run histories (date/result/context/sectionals/speed ratings).

Default DB path: `data/db.duckdb`.
