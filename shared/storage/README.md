# Storage (DuckDB)

DuckDB is used as the single local store for markets, runners, snapshots, results, bets, and model runs.

## Files
- `schemas.sql`: table DDL and migrations.
- `duckdb_store.py`: thin wrapper for inserts/updates and common queries.

## Tables
- `markets`, `runners`, `snapshots`, `results`
- `bets` (simulated backtest bets)
- `model_runs` (model metadata and metrics)

Default DB path: `data/db.duckdb`.
