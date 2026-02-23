# External Form Ingestion

This module normalizes licensed provider exports into point-in-time run-history rows for leakage-safe feature building.

## Entry points
- `shared.form.ingest.load_external_form_rows(...)`:
  - supports `.csv`, `.parquet`, `.json`, `.jsonl` / `.ndjson`;
  - supports both row-per-run and nested `runs` payload shapes;
  - derives `snapshot_time`/`seconds_to_start` defaults when missing.

## Storage target
- `external_runner_form_runs` in DuckDB (`shared/storage/schemas.sql`).

## Feature usage
`shared/features/builder.py` consumes this table at cutoff via `load_external_runner_form_for_cutoff(...)` and computes:
- last-10 win/place and decayed recency rates,
- distance/surface/track split performance,
- class progression plus sectional/speed summaries,
- horse-jockey / horse-trainer / jockey-trainer interaction rates.
