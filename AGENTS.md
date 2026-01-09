# AGENTS

This file is guidance for automated agents working in this repository.

## Repo overview
- CLI entrypoint: `punter` (defined in `workflow/cli.py`).
- Core logic lives in `shared/` (do not add logic to `workflow/`).
- Data is stored in DuckDB (`data/db.duckdb`) and tar archives (`data/*.tar`).

## Environment
- Use Conda (`environment.yml`).
- Run commands via `punter` from repo root.

## Data safety
- Do not commit large data files (e.g., `data/*.tar`, `data/*.duckdb`, `artifacts/*.csv`).
- Keep downloads and derived outputs in `data/` and `artifacts/` only.

## Coding standards
- snake_case everywhere.
- Every function must have a docstring explaining why it exists + what it does.
- Keep CLI scripts thin; put logic in `shared/`.

## Logging and progress
- Prefer existing progress/log helpers in `shared/utils/`.
- Keep logs concise but informative (timestamps/durations already expected).

## Tests
- Run `pytest` before finalizing if you changed logic.
- Minimal tests exist in `tests/`.

## Changelog
- Update `CHANGELOG.md` for any code changes.

## Common commands
- Download data: `punter download-historic --auto`.
- Ingest: `punter ingest --archive data/data.tar`.
- Full pipeline: `punter pipeline --cutoff-minutes 10`.
- Train only: `punter train --cutoff-minutes 10`.
- Backtest: `punter backtest --cutoff-minutes 10`.
- Report: `punter report`.
