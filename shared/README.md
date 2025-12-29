# Shared library

All production logic lives here. Workflows and notebooks should import from this package rather than re-implementing logic.

## Modules
- `shared/betfair/`: Betfair API wrappers and historic downloader helpers.
- `shared/storage/`: DuckDB schema and store helpers.
- `shared/features/`: leakage-safe feature builder.
- `shared/model/`: LightGBM training, calibration, and prediction helpers.
- `shared/backtest/`: value strategy engine and strategy tuning.
- `shared/utils/`: logging and bet-explain utilities.

Each submodule has its own README with details.
