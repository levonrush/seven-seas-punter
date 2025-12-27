# Seven Seas Punter

![Seven Seas Punter logo](images/seven-seas-logo.png)

Welcome to the Punters Club at the Seven Seas Hotel — where we turn pre‑jump markets into pre‑pub tall tales. The plan: scoop Betfair data, bottle it into features, train a model that can spot value, and then roll up with a cheeky shortlist of bets that *might* just pay for the next round. And yes, the entire project is vibe coded, plus a healthy dose of “blind coding” when the dev team is at the Sevens and absolutely blind.

Pipeline for Betfair Exchange AU horse racing: download markets, snapshot odds, build leakage-safe features, train calibrated models, backtest value strategies, and score daily opportunities.

## Project layout
- `workflow/`: thin CLI entrypoints orchestrating tasks (all logic in `shared/`).
- `shared/`: reusable library for Betfair I/O, storage, features, models, and backtesting.
- `research/`: notebooks import helpers from `shared/`.
- `data/`: DuckDB file (default `data/db.duckdb`).
- `artifacts/`: saved models and calibrators.

## Setup (Conda or venv)
```bash
# requires Python 3.10–3.12
conda create -n punter python=3.12 -y
conda activate punter
pip install -e .       # installs deps + CLI entrypoint
pip install pytest black isort  # dev/test helpers
cp .env.example .env   # fill credentials or leave blank for dry-run
```
Or bootstrap from the included environment file:
```bash
conda env create -f environment.yml
conda activate punter
cp .env.example .env
```
If you prefer stdlib venv:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest black isort
```

## Make-like commands
Each step is runnable independently; all commands assume the repo root with your conda/venv activated.

- Download range (dry-run creates synthetic fixtures):
  ```bash
  python workflow/download_data.py --date 2024-01-01 --dry-run
  ```
- Ingest existing archive:
  - For Betfair historical stream tarballs (`.bz2` inside tar, e.g., BASIC/HISTORICAL): ingests definitions, snapshots, and winners.
  - For tabular tar (CSV/Parquet with markets/runners/snapshots/results):
  ```bash
  python workflow/ingest_archive.py --archive data/data.tar
  ```
- Build features at a cutoff (e.g., T-10):
  ```bash
  python workflow/build_features.py --cutoff-minutes 10
  ```
- Train + calibrate baseline:
  ```bash
  python workflow/train_model.py --cutoff-minutes 10 --split-date 2023-01-01
  ```
- Backtest value strategy:
  ```bash
  python workflow/backtest.py --cutoff-minutes 10 --split-date 2023-01-01
  ```
- Score today:
  ```bash
  python workflow/score_today.py --cutoff-minutes 10 --dry-run
  ```

## Verification / quick dry-run
The repo ships with deterministic fixtures so you can exercise the full path without credentials:
```bash
python workflow/download_data.py --date 2024-01-01 --dry-run
python workflow/build_features.py --cutoff-minutes 10
python workflow/train_model.py --cutoff-minutes 10
python workflow/backtest.py --cutoff-minutes 10
python workflow/score_today.py --cutoff-minutes 10 --dry-run
pytest
```

## Unified CLI (`punter`)
One wrapper to run individual steps or a full pipeline:
```bash
# Ingest stream tar, build features, train, backtest, and score (dry-run scoring by default)
punter pipeline --archive data/data.tar --cutoff-minutes 10 --split-date 2023-01-01 --dry-run  # auto-prints report + predictions; use --no-report/--no-show-preds

# Or run individual subcommands:
punter ingest --archive data/data.tar
punter ingest --archive data/data.tar --workers 6  # override auto-parallelism
punter features --cutoff-minutes 10
punter train --cutoff-minutes 10 --split-date 2023-01-01  # auto-prints report + predictions; use --no-report/--no-show-preds
punter backtest --cutoff-minutes 10 --split-date 2023-01-01  # auto-prints top bets; use --no-show-bets
punter score --cutoff-minutes 10 --dry-run
punter status
punter report
punter quickstart --cutoff-minutes 10
```
If you haven't installed the editable package, you can still run:
```bash
python punter.py pipeline --archive data/data.tar --cutoff-minutes 10 --split-date 2023-01-01 --dry-run  # auto-prints report + predictions; use --no-report/--no-show-preds
```

## Notes
- Betfair integration uses `betfairlightweight`. Missing credentials automatically enable dry-run mock data.
- Model training uses LightGBM tuned with Optuna (Bayesian search) when available; it falls back to scikit-learn HistGradientBoosting if LightGBM/Optuna are missing. Tuning uses rolling time-based CV by default.
- Use `--split-date` to keep training and backtests leakage-safe (train on races before the date, test on races after).
- Prediction previews after training are based on out-of-fold probabilities from the rolling CV folds.
- Training metrics are computed from out-of-fold predictions (no holdout required).
- Preview/backtest filters default to `min_ev=0.02`, `min_edge=0.1`, `max_price=200`, `max_edge_mult=5`. Override with `--preds-*` or `--min-edge/--max-price/--max-edge-mult`.
- All functions use snake_case and include docstrings describing purpose + behavior.
- Models are saved in `artifacts/` and tracked in DuckDB `model_runs`.
## Unified CLI (`punter`)
From repo root:
```bash
python punter.py pipeline --archive data/data.tar --cutoff-minutes 10 --split-date 2023-01-01 --dry-run
```
Or individual steps:
```bash
python punter.py ingest --archive data/data.tar
python punter.py features --cutoff-minutes 10
python punter.py train --cutoff-minutes 10 --split-date 2023-01-01  # auto-prints report + predictions; use --no-report/--no-show-preds
python punter.py backtest --cutoff-minutes 10 --split-date 2023-01-01  # auto-prints top bets; use --no-show-bets
python punter.py score --cutoff-minutes 10 --dry-run
python punter.py status  # show table row counts

# Quick dry-run end-to-end (no archive needed; uses mocked download)
python punter.py quickstart --cutoff-minutes 10

# Pipeline with download instead of archive
python punter.py pipeline --download-date 2024-01-01 --cutoff-minutes 10 --split-date 2023-01-01 --dry-run
# Skip parts of the pipeline if you already have them:
#   --skip-features / --skip-train / --skip-backtest / --skip-score
```
