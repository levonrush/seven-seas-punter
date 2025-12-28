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
cp .env.template .env  # fill credentials or leave blank for dry-run
```
Or bootstrap from the included environment file:
```bash
conda env create -f environment.yml
conda activate punter
cp .env.template .env
```
If you prefer stdlib venv:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest black isort
```

### Betfair certs (optional for headless automation)
For Betfair AU, certificate downloads can be gated for new accounts. You can run without certs:
1. Set `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, and `BETFAIR_APP_KEY`.
2. Leave `BETFAIR_CERT_FILE` and `BETFAIR_KEY_FILE` blank.
If you later get cert access, store them locally and set absolute paths in `.env`.
If login returns a non-JSON response, set `BETFAIR_SSO_URL` to the AU endpoint:
`https://identitysso.betfair.com.au/api/login`.
If your network is flaky, bump the SSO retry knobs in `.env`:
`BETFAIR_SSO_RETRIES` and `BETFAIR_SSO_RETRY_WAIT`.

## Getting started (first run)
If you want real historic data:
```bash
punter download-historic --auto
punter pipeline --cutoff-minutes 10
```
If you just want to sanity check the stack with mock data:
```bash
punter quickstart --cutoff-minutes 10
```
Notes:
- `punter pipeline` will auto-use `data/data.tar` if it exists, and will skip ingest if it doesn't.
- The pipeline does not download historic data; run `download-historic` first when you want real data.

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
- Download Betfair historic data (automated):
  ```bash
  punter download-historic --from-date 2017-03-01 --to-date 2017-03-31 \
    --market-types WIN --countries AU --file-types M --download --output data/data.tar
  ```
  Or download everything you’ve purchased for Horse Racing (all plans, skips files already seen):
  ```bash
  punter download-historic --auto --market-types WIN --countries AU --file-types M --output data/data.tar
  ```
  Simplest default (auto + download + default output):
  ```bash
  punter download-historic --auto
  ```
  Clean the temp download folder before starting:
  ```bash
  punter download-historic --auto --download --output data/data.tar --clean-temp
  ```
  Add retries if the historic API is flaky (defaults are retries=5, retry-wait=3):
  ```bash
  punter download-historic --auto --download --output data/data.tar --retries 5 --retry-wait 3
  ```
  Defaults to half your CPU cores (capped at 8) for download workers; override if needed:
  ```bash
  punter download-historic --auto --workers 2
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
punter pipeline --cutoff-minutes 10 --split-date 2023-01-01 --dry-run  # uses data/data.tar if present; auto-prints report + predictions; use --no-report/--no-show-preds

# Or run individual subcommands:
punter ingest --archive data/data.tar
punter ingest --archive data/data.tar --workers 6  # override auto-parallelism
punter download-historic --from-date 2017-03-01 --to-date 2017-03-31 --market-types WIN --countries AU --file-types M --download --output data/data.tar
punter download-historic --auto --market-types WIN --countries AU --file-types M --download --output data/data.tar  # auto-resume
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

## Modeling & leakage safeguards
What we model (v1):
- Each row is `(market_id, selection_id)` at a chosen cutoff (e.g., T-10 minutes).
- Features are built only from snapshots strictly before the cutoff and before `race_start_time`.
- Core features include implied probability by snapshot, odds movement, spreads, volume deltas, rank-in-market, and market tightness.
- Target is `win_target` (1 if winner else 0); place targets are optional and only used when reliable.

Training and calibration:
- Default model is LightGBM tuned with Optuna (Bayesian search); falls back to HistGradientBoosting if missing.
- Tuning uses rolling time-based CV (expanding windows) over `race_start_time` with a gap day.
- Out-of-fold (OOF) predictions from the time folds are used for calibration (isotonic, fallback Platt).
- Metrics are computed on OOF predictions to avoid training-set bias.

Leakage mitigations:
- Snapshots are filtered to be strictly before the cutoff and before `race_start_time`.
- Feature rows carry a `feature_time_cutoff` marker (so downstream can enforce cutoff discipline).
- Train/test splits are time-based (no future races in the past), with a gap to reduce spillover.
- Calibration uses market-grouped splits when time folds aren’t available.
- Prediction previews after training show OOF probabilities, not in-sample fits.

## Notes
- Betfair integration uses `betfairlightweight`. Missing credentials automatically enable dry-run mock data.
- Historic data automation uses the Betfair historic API (SSO session token from your app key + credentials). Certificates are optional and mainly for headless automation.
- Historic downloads keep a manifest at `data/historic_manifest.json` so reruns only fetch new files; use `--force` to re-download.
- Model training uses LightGBM tuned with Optuna (Bayesian search) when available; it falls back to scikit-learn HistGradientBoosting if LightGBM/Optuna are missing. Tuning uses rolling time-based CV by default.
- Use `--split-date` to keep training and backtests leakage-safe (train on races before the date, test on races after).
- Prediction previews after training are based on out-of-fold probabilities from the rolling CV folds.
- Training metrics are computed from out-of-fold predictions (no holdout required).
- Preview/backtest filters default to `min_ev=0.02`, `min_edge=0.1`, `max_price=200`, `max_edge_mult=5`. Override with `--preds-*` or `--min-edge/--max-price/--max-edge-mult`.
- All functions use snake_case and include docstrings describing purpose + behavior.
- Models are saved in `artifacts/` and tracked in DuckDB `model_runs`.
