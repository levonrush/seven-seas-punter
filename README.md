# Seven Seas Punter

![Seven Seas Punter logo](images/seven-seas-logo.png)

Welcome to the Punters Club at the Seven Seas Hotel — where we turn pre-jump markets into pre-pub tall tales. The plan: scoop Betfair data, bottle it into features, train a model that can spot value, and then roll up with a cheeky shortlist of bets that *might* just pay for the next round. And yes, the entire project is vibe coded, plus a healthy dose of “blind coding” when the dev team is at the Sevens and absolutely blind.

Pipeline for Betfair Exchange AU horse racing: download markets, snapshot odds, build leakage-safe features, train calibrated models, backtest value strategies, and score daily opportunities.

## At a glance
- Build a local DuckDB of markets, runners, and snapshots.
- Train LightGBM models with time-based CV and calibrated probabilities.
- Backtest value strategies with commission and basic execution filters.
- Score “today” (or dry-run) and export a CSV of opportunities.

## Repo map
- `workflow/`: CLI entrypoints and orchestration. See `workflow/README.md`.
- `shared/`: core library (Betfair I/O, storage, features, models, backtesting). See `shared/README.md`.
- `research/`: notebooks and experiments. See `research/README.md`.
- `literature_review/`: background PDFs. See `literature_review/README.md`.
- `tests/`: unit tests. See `tests/README.md`.

## Setup
```bash
# requires Python 3.10–3.12
conda env create -f environment.yml
conda activate punter
cp .env.template .env
```
Or, if you want a minimal venv:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick start
Real historic data:
```bash
punter download-historic --auto
punter pipeline --cutoff-minutes 10
```
Dry-run (no credentials):
```bash
punter quickstart --cutoff-minutes 10
```

Updating data (append new downloads and ingest only new members):
```bash
punter download-historic --auto
punter ingest --incremental
# or
punter pipeline --cutoff-minutes 10 --ingest-new
```

## Where to look next
- Workflow usage and CLI examples: `workflow/README.md`
- Modeling details and leakage safeguards: `shared/model/README.md`
- Backtesting and strategy tuning: `shared/backtest/README.md`
- Storage schema and DuckDB notes: `shared/storage/README.md`
- Betfair credentials and historic API notes: `shared/betfair/README.md`
