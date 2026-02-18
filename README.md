# Seven Seas Punter

![Seven Seas Punter logo](images/seven-seas-logo.png)

Welcome to the Punters Club at the Seven Seas Hotel — where we turn pre-jump markets into pre-pub tall tales. The plan: scoop Betfair data, bottle it into features, train a model that can spot value, and then roll up with a cheeky shortlist of bets that *might* just pay for the next round. And yes, the entire project is vibe coded, plus a healthy dose of “blind coding” when the dev team is at the Sevens and absolutely blind.

Pipeline for Betfair Exchange horse racing: download markets, snapshot odds, build leakage-safe features, train calibrated models, backtest value strategies, and score daily opportunities.

## At a glance
- Build a local DuckDB of markets, runners, and snapshots.
- Train LightGBM models with time-based CV and calibrated probabilities.
- Backtest value strategies with commission and basic execution filters.
- Score “today” (or dry-run) and export a CSV of opportunities.

## Current scope (February 18, 2026)
- Live and pub scoring paths now default to ALL market types unless you set a filter.
- Country defaults are still AU across most commands.
- The project is most battle-tested on horse-racing WIN-style modeling; non-WIN market coverage is available but less validated.

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
One-command optimized run:
```bash
punter go
```
Use `punter <command> --help` to tune any step (`download-historic`, `pipeline`, `live`, etc.).

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

## Live betting (polling)
Live execution reuses the trained model artefacts and runs directly from the unified CLI.

Environment variables (required for live mode):
- `BETFAIR_APP_KEY`
- `BETFAIR_USERNAME`
- `BETFAIR_PASSWORD`
- Optional cert auth:
  - `BETFAIR_CERT_PATH` (directory containing `client-2048.crt` and `client-2048.key`)
  - or `BETFAIR_CERT_FILE` + `BETFAIR_KEY_FILE`

Use the example config:
```bash
cp docs/live_betfair_config.example.yaml config.yaml
punter live --config config.yaml --once
```

Dry-run is `true` by default in the example config and logs simulated orders to `artifacts/live_decisions.csv`.
Set `dry_run: false` only after stake caps are reviewed and credentials are confirmed.

Useful CLI options:
```bash
# write a starter config
punter live --write-config config/live.yaml

# run continuously in dry-run mode
punter live --config config/live.yaml --dry-run

# run live with explicit safety overrides
punter live --config config/live.yaml --live --max-stake-per-market 5 --max-daily-exposure 50
```

## Two practical modes
Use this when you want automatic API execution near jump time:
```bash
punter live --write-config config/live.yaml
punter live --config config/live.yaml --once --dry-run
punter live --config config/live.yaml --dry-run
# switch to real order placement only after dry-run checks
punter live --config config/live.yaml --live
# optional market-type filter (default is ALL)
punter live --config config/live.yaml --market-types WIN,PLACE
```

Use this when you want a pub/day sheet for manual bets:
```bash
# easiest path: TAB-first conservative shortlist
punter pub

# write to a specific file and tighten probability filter
punter pub --min-prob 0.12 --output artifacts/pub_sheet.csv

# add bankroll sizing (fractional Kelly by default)
punter pub --budget 100 --kelly-fraction 0.25 --max-bet-pct 0.2

# optional: use a trained TAB translation model if you have one
punter pub --tab-translation-model artifacts/tab_translation_cutoff_10.joblib

# optional: change conservatism (default uses TAB q10 odds)
punter pub --tab-odds-quantile 0.15

# optional: legacy exchange-style pricing
punter pub --execution-domain betfair
```

Pub-mode output now includes:
- `execution_domain`: whether EV was computed in `tab` or `betfair` mode.
- `price`: the decision price used for EV and stake sizing.
- `betfair_price`: the source exchange price.
- `tab_price_q10/q50/q90`: TAB translated odds bands (or blank in betfair mode).
- `tab_price_source`: whether TAB pricing came from a trained model or conservative fallback.

### Why this works (theory)
- The app now treats manual TAB execution as a domain adaptation problem: your probability model is trained from Betfair states, but execution happens in a different price-formation domain (TAB).
- `punter pub` separates the decision into two parts:
  - outcome model: `p_hat` from Betfair features.
  - price translation: estimate a distribution of executable TAB odds.
- Decision rule is conservative by default:
  - estimate TAB odds quantiles (`q10/q50/q90`).
  - compute EV using a lower quantile (default `q10`), not the optimistic point quote.
  - only shortlist bets that stay positive under that conservative assumption.
- If no TAB translation model exists yet, pub mode uses a built-in haircut fallback from Betfair odds so the CLI remains one-command usable while you collect manual TAB labels.
- This matches the literature strategy: partial observability, conservative EV under price uncertainty, and progressive improvement as TAB labels accumulate.

What Kelly fraction means:
- Kelly criterion chooses stake size to maximize expected long-run log bankroll growth.
- Full Kelly for a back bet is `f* = (b*p - q)/b`, where:
  - `p` is model win probability
  - `q = 1 - p`
  - `b = (odds - 1) * (1 - commission)` is net odds
- `kelly_fraction` scales full Kelly to reduce risk: `f = kelly_fraction * max(0, f*)`.
- Example: if full Kelly suggests `8%` and `kelly_fraction=0.25`, the stake is `2%` of budget.
- This project applies Kelly only when you provide a daily `--budget` to `punter pub`/`punter score`.
- Full theory and assumptions are in `shared/backtest/README.md`.

## Where to look next
- Workflow usage and CLI examples: `workflow/README.md`
- Modeling details and leakage safeguards: `shared/model/README.md`
- Backtesting and strategy tuning: `shared/backtest/README.md`
- Storage schema and DuckDB notes: `shared/storage/README.md`
- Betfair credentials and historic API notes: `shared/betfair/README.md`
