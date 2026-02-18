# Workflow

CLI entrypoints live here. These scripts are thin wrappers around the shared library in `shared/`.

## Common flows

One-command optimized run (recommended default):
```bash
punter go
```
This runs:
- `download-historic --auto --market-types ALL --workers <auto> --clean-temp`
- `pipeline --ingest-new --cutoff-minutes 10`

Full pipeline (ingest -> features -> train -> backtest -> score -> report):
```bash
punter pipeline --cutoff-minutes 10
```

Historic download (auto, resume from manifest):
```bash
punter download-historic --auto
```
Show available market types before downloading:
```bash
punter download-historic --auto --show-market-types
```

Incremental ingest (only new archive members):
```bash
punter ingest --incremental
punter pipeline --cutoff-minutes 10 --ingest-new
```

Train + backtest with an explicit split:
```bash
punter train --cutoff-minutes 10 --split-date 2017-04-01 --run-id myrun
punter backtest --cutoff-minutes 10 --split-date 2017-04-01 --run-id myrun
punter report --run-id myrun
```

Live polling loop (dry-run by default in starter config):
```bash
punter live --write-config config/live.yaml
punter live --config config/live.yaml --once
```

Pub/manual betting sheet from live markets:
```bash
punter pub
```

Current operational scope (February 18, 2026):
- Live and pub flows default to ALL market types.
- Country defaults remain AU.
- Modeling quality is strongest on WIN-style markets; broader market types are supported but less validated.

## Key scripts
- `cli.py`: unified `punter` CLI (recommended).
- `download_historic.py`: Betfair Historic API downloader (manifest + list cache).
- `ingest_archive.py`: ingest stream archives (tar of `.bz2`).
- `build_features.py`: build leakage-safe feature rows.
- `train_model.py`: train + calibrate model.
- `backtest.py`: run value strategy backtest.
- `score_today.py`: score and write a CSV of opportunities.
- `live_betfair.py`: live polling/inference/execution entrypoint (`punter live` wraps this flow).

## Usage playbook
Automatic API execution (near jump):
```bash
punter live --write-config config/live.yaml
punter live --config config/live.yaml --once --dry-run
punter live --config config/live.yaml --dry-run
punter live --config config/live.yaml --live
punter live --config config/live.yaml --market-types WIN,PLACE
```

Manual day sheet (pub mode):
```bash
punter pub
punter pub --min-prob 0.12 --output artifacts/pub_sheet.csv
punter pub --budget 100 --kelly-fraction 0.25 --max-bet-pct 0.2
punter pub --market-types WIN,EXACTA
punter pub --tab-translation-model artifacts/tab_translation_cutoff_10.joblib
punter pub --tab-odds-quantile 0.15
punter pub --execution-domain betfair
```

Pub-mode theory (domain adaptation):
- `punter pub` now defaults to `--execution-domain tab`.
- Outcome probabilities still come from Betfair features (`p_hat`).
- Price decisioning is translated into a TAB odds distribution (`tab_price_q10/q50/q90`) and EV uses a conservative quantile (`--tab-odds-quantile`, default `0.10`).
- If no TAB model is available, a conservative haircut fallback is used so the CLI remains one-command operational.
- Result: shortlist and stake sizing are less sensitive to Betfair→TAB slippage risk.

Kelly fraction (brief):
- Kelly is a bankroll-growth stake formula based on probability and net odds.
- `kelly_fraction` is a risk scaler on top of full Kelly.
- Typical values: `0.25` conservative, `0.5` moderate, `1.0` full Kelly.
- When `--budget` is set, this drives `suggested_stake`.
- See `shared/backtest/README.md` for formula and assumptions.

## Split options
- `--split-date YYYY-MM-DD`: train on races before date, backtest on races on/after it.
- `--split-days N`: set split-date to the last N days of data.
- `--split-last-month`: shortcut for last 30 days.
- `--split-last-180`: shortcut for last 180 days.

## Historic download behavior
- Manifest: `data/historic_manifest.json` prevents re-downloading files.
- List cache: `data/historic_lists/` caches `DownloadListOfFiles` responses.
- Default workers: auto (2..8 based on CPU). Override with `--workers`.
- Auto-sharding: oversized windows are split via `GetAdvBasketDataSize` (`--target-basket-mb`, `--max-shard-depth`).
- Adaptive retries: transient download failures can retry with fewer workers (`--max-download-rounds`, `--no-adaptive-workers`).
- Global request throttle: shared API cap across threads (`--max-requests`, `--request-window-seconds`).
- Retry knobs: `--retries`, `--retry-wait`.
- Use `--force` to re-download everything.

`punter go` uses an auto worker preset from `shared/utils/cli_presets.py` for faster no-tune runs.

## Tuning knobs
When you need control beyond `punter go`, tune directly with command-specific flags:
- Historic download tuning: `punter download-historic --help`
- Ingest tuning: `punter ingest --help`
- Modeling/backtest/scoring tuning: `punter pipeline --help`
- Live execution tuning: `punter live --help`
- Market-type scope tuning: `--market-types` on `train`, `backtest`, `score`, `pub`, and `live`.

## Ingest behavior
- Incremental ingest uses `data/ingest_manifest.json` to skip already ingested members.
- If snapshots already exist and the ingest manifest is present, ingest auto-switches to incremental mode.
- If snapshots exist but the ingest manifest is missing, ingest seeds the manifest from the archive and skips ingest to avoid duplicates.
- First incremental run creates the manifest; keep it alongside your DuckDB for repeats.
- Use `--force-ingest` to reprocess the full archive (may duplicate data).

## Logging
All scripts log elapsed time and progress via `shared/utils/progress.py`.
