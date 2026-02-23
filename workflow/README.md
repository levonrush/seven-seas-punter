# Workflow

CLI entrypoints live here. These scripts are thin wrappers around the shared library in `shared/`.

Canonical CLI runbook: `docs/cli_playbook.md`.
Form-modeling design and leakage controls: `docs/form_intelligence_layer.md`.

## Common flows

One-command optimized run (recommended default):
```bash
punter go
```
This runs:
- `download-historic --auto --market-types ALL --workers <auto> --clean-temp` (only when needed)
- `pipeline --ingest-new --cutoff-minutes 10`
- external form auto-ingest (enabled by default when a default file is found and the table is empty)

`go` checks DuckDB freshness first:
- if snapshots are fresh, it skips `download-historic`;
- if snapshots are stale (default: older than 30 days), it refreshes historic data automatically;
- `--refresh-historic` always forces download;
- `--refresh-historic-if-stale-days N` changes the stale threshold (negative disables stale auto-refresh).
- `train/backtest/score` steps run with strategy thresholds disabled by default; pass filter flags explicitly when needed.

Full pipeline (ingest -> features -> train -> backtest -> score -> report):
```bash
punter pipeline --cutoff-minutes 10
```

Historic download (auto, resume from manifest):
```bash
punter download-historic --auto
```
Add a new sport (example):
```bash
punter download-historic --auto --sport "Greyhound Racing" --market-types ALL --clean-temp
punter pipeline --ingest-new --cutoff-minutes 10
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

Ingest licensed external form-run exports (for P1 features):
```bash
punter ingest-form --input data/external_form_runs.json --source punting_form
```
Default-on auto-ingest also runs in `features/train/backtest/score/pipeline/go` when:
- `external_runner_form_runs` is empty, and
- one of these files exists: `data/external_form_runs.parquet|csv|json|jsonl` or `data/external_form.json`.

Disable auto-ingest per command when needed:
```bash
punter go --no-external-form-ingest
punter pipeline --no-external-form-ingest
```

Repair manifests after detecting bad stream members:
```bash
punter repair-manifests --bad-list artifacts/ingest_bad_stream_members.txt
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
punter live --config config/live.yaml --live
```

Manual day sheet (pub mode):
```bash
punter pub
punter pub --budget 100 --kelly-fraction 0.25 --max-bet-pct 0.2
```

Pub-mode theory and advanced flags live in `docs/pub_domain_adaptation.md`.
Full command recipes live in `docs/cli_playbook.md`.

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
- Stream ingest pre-validates `.bz2` headers and records invalid members to `artifacts/ingest_bad_stream_members.txt`.
- Use `punter repair-manifests` to prune those bad members from download/ingest manifests for recovery reruns.
- Use `--force-ingest` to reprocess the full archive (may duplicate data).

## Logging
All scripts log elapsed time and progress via `shared/utils/progress.py`.
