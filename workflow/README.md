# Workflow

CLI entrypoints live here. These scripts are thin wrappers around the shared library in `shared/`.

## Common flows

Full pipeline (ingest -> features -> train -> backtest -> score -> report):
```bash
punter pipeline --cutoff-minutes 10
```

Historic download (auto, resume from manifest):
```bash
punter download-historic --auto
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

## Key scripts
- `cli.py`: unified `punter` CLI (recommended).
- `download_historic.py`: Betfair Historic API downloader (manifest + list cache).
- `ingest_archive.py`: ingest stream archives (tar of `.bz2`).
- `build_features.py`: build leakage-safe feature rows.
- `train_model.py`: train + calibrate model.
- `backtest.py`: run value strategy backtest.
- `score_today.py`: score and write a CSV of opportunities.

## Split options
- `--split-date YYYY-MM-DD`: train on races before date, backtest on races on/after it.
- `--split-days N`: set split-date to the last N days of data.
- `--split-last-month`: shortcut for last 30 days.
- `--split-last-180`: shortcut for last 180 days.

## Historic download behavior
- Manifest: `data/historic_manifest.json` prevents re-downloading files.
- List cache: `data/historic_lists/` caches `DownloadListOfFiles` responses.
- Default workers: 1 (to avoid API throttling). Override with `--workers`.
- Retry knobs: `--retries`, `--retry-wait`.
- Use `--force` to re-download everything.

## Ingest behavior
- Incremental ingest uses `data/ingest_manifest.json` to skip already ingested members.
- If snapshots already exist and the ingest manifest is present, ingest auto-switches to incremental mode.
- First incremental run creates the manifest; keep it alongside your DuckDB for repeats.
- Use `--force-ingest` to reprocess the full archive (may duplicate data).

## Logging
All scripts log elapsed time and progress via `shared/utils/progress.py`.
