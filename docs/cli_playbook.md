# CLI Playbook

This is the canonical runbook for day-to-day CLI usage.
For modeling rationale behind the new form layers, see `docs/form_intelligence_layer.md`.

## Fast Paths
One command (recommended default):
```bash
punter go
```
Default `go` behavior:
- checks DuckDB snapshots before downloading;
- skips `download-historic` when data is fresh;
- auto-refreshes historic data when the latest snapshot is older than 30 days.

Force or tune refresh behavior:
```bash
# always refresh historic first
punter go --refresh-historic

# refresh only if older than 14 days
punter go --refresh-historic-if-stale-days 14

# disable stale-data auto-refresh (skip historic when snapshots exist)
punter go --refresh-historic-if-stale-days -1
```

Manual pipeline control:
```bash
punter download-historic --auto
punter pipeline --cutoff-minutes 10
```

Incremental update cycle:
```bash
punter download-historic --auto
punter ingest --incremental
```

## Add New Sports / Scopes
Add another sport into the local archive+DB (example: greyhounds):
```bash
punter download-historic --auto --sport "Greyhound Racing" --market-types ALL --clean-temp
punter pipeline --ingest-new --cutoff-minutes 10
```

Narrow to specific market types while onboarding:
```bash
punter download-historic --auto --sport "Horse Racing" --market-types WIN,PLACE
punter pipeline --ingest-new --cutoff-minutes 10
```

Inspect available market-type codes before committing:
```bash
punter download-historic --auto --sport "Horse Racing" --show-market-types
```

If ingest/download holes were detected (bad stream members):
```bash
punter repair-manifests --bad-list artifacts/ingest_bad_stream_members.txt
punter download-historic --auto --refresh-list-cache
punter pipeline --ingest-new --cutoff-minutes 10
```

## Two Operating Modes
Automated exchange execution loop (safety-first):
```bash
punter live --write-config config/live.yaml
punter live --config config/live.yaml --once --dry-run
punter live --config config/live.yaml --dry-run
# promote only after checks:
punter live --config config/live.yaml --live
```

Manual pub sheet (TAB-focused by default):
```bash
punter pub
```

## Pub Mode Common Commands
Default conservative shortlist:
```bash
punter pub
```

Filter and output:
```bash
punter pub --min-prob 0.12 --market-types WIN --output artifacts/pub_sheet.csv
```

Budget-aware stake sizing:
```bash
punter pub --budget 100 --kelly-fraction 0.25 --max-bet-pct 0.2
```

Use trained TAB translation model:
```bash
punter pub --tab-translation-model artifacts/tab_translation_cutoff_10.joblib
```

Adjust conservatism:
```bash
punter pub --tab-odds-quantile 0.15
```

Force exchange-style scoring:
```bash
punter pub --execution-domain betfair
```

## Core Build/Eval Commands
Train:
```bash
punter train --cutoff-minutes 10
```

Backtest:
```bash
punter backtest --cutoff-minutes 10
```

Report latest run:
```bash
punter report
```
`punter report` now also prints latest runner-metadata completeness (when metadata snapshots exist).

Capture prospective Betfair runner metadata snapshots (for form features):
```bash
punter download --date 2026-02-23
```

Ingest licensed external form runs (P1 form layer):
```bash
punter ingest-form --input data/external_form_runs.json --source punting_form
```
Auto-ingest is now default-on in `features/train/backtest/score/pipeline/go` when the external-form table is empty and a default file exists (`data/external_form_runs.parquet|csv|json|jsonl` or `data/external_form.json`).

Disable per command when needed:
```bash
punter go --no-external-form-ingest
punter pipeline --no-external-form-ingest
```

## Default Filter Behavior
By default, `train`, `backtest`, `score`, `pipeline`, and `go` do not apply EV/edge/price/spread/min-prob strategy filters automatically.

Enable filters explicitly when needed:
```bash
punter backtest --min-ev 0.02 --min-edge 0.1 --max-price 20 --max-spread 0.5
punter score --min-prob 0.2 --min-ev 0.01
punter train --tune-strategy
```

Opt into tuned probability cutoffs explicitly:
```bash
punter backtest --use-kappa-thresholds
punter backtest --use-market-type-kappa-thresholds
punter score --use-kappa-thresholds
```

## Helpful Help
```bash
punter --help
punter pub --help
punter live --help
punter pipeline --help
```
