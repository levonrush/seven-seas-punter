# CLI Playbook

This is the canonical runbook for day-to-day CLI usage.

## Fast Paths
One command (recommended default):
```bash
punter go
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

## Helpful Help
```bash
punter --help
punter pub --help
punter live --help
punter pipeline --help
```
