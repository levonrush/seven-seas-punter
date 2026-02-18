# Documentation Hub

Use this folder as the primary docs entrypoint. Top-level and module READMEs stay intentionally concise and link here for depth.

## Start Here
- Project quick start: `README.md`
- Workflow CLI overview: `workflow/README.md`
- Shared library modules: `shared/README.md`

## By Task
- Run commands end-to-end: `docs/cli_playbook.md`
- Understand live integration constraints and architecture: `docs/live_integration_plan.md`
- Understand pub-mode domain adaptation (Betfair -> TAB): `docs/pub_domain_adaptation.md`
- Review staking and Kelly assumptions: `shared/backtest/README.md`
- Betfair API implementation notes: `docs/betfair_api/authentication.md`, `docs/betfair_api/market_discovery.md`, `docs/betfair_api/market_prices.md`, `docs/betfair_api/bet_execution.md`, `docs/betfair_api/rate_limits.md`

## Design Principle
- Keep runbook commands in `docs/cli_playbook.md`.
- Keep model/execution theory in focused docs (for example `docs/pub_domain_adaptation.md`).
- Keep root and module READMEs short, stable, and navigation-first.
