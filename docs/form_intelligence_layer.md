# Form Intelligence Layer (P0/P1/P2)

This page documents the form-modeling work added on top of the exchange microstructure baseline, including what was built, why the design is layered, and where leakage controls are enforced.

## Why this was built
The original stack was strong on Betfair market-state signals (price, spread, liquidity, rank, momentum) but weak on non-market racing context:
- horse condition and recency,
- jockey/trainer effects,
- distance/surface/track suitability,
- finish-order structure beyond winner-only labels.

The form layer addresses those gaps while preserving point-in-time reproducibility.

## Why we used multiple approaches (instead of one)
We intentionally split form work into three phases because data availability, reliability, and leakage risk differ by source:
- P0 uses prospectively captured Betfair metadata and conservative updates for a robust base.
- P1 adds richer external provider form runs behind explicit ingestion and strict pre-race filtering.
- P2 adds richer ranking dynamics (Plackett-Luce top-m + hierarchical components) when finish-order data is available.

This lets us gain signal incrementally without coupling model quality to any single upstream feed.

## P0: Foundational form features
### What was implemented
- Prospective runner metadata snapshots table:
  - `runner_metadata_snapshots` in `shared/storage/schemas.sql`
- Ingestion from market-catalog metadata:
  - `workflow/download_data.py` -> `extract_runner_metadata_snapshots(...)`
- Point-in-time metadata joins in feature building:
  - `shared/features/builder.py` -> `_attach_runner_metadata_features(...)`
- Leakage-safe base form features:
  - horse pre-race Elo (`_add_pre_race_elo_features(...)`)
  - conservative jockey/trainer rolling rates (`_add_pre_race_entity_rolling_rates(...)`)

### Why this approach
- Betfair metadata is available in existing acquisition flow, so P0 is low-friction and reproducible.
- Conservative rolling rates (365d + min-history gating) reduce overreaction on sparse entities.
- Elo gives a stable first step for field-relative strength without requiring full placings.

### Leakage controls
- Metadata is stored as pre-race snapshots; joins are cutoff-aware.
- Feature build only uses snapshots before race start and at/after selected cutoff.
- Elo/rolling updates are strictly race-time ordered and use prior race outcomes only.

## P1: External provider form-run intelligence
### What was implemented
- Normalized external run-history ingestion:
  - `shared/form/ingest.py` and CLI `punter ingest-form`
- New table:
  - `external_runner_form_runs` in `shared/storage/schemas.sql`
- Store read/write helpers:
  - `shared/storage/duckdb_store.py`
- Feature block for last-N and context splits:
  - `shared/features/builder.py` -> `_attach_external_form_features(...)`
  - `_compute_external_form_runner_features(...)`

### Feature families
- last-10 starts: wins, places, rates
- recency and 60-day decayed win/place rates
- distance/surface/track split rates
- class progression + sectional/speed summaries
- horse/jockey/trainer interaction rates

### Why this approach
- External providers carry stronger racing context than exchange feed alone.
- We normalize provider payloads to one stable schema so model code stays provider-agnostic.
- We keep this layer optional at data-source level (features naturally absent if no data), but operationally default-on when data is present.

### Leakage controls
- Runs are filtered to `run_date < race_start_time` for each target runner.
- Only rows visible at cutoff are loaded from store.
- Minimum-count gates avoid false confidence for sparse context splits.

## P2: Plackett-Luce top-m hierarchical ratings
### What was implemented
- Finish-order extraction with fallback:
  - uses `place_position` when available; falls back to winner-only
  - `shared/features/builder.py` -> `_extract_market_finish_orders(...)`
- Pre-race hierarchical rating features:
  - `_add_pre_race_pl_hierarchical_features(...)`
- Integrated into final form block:
  - `_add_form_intelligence_features(..., results_df=...)`

### Feature families
- effective pre-race rating and rank
- pre-race win/top2/top3 probabilities
- field entropy
- component contributions:
  - horse,
  - jockey,
  - trainer,
  - race context (surface/distance/track)

### Why this approach
- Winner-only updates lose information about strong non-winners.
- Top-m PL updates use partial finishing order for better ranking dynamics.
- Hierarchical components share signal across contexts while staying interpretable.

### Leakage controls
- Features are written before current-race updates.
- Updates are applied only after each race and only from historical results.
- Race ordering uses timestamp + market grouping to maintain causality.

## Default CLI behavior (current)
Form intelligence is now default behavior for core flows:
- `punter features`
- `punter train`
- `punter backtest`
- `punter score`
- `punter pipeline`
- `punter go`

External-form auto-ingest is default-on when:
- `external_runner_form_runs` is empty, and
- a default file exists in `data/`:
  - `external_form_runs.parquet|csv|json|jsonl`
  - `external_form.json`

Opt-out is explicit:
- `--no-external-form-ingest`

Important distinction:
- Form features default-on.
- Strategy thresholds/filters remain opt-in by design.

## Why these defaults were chosen
- New model capability should be active without requiring users to remember new flags.
- Disabling should be explicit and intentional (`--no-*`), especially for production workflows.
- We preserve historical behavior where strategy/risk filters are user-controlled rather than silently tightened.

## Data quality and observability
`punter report` surfaces metadata and external form completeness summaries so feed drift is visible without custom queries.

## Validation added
- Feature tests for:
  - Elo causality behavior,
  - external-form context features,
  - PL hierarchical place-order updates,
  - winner-only fallback when place data is missing.
- CLI tests for:
  - new parser defaults,
  - `go` argument forwarding for external-form controls.

## Practical guidance
- If you have external provider history, keep one default file in `data/` and run `punter go`.
- If you need a controlled run without external-form ingestion:
  - `punter go --no-external-form-ingest`
- If provider schema or timestamp fields vary, use explicit ingestion first:
  - `punter ingest-form --input <path> --source <label>`
