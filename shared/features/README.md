# Feature builder

The feature builder constructs leakage-safe rows per `(market_id, selection_id)` at a chosen cutoff.
For full rationale behind the form layers (P0/P1/P2), see `docs/form_intelligence_layer.md`.

## Cutoffs
Snapshots are filtered to:
- `snapshot_time < race_start_time`
- `seconds_to_start >= cutoff_minutes * 60`

Supported offsets: `60, 30, 10, 5, 2, 1` minutes. Offsets later than the cutoff are blanked.

## Core features
- Back/lay prices, last traded price, implied probability, mid price, and spread (absolute + %).
- Tick-normalized prices/spreads plus tick deltas between offsets.
- Top-of-book sizes, liquidity, liquidity-weighted price, and size imbalance at each offset.
- Log-odds transforms plus price/volume/spread/imbalance deltas between offsets.
- Market rank, implied-probability share, and market tightness/overround per offset.
- Field size plus basic price/spread ranges and volatility across offsets.
- Form-intelligence block:
  - pre-race horse Elo ratings (winner-only multinomial updates),
  - pre-race Plackett-Luce top-m hierarchical ratings using prior finishing order (`place_position` when available, winner fallback otherwise),
  - hierarchical rating components for horse, jockey, trainer, and race context (surface/distance/track),
  - PL-derived pre-race probabilities (win/top2/top3) and field entropy,
  - field-relative strength (percentile/rank, gap-to-top, entropy),
  - conservative rolling jockey/trainer rates on a 365-day window with minimum-history gating.
- Optional Betfair `RUNNER_METADATA` joins (when captured prospectively) with explicit missingness flags.
- Optional licensed external-form block (`external_runner_form_runs`):
  - last-10 starts (wins/places/rates),
  - recency + 60-day decayed win/place rates,
  - distance/surface/track context split rates,
  - class progression and sectional/speed summaries,
  - horse-jockey / horse-trainer / jockey-trainer interaction rates.

Targets are joined from results (`win_target`) when available.
