# Feature builder

The feature builder constructs leakage-safe rows per `(market_id, selection_id)` at a chosen cutoff.

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

Targets are joined from results (`win_target`) when available.
