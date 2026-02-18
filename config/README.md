# Config

Configuration files for runtime behavior live here.

## Files
- `live.yaml`: main config for `punter live`.

## Quick start
Create or refresh a starter config:

```bash
punter live --write-config config/live.yaml
```

Run one dry-run iteration with that config:

```bash
punter live --config config/live.yaml --once --dry-run
```

## What `live.yaml` controls
- `dry_run`, `poll_interval_seconds`, `max_iterations`: loop behavior.
- `model.*`: model artifact paths and cutoff used for inference.
- `markets.*`: market discovery filters (countries, market types, horizon).
- `strategy.*`: bet selection thresholds and default stake.
- `safety.*`: hard risk caps (market/day exposure and time-to-start guardrail).
- `state.*`: persisted exposure and decision log output paths.

## Editing guidance
- Keep this file environment-agnostic where possible (relative paths under repo root).
- Prefer generating defaults with `--write-config`, then edit small deltas.
- Validate config changes with `--dry-run` before any live mode run.
- If you want all market types, set:
  - `markets.market_type_codes: [ALL]`
