# Pub Domain Adaptation (Betfair -> TAB)

This page explains the theory and practical behavior behind `punter pub`.

## Problem Framing
`punter pub` uses Betfair market states for inference but targets manual TAB execution.
That creates a domain mismatch:
- Outcome signal domain: Betfair features and market microstructure.
- Execution price domain: TAB accepted odds, which are only partially observed.

If you compute EV directly on Betfair prices for TAB execution, EV is systematically optimistic.

## Two-Model Decision Structure
`punter pub` follows a two-model view:
1. Outcome model: estimate `p_hat` (win probability) from Betfair-derived features.
2. Price translation model: estimate a distribution of executable TAB odds from Betfair state and context.

Decisioning then combines both:
- compute EV with `p_hat` and a conservative TAB price quantile.
- shortlist only bets with positive conservative EV.

## Default Conservative Rule
Pub mode defaults:
- `--execution-domain tab`
- `--tab-odds-quantile 0.10`

Interpretation:
- `q10` means "use a lower, defensive TAB odds estimate rather than median odds".
- A bet survives only if EV is still positive under that downside price assumption.

## No TAB Model Yet? Fallback Behavior
If `--tab-translation-model` is missing or unavailable:
- the system applies a conservative fallback haircut from Betfair odds.
- it still emits TAB quantile columns (`tab_price_q10`, `tab_price_q50`, `tab_price_q90`) from the fallback.

This keeps CLI usage simple while you gather labels.

## Output Fields to Watch
Pub sheet includes:
- `execution_domain`
- `price` (decision price used for EV + staking)
- `betfair_price`
- `tab_price_q10`, `tab_price_q50`, `tab_price_q90`
- `tab_price_source` (`model_quantiles`, `model_median`, or `fallback_haircut`)

## CLI Recipes
Default:
```bash
punter pub
```

With explicit TAB model:
```bash
punter pub --tab-translation-model artifacts/tab_translation_cutoff_10.joblib
```

Increase conservatism:
```bash
punter pub --tab-odds-quantile 0.05
```

Relax conservatism:
```bash
punter pub --tab-odds-quantile 0.20
```

Legacy exchange-style EV:
```bash
punter pub --execution-domain betfair
```

## Label Collection Path
To support translation model training over time, storage includes:
- `tab_quotes` for manually observed display odds.
- `tab_executions` for accepted/repriced/refused execution outcomes.

See schema in `shared/storage/schemas.sql`.
