# Backtesting

The backtest engine simulates a simple value strategy using model probabilities.

## Strategy
- Compute expected value: `p_hat * (price - 1) * (1 - commission) - (1 - p_hat)`.
- Filter by minimum EV, edge, spread, max price, edge multiplier, and optional `min_prob`.
- Apply optional risk controls (max bets per day, max exposure per race).
- The CLI prefers out-of-fold predictions for in-sample backtests to reduce optimism bias.
- When `market_type` is present, the CLI filters to WIN markets to keep targets consistent.
- Commission settlement supports `per_bet` and `market_net` modes; rescue mode defaults to market-net.

If `min_prob` is unset, the CLI will use the tuned `kappa_threshold` from the latest model run.

## Outputs
- Metrics: profit, ROI, expected ROI, hit rate, max drawdown.
- Bet preview tables include market/bet annotations to make exotic selections obvious.

## Strategy tuning
The tuner sweeps a small grid of filter settings to surface configs that maximize expected ROI
and shows the hit-rate trade-off per bucket. The grid includes `min_prob` cutoffs, and when a
`min_prob` is supplied it centers the grid around that value.

## Kelly sizing theory
When budget-based stake allocation is enabled for pub/score sheets, this repo uses Kelly sizing.

- Goal: maximize long-run log growth of bankroll, not short-run hit rate.
- For a back bet with model win probability `p`, decimal odds `o`, and commission `c`:
  - `b = (o - 1) * (1 - c)` (net odds if the bet wins)
  - `q = 1 - p`
  - Full Kelly fraction: `f* = (b * p - q) / b`
- If `f* <= 0`, theory says no bet.
- Fractional Kelly uses `f = alpha * max(0, f*)` where `alpha` is `kelly_fraction`.

Why fractional Kelly is used in practice:
- Probability estimates are noisy.
- Bets are correlated (same meeting/time block), violating ideal Kelly assumptions.
- Lower fractions reduce drawdown volatility and model-error blowups.

Rule of thumb:
- `alpha = 0.25`: conservative default.
- `alpha = 0.5`: more aggressive.
- `alpha = 1.0`: full Kelly, high variance and typically too unstable for real bankrolls.
