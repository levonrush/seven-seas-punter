# Backtesting

The backtest engine simulates a simple value strategy using model probabilities.

## Strategy
- Compute expected value: `p_hat * (price - 1) * (1 - commission) - (1 - p_hat)`.
- Filter by minimum EV, edge, spread, max price, edge multiplier, and optional `min_prob`.
- Apply optional risk controls (max bets per day, max exposure per race).

If `min_prob` is unset, the CLI will use the tuned `kappa_threshold` from the latest model run.

## Outputs
- Metrics: profit, ROI, expected ROI, hit rate, max drawdown.
- Bet preview tables include market/bet annotations to make exotic selections obvious.

## Strategy tuning
The tuner sweeps a small grid of filter settings to surface configs that maximize expected ROI
and shows the hit-rate trade-off per bucket. The grid includes `min_prob` cutoffs, and when a
`min_prob` is supplied it centers the grid around that value.
