from __future__ import annotations

from typing import Optional

import pandas as pd


def _full_kelly_fraction(prob: float, price: float, commission: float) -> float:
    """Return full Kelly fraction for a back bet so stake sizing is tied to edge and odds shape."""
    if prob <= 0 or prob >= 1:
        return 0.0
    net_odds = (price - 1.0) * (1.0 - commission)
    if net_odds <= 0:
        return 0.0
    q = 1.0 - prob
    fraction = (net_odds * prob - q) / net_odds
    return max(0.0, float(fraction))


def allocate_stakes_from_budget(
    candidates: pd.DataFrame,
    budget: float,
    commission: float = 0.05,
    method: str = "fractional_kelly",
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 0.2,
) -> pd.DataFrame:
    """Allocate stake suggestions from a day budget so the pub sheet includes concrete sizing guidance.

    The default method uses fractional Kelly to preserve bankroll growth logic while damping variance.
    """
    if candidates.empty:
        return candidates.copy()
    if budget <= 0:
        raise ValueError("budget must be > 0.")
    if max_bet_pct <= 0:
        raise ValueError("max_bet_pct must be > 0.")
    if kelly_fraction <= 0:
        raise ValueError("kelly_fraction must be > 0.")
    if method not in {"fractional_kelly", "equal"}:
        raise ValueError("method must be one of: fractional_kelly, equal.")

    if "p_hat" not in candidates.columns or "price" not in candidates.columns:
        raise ValueError("candidates must include `p_hat` and `price` columns.")

    allocated = candidates.copy()
    allocated["allocation_method"] = method

    if method == "equal":
        stake = budget / max(1, len(allocated))
        cap = budget * max_bet_pct
        stake = min(stake, cap)
        allocated["kelly_fraction_full"] = 0.0
        allocated["stake_fraction"] = stake / budget
        allocated["suggested_stake"] = round(stake, 2)
        return allocated

    allocated["kelly_fraction_full"] = allocated.apply(
        lambda row: _full_kelly_fraction(
            prob=float(row["p_hat"]),
            price=float(row["price"]),
            commission=float(commission),
        ),
        axis=1,
    )
    allocated["stake_fraction"] = allocated["kelly_fraction_full"] * float(kelly_fraction)
    allocated["suggested_stake"] = budget * allocated["stake_fraction"]

    max_bet_amount = budget * float(max_bet_pct)
    allocated["suggested_stake"] = allocated["suggested_stake"].clip(lower=0.0, upper=max_bet_amount)

    total = float(allocated["suggested_stake"].sum())
    if total > budget and total > 0:
        scale = budget / total
        allocated["suggested_stake"] = allocated["suggested_stake"] * scale
        allocated["stake_fraction"] = allocated["suggested_stake"] / budget

    allocated["suggested_stake"] = allocated["suggested_stake"].round(2)
    return allocated


def summarize_budget_usage(candidates: pd.DataFrame, budget: Optional[float]) -> tuple[float, float]:
    """Return used/remaining budget numbers for concise CLI logging."""
    if budget is None:
        return 0.0, 0.0
    used = float(candidates.get("suggested_stake", pd.Series(dtype=float)).fillna(0.0).sum())
    remaining = max(0.0, float(budget) - used)
    return used, remaining

