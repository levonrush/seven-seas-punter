from __future__ import annotations

from typing import Iterable, Optional

from shared.model.market_type import bucket_market_type


_BUCKET_ALIASES = {"EXOTIC", "OTHER", "UNKNOWN"}


def normalize_market_type_tokens(values: str | Iterable[str] | None) -> list[str]:
    """Normalize user market-type inputs so CLI/config filtering is deterministic across modules."""
    if values is None:
        return ["ALL"]
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = list(values)
    tokens: list[str] = []
    seen = set()
    for value in raw_values:
        token = str(value).strip().upper()
        if not token:
            continue
        if token in {"ALL", "*"}:
            return ["ALL"]
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens or ["ALL"]


def tokens_to_filter_set(tokens: Iterable[str] | None) -> Optional[set[str]]:
    """Convert normalized tokens to an in-memory filter set, or None when filtering is disabled."""
    if tokens is None:
        return None
    normalized = normalize_market_type_tokens(tokens)
    if "ALL" in normalized:
        return None
    return set(normalized)


def api_market_type_codes(tokens: Iterable[str] | None) -> list[str]:
    """Return API-compatible market type codes from token filters, omitting broad bucket aliases."""
    selected = tokens_to_filter_set(tokens)
    if selected is None:
        return []
    return sorted(token for token in selected if token not in _BUCKET_ALIASES)


def market_type_matches(market_type: str | None, selected: Optional[set[str]]) -> bool:
    """Return True when a market type matches either an exact token or bucket alias selection."""
    if selected is None:
        return True
    raw = (market_type or "").strip().upper()
    bucket = bucket_market_type(raw).upper()
    return raw in selected or bucket in selected
