from __future__ import annotations


EXOTIC_MARKET_TYPES = {
    "EXACTA",
    "FORECAST",
    "REVERSE_FORECAST",
    "QUINELLA",
    "TRIFECTA",
    "TRICAST",
    "SUPERFECTA",
    "FIRST4",
    "FIRST_FOUR",
    "QUAD",
    "DAILY_DOUBLE",
    "DUEL",
    "MATCH_BET",
}


def bucket_market_type(market_type: str | None) -> str:
    """Map a Betfair market type into WIN/PLACE/EXOTIC/OTHER buckets for model routing."""
    if not market_type:
        return "UNKNOWN"
    market_type = str(market_type).strip().upper()
    if not market_type:
        return "UNKNOWN"
    if market_type == "WIN":
        return "WIN"
    if market_type == "PLACE":
        return "PLACE"
    if market_type in EXOTIC_MARKET_TYPES:
        return "EXOTIC"
    return "OTHER"
