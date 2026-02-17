from shared.utils.market_types import (
    api_market_type_codes,
    market_type_matches,
    normalize_market_type_tokens,
    tokens_to_filter_set,
)


def test_normalize_market_type_tokens_all_behavior():
    assert normalize_market_type_tokens("ALL") == ["ALL"]
    assert normalize_market_type_tokens("*,WIN") == ["ALL"]
    assert normalize_market_type_tokens(None) == ["ALL"]


def test_normalize_market_type_tokens_dedupes_and_uppercases():
    assert normalize_market_type_tokens(" win ,EXACTA,Win ") == ["WIN", "EXACTA"]


def test_api_market_type_codes_omits_bucket_aliases():
    tokens = normalize_market_type_tokens("EXOTIC,EXACTA,WIN")
    assert api_market_type_codes(tokens) == ["EXACTA", "WIN"]


def test_market_type_matches_supports_bucket_alias():
    selected = tokens_to_filter_set(["EXOTIC"])
    assert market_type_matches("EXACTA", selected) is True
    assert market_type_matches("WIN", selected) is False

