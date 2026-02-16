from workflow.download_historic import _normalize_market_types


def test_normalize_market_types_all_disables_filter():
    assert _normalize_market_types(["ALL"]) == []
    assert _normalize_market_types(["*", "WIN"]) == []


def test_normalize_market_types_normalizes_and_deduplicates():
    assert _normalize_market_types([" win ", "EXACTA", "Win", "QUINELLA"]) == [
        "WIN",
        "EXACTA",
        "QUINELLA",
    ]


def test_normalize_market_types_empty_values_return_empty_list():
    assert _normalize_market_types([]) == []
