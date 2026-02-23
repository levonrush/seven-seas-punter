import pandas as pd

from shared.model.predictions import build_prediction_preview


def test_prediction_preview_keeps_market_type_with_overlapping_market_columns():
    """Ensure preview market_type survives market-table merges when feature rows already include market columns."""
    feature_df = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "selection_id": 101,
                "race_start_time": pd.Timestamp("2024-01-01T12:00:00Z"),
                "venue": "Albion Park",
                "market_type": "WIN",
                "back_price_t10": 5.0,
            }
        ]
    )
    probs = pd.Series([1.0], index=feature_df.index)
    runners = pd.DataFrame([{"market_id": "1.1", "selection_id": 101, "runner_name": "1. Test Runner"}])
    markets = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "venue": "Albion Park",
                "race_start_time": pd.Timestamp("2024-01-01T12:00:00Z"),
                "market_type": "WIN",
            }
        ]
    )

    preview = build_prediction_preview(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        runners=runners,
        markets=markets,
        limit=5,
        min_ev=0.02,
        min_edge=0.1,
        max_price=200.0,
        max_edge_multiplier=5.0,
        per_market_limit=1,
        min_prob=0.35,
    )

    assert len(preview) == 1
    assert preview.loc[preview.index[0], "market_type"] == "WIN"
    assert "UNKNOWN" not in preview.loc[preview.index[0], "market_type_label"]


def test_prediction_preview_backfills_market_type_from_market_table():
    """Ensure missing feature-frame market_type values are backfilled from persisted market metadata."""
    feature_df = pd.DataFrame(
        [
            {
                "market_id": "1.2",
                "selection_id": 202,
                "race_start_time": pd.Timestamp("2024-01-01T13:00:00Z"),
                "venue": "Bathurst",
                "market_type": None,
                "back_price_t10": 4.0,
            }
        ]
    )
    probs = pd.Series([0.9], index=feature_df.index)
    markets = pd.DataFrame(
        [
            {
                "market_id": "1.2",
                "venue": "Bathurst",
                "race_start_time": pd.Timestamp("2024-01-01T13:00:00Z"),
                "market_type": "PLACE",
            }
        ]
    )

    preview = build_prediction_preview(
        feature_df=feature_df,
        probs=probs,
        cutoff_minutes=10,
        markets=markets,
        limit=5,
        min_ev=0.02,
        min_edge=0.1,
        max_price=200.0,
        max_edge_multiplier=5.0,
        per_market_limit=1,
        min_prob=0.35,
    )

    assert len(preview) == 1
    assert preview.loc[preview.index[0], "market_type"] == "PLACE"
    assert "PLACE" in preview.loc[preview.index[0], "market_type_label"]
