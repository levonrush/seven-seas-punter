import pandas as pd

from shared.features.builder import build_features_from_store


class FakeStore:
    """Minimal stand-in for DuckDBStore so tests stay fast."""

    def __init__(self, snapshots: pd.DataFrame, results: pd.DataFrame | None = None):
        self._snapshots = snapshots
        self._results = results or pd.DataFrame()

    def load_snapshots(self) -> pd.DataFrame:
        return self._snapshots

    def load_results(self) -> pd.DataFrame:
        return self._results


def test_feature_builder_uses_cutoff_and_sets_ranks():
    snapshots = pd.DataFrame(
        [
            {
                "market_id": "1.1",
                "selection_id": 1,
                "snapshot_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 2.0,
                "best_back_size": 100,
                "best_lay_price": 2.1,
                "best_lay_size": 120,
                "last_traded_price": 2.05,
                "total_matched": 1000,
                "runner_status": "ACTIVE",
                "venue": "Test",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race 1",
            },
            {
                "market_id": "1.1",
                "selection_id": 2,
                "snapshot_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 2.5,
                "best_back_size": 80,
                "best_lay_price": 2.6,
                "best_lay_size": 100,
                "last_traded_price": 2.55,
                "total_matched": 900,
                "runner_status": "ACTIVE",
                "venue": "Test",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race 1",
            },
            {
                "market_id": "1.1",
                "selection_id": 1,
                "snapshot_time": pd.Timestamp("2024-01-01T00:05:00Z"),
                "seconds_to_start": 300,
                "best_back_price": 1.9,
                "best_back_size": 90,
                "best_lay_price": 2.0,
                "best_lay_size": 110,
                "last_traded_price": 1.95,
                "total_matched": 1200,
                "runner_status": "ACTIVE",
                "venue": "Test",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race 1",
            },
        ]
    )
    results = pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 1, "win_flag": True, "bsp": 2.0, "place_position": None},
            {"market_id": "1.1", "selection_id": 2, "win_flag": False, "bsp": 2.5, "place_position": None},
        ]
    )
    store = FakeStore(snapshots=snapshots, results=results)
    features = build_features_from_store(store, cutoff_minutes=5)
    assert len(features) == 2
    prob_runner1 = features.loc[features["selection_id"] == 1, "implied_prob_t10"].iloc[0]
    assert prob_runner1 == 0.5
    rank_runner1 = features.loc[features["selection_id"] == 1, "rank_t10"].iloc[0]
    rank_runner2 = features.loc[features["selection_id"] == 2, "rank_t10"].iloc[0]
    assert rank_runner1 < rank_runner2  # better odds -> higher prob -> lower rank number
    assert features["win_target"].isnull().sum() == 0
