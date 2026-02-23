import pandas as pd

from shared.features.builder import build_features_from_store, split_features_by_race_time


class FakeStore:
    """Minimal stand-in for DuckDBStore so tests stay fast."""

    def __init__(
        self,
        snapshots: pd.DataFrame,
        results: pd.DataFrame | None = None,
        runners: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        external_form: pd.DataFrame | None = None,
    ):
        self._snapshots = snapshots
        self._results = results if results is not None else pd.DataFrame()
        self._runners = runners if runners is not None else pd.DataFrame()
        self._metadata = metadata if metadata is not None else pd.DataFrame()
        self._external_form = external_form if external_form is not None else pd.DataFrame()

    def load_snapshots(self) -> pd.DataFrame:
        return self._snapshots

    def load_results(self) -> pd.DataFrame:
        return self._results

    def load_runners(self) -> pd.DataFrame:
        return self._runners

    def load_runner_metadata_for_cutoff(self, cutoff_minutes: int) -> pd.DataFrame:
        return self._metadata

    def load_external_runner_form_for_cutoff(self, cutoff_minutes: int) -> pd.DataFrame:
        return self._external_form


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
    assert pd.isna(features.loc[features["selection_id"] == 1, "back_price_t2"].iloc[0])


def test_split_features_by_race_time():
    features = pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 1, "race_start_time": "2024-01-01T00:10:00Z"},
            {"market_id": "1.2", "selection_id": 2, "race_start_time": "2024-01-02T00:10:00Z"},
        ]
    )
    train_df, test_df = split_features_by_race_time(features, "2024-01-02")
    assert len(train_df) == 1
    assert len(test_df) == 1


def test_feature_builder_filters_to_requested_market_ids():
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
                "venue": "A",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race A",
            },
            {
                "market_id": "1.2",
                "selection_id": 2,
                "snapshot_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 3.0,
                "best_back_size": 90,
                "best_lay_price": 3.1,
                "best_lay_size": 100,
                "last_traded_price": 3.05,
                "total_matched": 900,
                "runner_status": "ACTIVE",
                "venue": "B",
                "race_start_time": pd.Timestamp("2024-01-01T00:12:00Z"),
                "race_name": "Race B",
            },
        ]
    )
    store = FakeStore(snapshots=snapshots, results=pd.DataFrame())
    features = build_features_from_store(store, cutoff_minutes=10, market_ids=["1.1"])
    assert len(features) == 1
    assert features["market_id"].tolist() == ["1.1"]


def test_feature_builder_adds_elo_features_without_using_current_race_result():
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
                "venue": "A",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race 1",
            },
            {
                "market_id": "1.1",
                "selection_id": 2,
                "snapshot_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 3.0,
                "best_back_size": 80,
                "best_lay_price": 3.1,
                "best_lay_size": 90,
                "last_traded_price": 3.05,
                "total_matched": 900,
                "runner_status": "ACTIVE",
                "venue": "A",
                "race_start_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "race_name": "Race 1",
            },
            {
                "market_id": "1.2",
                "selection_id": 1,
                "snapshot_time": pd.Timestamp("2024-01-02T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 2.2,
                "best_back_size": 95,
                "best_lay_price": 2.3,
                "best_lay_size": 110,
                "last_traded_price": 2.25,
                "total_matched": 1100,
                "runner_status": "ACTIVE",
                "venue": "B",
                "race_start_time": pd.Timestamp("2024-01-02T00:10:00Z"),
                "race_name": "Race 2",
            },
            {
                "market_id": "1.2",
                "selection_id": 2,
                "snapshot_time": pd.Timestamp("2024-01-02T00:00:00Z"),
                "seconds_to_start": 600,
                "best_back_price": 2.8,
                "best_back_size": 85,
                "best_lay_price": 2.9,
                "best_lay_size": 105,
                "last_traded_price": 2.85,
                "total_matched": 950,
                "runner_status": "ACTIVE",
                "venue": "B",
                "race_start_time": pd.Timestamp("2024-01-02T00:10:00Z"),
                "race_name": "Race 2",
            },
        ]
    )
    runners = pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 1, "runner_name": "Horse Alpha", "stall_draw": 1},
            {"market_id": "1.1", "selection_id": 2, "runner_name": "Horse Beta", "stall_draw": 2},
            {"market_id": "1.2", "selection_id": 1, "runner_name": "Horse Alpha", "stall_draw": 3},
            {"market_id": "1.2", "selection_id": 2, "runner_name": "Horse Beta", "stall_draw": 4},
        ]
    )
    results = pd.DataFrame(
        [
            {"market_id": "1.1", "selection_id": 1, "win_flag": 1},
            {"market_id": "1.1", "selection_id": 2, "win_flag": 0},
            {"market_id": "1.2", "selection_id": 1, "win_flag": 0},
            {"market_id": "1.2", "selection_id": 2, "win_flag": 1},
        ]
    )
    store = FakeStore(snapshots=snapshots, results=results, runners=runners)
    features = build_features_from_store(store, cutoff_minutes=10).sort_values(
        ["market_id", "selection_id"]
    )

    race1 = features[features["market_id"] == "1.1"]
    assert race1["horse_elo_rating_pre"].nunique() == 1
    assert race1["horse_elo_rating_pre"].iloc[0] == 1500.0

    race2 = features[features["market_id"] == "1.2"].set_index("selection_id")
    assert race2.loc[1, "horse_elo_rating_pre"] > race2.loc[2, "horse_elo_rating_pre"]
    assert race2.loc[1, "horse_elo_prob_pre"] > race2.loc[2, "horse_elo_prob_pre"]
    assert race2["field_elo_entropy_pre"].between(0.0, 1.0).all()


def test_feature_builder_adds_metadata_missingness_and_conservative_entity_rates():
    snapshot_rows = []
    runner_rows = []
    result_rows = []
    metadata_rows = []
    for race_idx in range(1, 22):
        market_id = f"1.{race_idx}"
        race_start = pd.Timestamp("2024-01-01T00:10:00Z") + pd.Timedelta(days=race_idx)
        snapshot_rows.append(
            {
                "market_id": market_id,
                "selection_id": 1,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "seconds_to_start": 600,
                "best_back_price": 2.0,
                "best_back_size": 100,
                "best_lay_price": 2.1,
                "best_lay_size": 120,
                "last_traded_price": 2.05,
                "total_matched": 1000,
                "runner_status": "ACTIVE",
                "venue": "Test",
                "race_start_time": race_start,
                "race_name": f"Race {race_idx}",
            }
        )
        runner_rows.append(
            {
                "market_id": market_id,
                "selection_id": 1,
                "runner_name": "Consistent Horse",
                "stall_draw": 5,
            }
        )
        result_rows.append({"market_id": market_id, "selection_id": 1, "win_flag": 1})
        metadata_rows.append(
            {
                "market_id": market_id,
                "selection_id": 1,
                "jockey_name": "J Smith",
                "trainer_name": "T Jones",
                "age": 5,
                "official_rating": 75,
                "adjusted_rating": 74,
                "days_since_last_run": 14,
                "weight_value": 56.5,
                "jockey_claim": 0.0,
                "stall_draw": 5,
                "form_string": "1111",
            }
        )
    store = FakeStore(
        snapshots=pd.DataFrame(snapshot_rows),
        results=pd.DataFrame(result_rows),
        runners=pd.DataFrame(runner_rows),
        metadata=pd.DataFrame(metadata_rows),
    )
    features = build_features_from_store(store, cutoff_minutes=10).sort_values("race_start_time")
    assert (features["metadata_any_available"] == 1).all()
    assert (features["missing_jockey_name"] == 0).all()
    assert (features["missing_trainer_name"] == 0).all()
    assert features.iloc[19]["jockey_confident_history"] == 0
    assert features.iloc[20]["jockey_confident_history"] == 1
    assert features.iloc[20]["jockey_prior_rides_365d"] == 20
    assert features.iloc[20]["jockey_win_rate_365d"] == 1.0


def test_feature_builder_adds_external_form_features_with_context_splits():
    race_start = pd.Timestamp("2024-01-20T05:00:00Z")
    snapshots = pd.DataFrame(
        [
            {
                "market_id": "1.200",
                "selection_id": 1,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "seconds_to_start": 600,
                "best_back_price": 2.4,
                "best_back_size": 120,
                "best_lay_price": 2.5,
                "best_lay_size": 140,
                "last_traded_price": 2.45,
                "total_matched": 1500,
                "runner_status": "ACTIVE",
                "venue": "Randwick",
                "race_start_time": race_start,
                "race_name": "Race P1",
            },
            {
                "market_id": "1.200",
                "selection_id": 2,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "seconds_to_start": 600,
                "best_back_price": 3.6,
                "best_back_size": 90,
                "best_lay_price": 3.7,
                "best_lay_size": 100,
                "last_traded_price": 3.65,
                "total_matched": 1000,
                "runner_status": "ACTIVE",
                "venue": "Randwick",
                "race_start_time": race_start,
                "race_name": "Race P1",
            },
        ]
    )
    runners = pd.DataFrame(
        [
            {"market_id": "1.200", "selection_id": 1, "runner_name": "Horse Prime", "stall_draw": 4},
            {"market_id": "1.200", "selection_id": 2, "runner_name": "Horse Drift", "stall_draw": 7},
        ]
    )
    metadata = pd.DataFrame(
        [
            {"market_id": "1.200", "selection_id": 1, "jockey_name": "A Rider", "trainer_name": "Stable A"},
            {"market_id": "1.200", "selection_id": 2, "jockey_name": "B Rider", "trainer_name": "Stable B"},
        ]
    )
    external_form = pd.DataFrame(
        [
            {
                "market_id": "1.200",
                "selection_id": 1,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "race_start_time": race_start,
                "seconds_to_start": 600,
                "run_index": 1,
                "distance_m": 1200,
                "surface": "TURF",
                "track": "Randwick",
                "class_index": 6.0,
                "run_date": race_start - pd.Timedelta(days=7),
                "run_finish_pos": 1,
                "run_distance_m": 1200,
                "run_surface": "TURF",
                "run_track": "Randwick",
                "run_class_index": 6.0,
                "run_sectional_time": 34.2,
                "run_speed_rating": 92.0,
                "run_jockey_name": "A Rider",
                "run_trainer_name": "Stable A",
                "run_won": True,
                "run_placed": True,
            },
            {
                "market_id": "1.200",
                "selection_id": 1,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "race_start_time": race_start,
                "seconds_to_start": 600,
                "run_index": 2,
                "distance_m": 1200,
                "surface": "TURF",
                "track": "Randwick",
                "class_index": 6.0,
                "run_date": race_start - pd.Timedelta(days=28),
                "run_finish_pos": 2,
                "run_distance_m": 1200,
                "run_surface": "TURF",
                "run_track": "Randwick",
                "run_class_index": 6.0,
                "run_sectional_time": 34.8,
                "run_speed_rating": 89.0,
                "run_jockey_name": "A Rider",
                "run_trainer_name": "Stable A",
                "run_won": False,
                "run_placed": True,
            },
            {
                "market_id": "1.200",
                "selection_id": 1,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "race_start_time": race_start,
                "seconds_to_start": 600,
                "run_index": 3,
                "distance_m": 1200,
                "surface": "TURF",
                "track": "Randwick",
                "class_index": 6.0,
                "run_date": race_start + pd.Timedelta(days=3),
                "run_finish_pos": 1,
                "run_distance_m": 1200,
                "run_surface": "TURF",
                "run_track": "Randwick",
                "run_class_index": 6.0,
                "run_sectional_time": 33.9,
                "run_speed_rating": 95.0,
                "run_jockey_name": "A Rider",
                "run_trainer_name": "Stable A",
                "run_won": True,
                "run_placed": True,
            },
            {
                "market_id": "1.200",
                "selection_id": 2,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "race_start_time": race_start,
                "seconds_to_start": 600,
                "run_index": 1,
                "distance_m": 1200,
                "surface": "TURF",
                "track": "Randwick",
                "class_index": 7.0,
                "run_date": race_start - pd.Timedelta(days=10),
                "run_finish_pos": 5,
                "run_distance_m": 1400,
                "run_surface": "TURF",
                "run_track": "Rosehill",
                "run_class_index": 7.0,
                "run_sectional_time": 35.6,
                "run_speed_rating": 80.0,
                "run_jockey_name": "B Rider",
                "run_trainer_name": "Stable B",
                "run_won": False,
                "run_placed": False,
            },
            {
                "market_id": "1.200",
                "selection_id": 2,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "race_start_time": race_start,
                "seconds_to_start": 600,
                "run_index": 2,
                "distance_m": 1200,
                "surface": "TURF",
                "track": "Randwick",
                "class_index": 7.0,
                "run_date": race_start - pd.Timedelta(days=24),
                "run_finish_pos": 4,
                "run_distance_m": 1200,
                "run_surface": "TURF",
                "run_track": "Randwick",
                "run_class_index": 7.0,
                "run_sectional_time": 35.2,
                "run_speed_rating": 82.0,
                "run_jockey_name": "B Rider",
                "run_trainer_name": "Stable B",
                "run_won": False,
                "run_placed": False,
            },
        ]
    )
    store = FakeStore(
        snapshots=snapshots,
        results=pd.DataFrame(
            [
                {"market_id": "1.200", "selection_id": 1, "win_flag": 1},
                {"market_id": "1.200", "selection_id": 2, "win_flag": 0},
            ]
        ),
        runners=runners,
        metadata=metadata,
        external_form=external_form,
    )
    features = build_features_from_store(store, cutoff_minutes=10).set_index("selection_id")
    assert features.loc[1, "ext_form_available"] == 1
    assert features.loc[1, "ext_form_runs_last10"] == 2
    assert features.loc[1, "ext_form_win_rate_last10"] > features.loc[2, "ext_form_win_rate_last10"]
    assert features.loc[1, "ext_form_track_starts_last10"] == 2
    assert features.loc[1, "ext_form_horse_jockey_starts_last10"] == 2
    assert features.loc[1, "ext_form_last_speed_rating"] == 92.0


def test_feature_builder_adds_pl_hierarchical_features_from_prior_place_order():
    snapshots = pd.DataFrame(
        [
            {
                "market_id": market_id,
                "selection_id": selection_id,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "seconds_to_start": 600,
                "best_back_price": back_price,
                "best_back_size": 100,
                "best_lay_price": back_price + 0.1,
                "best_lay_size": 110,
                "last_traded_price": back_price + 0.05,
                "total_matched": 1000,
                "runner_status": "ACTIVE",
                "venue": "Randwick",
                "race_start_time": race_start,
                "race_name": market_name,
            }
            for market_id, race_start, market_name in [
                ("1.500", pd.Timestamp("2024-02-01T05:00:00Z"), "Race P2A"),
                ("1.501", pd.Timestamp("2024-02-02T05:00:00Z"), "Race P2B"),
            ]
            for selection_id, back_price in [(1, 2.4), (2, 3.1), (3, 5.0), (4, 9.0)]
        ]
    )
    runners = pd.DataFrame(
        [
            {"market_id": market_id, "selection_id": selection_id, "runner_name": runner_name, "stall_draw": draw}
            for market_id in ["1.500", "1.501"]
            for selection_id, runner_name, draw in [
                (1, "Horse One", 1),
                (2, "Horse Two", 2),
                (3, "Horse Three", 3),
                (4, "Horse Four", 4),
            ]
        ]
    )
    results = pd.DataFrame(
        [
            {"market_id": "1.500", "selection_id": 1, "win_flag": 1, "place_position": 1},
            {"market_id": "1.500", "selection_id": 2, "win_flag": 0, "place_position": 2},
            {"market_id": "1.500", "selection_id": 3, "win_flag": 0, "place_position": 3},
            {"market_id": "1.500", "selection_id": 4, "win_flag": 0, "place_position": 4},
            {"market_id": "1.501", "selection_id": 1, "win_flag": 0, "place_position": 4},
            {"market_id": "1.501", "selection_id": 2, "win_flag": 1, "place_position": 1},
            {"market_id": "1.501", "selection_id": 3, "win_flag": 0, "place_position": 2},
            {"market_id": "1.501", "selection_id": 4, "win_flag": 0, "place_position": 3},
        ]
    )
    store = FakeStore(snapshots=snapshots, results=results, runners=runners)
    features = build_features_from_store(store, cutoff_minutes=10).sort_values(["market_id", "selection_id"])

    race1 = features[features["market_id"] == "1.500"]
    assert race1["pl_eff_rating_pre"].nunique() == 1
    assert race1["pl_eff_rating_pre"].iloc[0] == 1500.0
    assert race1["pl_eff_prob_top2_pre"].round(6).tolist() == [0.5, 0.5, 0.5, 0.5]
    assert race1["pl_eff_prob_top3_pre"].round(6).tolist() == [0.75, 0.75, 0.75, 0.75]

    race2 = features[features["market_id"] == "1.501"].set_index("selection_id")
    assert race2.loc[1, "pl_eff_rating_pre"] > race2.loc[4, "pl_eff_rating_pre"]
    assert race2.loc[1, "pl_eff_prob_win_pre"] > race2.loc[4, "pl_eff_prob_win_pre"]
    assert race2.loc[1, "pl_eff_prob_top3_pre"] > race2.loc[4, "pl_eff_prob_top3_pre"]
    assert race2.loc[1, "pl_component_horse_pre"] > race2.loc[4, "pl_component_horse_pre"]
    assert race2["pl_field_entropy_pre"].between(0.0, 1.0).all()


def test_feature_builder_pl_hierarchical_falls_back_to_winner_when_place_missing():
    snapshots = pd.DataFrame(
        [
            {
                "market_id": market_id,
                "selection_id": selection_id,
                "snapshot_time": race_start - pd.Timedelta(minutes=10),
                "seconds_to_start": 600,
                "best_back_price": back_price,
                "best_back_size": 90,
                "best_lay_price": back_price + 0.1,
                "best_lay_size": 100,
                "last_traded_price": back_price + 0.05,
                "total_matched": 900,
                "runner_status": "ACTIVE",
                "venue": "Flemington",
                "race_start_time": race_start,
                "race_name": race_name,
            }
            for market_id, race_start, race_name in [
                ("1.600", pd.Timestamp("2024-03-01T05:00:00Z"), "Race P2C"),
                ("1.601", pd.Timestamp("2024-03-02T05:00:00Z"), "Race P2D"),
            ]
            for selection_id, back_price in [(11, 2.8), (12, 3.0)]
        ]
    )
    results = pd.DataFrame(
        [
            {"market_id": "1.600", "selection_id": 11, "win_flag": 0},
            {"market_id": "1.600", "selection_id": 12, "win_flag": 1},
            {"market_id": "1.601", "selection_id": 11, "win_flag": 1},
            {"market_id": "1.601", "selection_id": 12, "win_flag": 0},
        ]
    )
    store = FakeStore(snapshots=snapshots, results=results)
    features = build_features_from_store(store, cutoff_minutes=10)
    race2 = features[features["market_id"] == "1.601"].set_index("selection_id")
    assert race2.loc[12, "pl_eff_rating_pre"] > race2.loc[11, "pl_eff_rating_pre"]
    assert race2.loc[12, "pl_eff_prob_win_pre"] > race2.loc[11, "pl_eff_prob_win_pre"]
