import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime

import duckdb
import pandas as pd

from shared.utils.progress import log

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas.sql"


class DuckDBStore:
    """Thin wrapper around DuckDB to persist markets, snapshots, and results so downstream steps stay reproducible."""

    def __init__(self, db_path: Optional[str] = None, auto_recover: bool = True) -> None:
        """Initialise the store and apply schemas so callers can assume tables exist."""
        self.db_path = Path(db_path or os.getenv("DATA_PATH", "data/db.duckdb"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_recover = auto_recover
        log(f"DuckDBStore: using {self.db_path}")
        self._ensure_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection for short-lived operations."""
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        """Apply schema DDL so the database is ready for inserts/updates."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as ddl_file:
            ddl = ddl_file.read()
        try:
            with self._connect() as con:
                con.execute(ddl)
        except Exception as exc:  # pragma: no cover - depends on runtime DB state
            if not self.auto_recover or not self.db_path.exists():
                raise
            backup_path = self._backup_db()
            log(f"DuckDBStore: schema apply failed ({exc}); backed up DB to {backup_path} and retrying.")
            with self._connect() as con:
                con.execute(ddl)

    def _backup_db(self) -> Path:
        """Move the current DB aside so a fresh schema can be created."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backups_dir = self.db_path.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backups_dir / f"{self.db_path.stem}_{timestamp}.duckdb"
        self.db_path.rename(backup_path)
        return backup_path

    def upsert_markets(self, markets: Iterable[Dict[str, Any]]) -> None:
        """Insert or replace market metadata, allowing repeat downloads without duplication."""
        df = pd.DataFrame(list(markets))
        if df.empty:
            return
        cols = [
            "market_id",
            "venue",
            "race_start_time",
            "race_name",
            "market_type",
            "country_code",
            "event_type",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        df = df.drop_duplicates(subset=["market_id"], keep="last")
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO markets (
                    market_id,
                    venue,
                    race_start_time,
                    race_name,
                    market_type,
                    country_code,
                    event_type
                )
                SELECT market_id, venue, race_start_time, race_name, market_type, country_code, event_type FROM df
                ON CONFLICT (market_id) DO UPDATE
                SET venue=excluded.venue,
                    race_start_time=excluded.race_start_time,
                    race_name=excluded.race_name,
                    market_type=excluded.market_type,
                    country_code=excluded.country_code,
                    event_type=excluded.event_type
                """
            )

    def upsert_runners(self, runners: Iterable[Dict[str, Any]]) -> None:
        """Insert or replace runner metadata for each market."""
        df = pd.DataFrame(list(runners))
        if df.empty:
            return
        cols = ["market_id", "selection_id", "runner_name", "stall_draw"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        df = df.drop_duplicates(subset=["market_id", "selection_id"], keep="last")
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO runners AS r
                SELECT market_id, selection_id, runner_name, stall_draw FROM df
                ON CONFLICT (market_id, selection_id) DO UPDATE
                SET runner_name=excluded.runner_name,
                    stall_draw=excluded.stall_draw
                """
            )

    def append_runner_metadata_snapshots(self, snapshots: Iterable[Dict[str, Any]]) -> None:
        """Persist point-in-time runner metadata so form features can be rebuilt exactly as seen pre-race."""
        df = pd.DataFrame(list(snapshots))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "snapshot_time",
            "race_start_time",
            "seconds_to_start",
            "source",
            "runner_name",
            "jockey_name",
            "trainer_name",
            "age",
            "official_rating",
            "adjusted_rating",
            "days_since_last_run",
            "weight_value",
            "weight_units",
            "jockey_claim",
            "stall_draw",
            "form_string",
            "metadata",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df["metadata"] = df["metadata"].map(
            lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        df = df[cols]
        df = df.drop_duplicates(
            subset=["market_id", "selection_id", "snapshot_time", "source"],
            keep="last",
        )
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO runner_metadata_snapshots (
                    market_id,
                    selection_id,
                    snapshot_time,
                    race_start_time,
                    seconds_to_start,
                    source,
                    runner_name,
                    jockey_name,
                    trainer_name,
                    age,
                    official_rating,
                    adjusted_rating,
                    days_since_last_run,
                    weight_value,
                    weight_units,
                    jockey_claim,
                    stall_draw,
                    form_string,
                    metadata
                )
                SELECT * FROM df
                """
            )

    def append_external_runner_form_runs(self, rows: Iterable[Dict[str, Any]]) -> None:
        """Persist point-in-time external form runs so last-N/sectional features are reproducible at cutoff."""
        df = pd.DataFrame(list(rows))
        if df.empty:
            return
        cols = [
            "source",
            "market_id",
            "selection_id",
            "snapshot_time",
            "race_start_time",
            "seconds_to_start",
            "run_index",
            "runner_name",
            "horse_name",
            "jockey_name",
            "trainer_name",
            "track",
            "surface",
            "distance_m",
            "class_label",
            "class_index",
            "track_condition",
            "run_date",
            "run_finish_pos",
            "run_field_size",
            "run_distance_m",
            "run_surface",
            "run_track",
            "run_class_label",
            "run_class_index",
            "run_track_condition",
            "run_sectional_time",
            "run_speed_rating",
            "run_weight_value",
            "run_barrier",
            "run_jockey_name",
            "run_trainer_name",
            "run_won",
            "run_placed",
            "metadata",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df["metadata"] = df["metadata"].map(
            lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        df = df[cols]
        df = df.drop_duplicates(
            subset=["source", "market_id", "selection_id", "snapshot_time", "run_index"],
            keep="last",
        )
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO external_runner_form_runs (
                    source,
                    market_id,
                    selection_id,
                    snapshot_time,
                    race_start_time,
                    seconds_to_start,
                    run_index,
                    runner_name,
                    horse_name,
                    jockey_name,
                    trainer_name,
                    track,
                    surface,
                    distance_m,
                    class_label,
                    class_index,
                    track_condition,
                    run_date,
                    run_finish_pos,
                    run_field_size,
                    run_distance_m,
                    run_surface,
                    run_track,
                    run_class_label,
                    run_class_index,
                    run_track_condition,
                    run_sectional_time,
                    run_speed_rating,
                    run_weight_value,
                    run_barrier,
                    run_jockey_name,
                    run_trainer_name,
                    run_won,
                    run_placed,
                    metadata
                )
                SELECT * FROM df
                """
            )

    def append_snapshots(self, snapshots: Iterable[Dict[str, Any]]) -> None:
        """Append snapshot rows captured at fixed offsets; rows are immutable historical observations."""
        df = pd.DataFrame(list(snapshots))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "snapshot_time",
            "seconds_to_start",
            "best_back_price",
            "best_back_size",
            "best_lay_price",
            "best_lay_size",
            "last_traded_price",
            "total_matched",
            "runner_status",
            "venue",
            "race_start_time",
            "race_name",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO snapshots (
                    market_id,
                    selection_id,
                    snapshot_time,
                    seconds_to_start,
                    best_back_price,
                    best_back_size,
                    best_lay_price,
                    best_lay_size,
                    last_traded_price,
                    total_matched,
                    runner_status,
                    venue,
                    race_start_time,
                    race_name
                )
                SELECT * FROM df
                """
            )

    def upsert_results(self, results: Iterable[Dict[str, Any]]) -> None:
        """Store official outcomes (win flag, BSP) for model targets and backtesting."""
        df = pd.DataFrame(list(results))
        if df.empty:
            return
        df = df.drop_duplicates(subset=["market_id", "selection_id"], keep="last")
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO results AS r
                SELECT * FROM df
                ON CONFLICT (market_id, selection_id) DO UPDATE
                SET win_flag=excluded.win_flag,
                    bsp=excluded.bsp,
                    place_position=excluded.place_position
                """
            )

    def record_bets(self, bets: Iterable[Dict[str, Any]], run_id: Optional[str] = None) -> None:
        """Persist simulated bets so backtests are auditable and traceable to a run."""
        df = pd.DataFrame(list(bets))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "run_id",
            "bet_time",
            "stake",
            "price",
            "bet_type",
            "expected_value",
            "commission_rate",
            "result_profit",
        ]
        df["run_id"] = run_id
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        with self._connect() as con:
            if run_id:
                # Keep run-level reports deterministic when the same run_id is backtested repeatedly.
                con.execute("DELETE FROM bets WHERE run_id = ?", [run_id])
            con.register("df", df)
            con.execute(
                """
                INSERT INTO bets (
                    market_id,
                    selection_id,
                    run_id,
                    bet_time,
                    stake,
                    price,
                    bet_type,
                    expected_value,
                    commission_rate,
                    result_profit
                )
                SELECT * FROM df
                """
            )

    def record_model_run(
        self,
        model_path: str,
        calibrator_path: Optional[str],
        cutoff_minutes: int,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
    ) -> None:
        """Log model artefact locations and metrics for traceability."""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO model_runs (model_path, calibrator_path, cutoff_minutes, metrics, notes, run_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [model_path, calibrator_path, cutoff_minutes, metrics or {}, notes, run_id],
            )

    def record_oof_predictions(
        self,
        predictions: Iterable[Dict[str, Any]],
        run_id: str,
        cutoff_minutes: int,
    ) -> None:
        """Persist out-of-fold predictions so backtests can avoid in-sample bias."""
        df = pd.DataFrame(list(predictions))
        if df.empty:
            return
        df["run_id"] = run_id
        df["cutoff_minutes"] = cutoff_minutes
        cols = ["run_id", "cutoff_minutes", "market_id", "selection_id", "p_hat"]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        df = df.dropna(subset=["market_id", "selection_id", "p_hat"])
        if df.empty:
            return
        with self._connect() as con:
            con.execute(
                "DELETE FROM oof_predictions WHERE run_id = ? AND cutoff_minutes = ?",
                [run_id, cutoff_minutes],
            )
            con.register("df", df)
            con.execute(
                """
                INSERT INTO oof_predictions (run_id, cutoff_minutes, market_id, selection_id, p_hat)
                SELECT run_id, cutoff_minutes, market_id, selection_id, p_hat FROM df
                """
            )

    def load_oof_predictions(self, run_id: str, cutoff_minutes: int) -> pd.DataFrame:
        """Load stored out-of-fold predictions for a specific training run."""
        with self._connect() as con:
            return con.execute(
                """
                SELECT market_id, selection_id, p_hat
                FROM oof_predictions
                WHERE run_id = ? AND cutoff_minutes = ?
                """,
                [run_id, cutoff_minutes],
            ).df()

    def append_tab_quotes(self, quotes: Iterable[Dict[str, Any]]) -> None:
        """Persist manually captured TAB display quotes so translation models can learn executable odds drift."""
        df = pd.DataFrame(list(quotes))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "quote_time",
            "source_channel",
            "product_type",
            "display_odds",
            "notes",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO tab_quotes (
                    market_id,
                    selection_id,
                    quote_time,
                    source_channel,
                    product_type,
                    display_odds,
                    notes
                )
                SELECT * FROM df
                """
            )

    def append_tab_executions(self, executions: Iterable[Dict[str, Any]]) -> None:
        """Persist accepted/repriced/refused TAB execution outcomes for settlement and translation supervision."""
        df = pd.DataFrame(list(executions))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "placed_time",
            "source_channel",
            "product_type",
            "stake",
            "accepted_odds",
            "was_repriced",
            "was_refused",
            "notes",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO tab_executions (
                    market_id,
                    selection_id,
                    placed_time,
                    source_channel,
                    product_type,
                    stake,
                    accepted_odds,
                    was_repriced,
                    was_refused,
                    notes
                )
                SELECT * FROM df
                """
            )

    def load_tab_quotes(self) -> pd.DataFrame:
        """Load captured TAB display quotes so translation model training can reuse persisted manual labels."""
        with self._connect() as con:
            return con.execute("SELECT * FROM tab_quotes").df()

    def load_tab_executions(self) -> pd.DataFrame:
        """Load TAB execution outcomes so realised slippage and refusal rates can be monitored over time."""
        with self._connect() as con:
            return con.execute("SELECT * FROM tab_executions").df()

    def load_snapshots(self) -> pd.DataFrame:
        """Return all stored snapshots for downstream feature construction."""
        with self._connect() as con:
            return con.execute("SELECT * FROM snapshots").df()

    def load_snapshots_for_cutoff(
        self,
        cutoff_minutes: int,
        market_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return cutoff-filtered snapshots for feature building to reduce Python-side scan volume."""
        cutoff_seconds = max(0, int(cutoff_minutes) * 60)
        market_id_values = list(market_ids or [])
        market_id_values = [str(value) for value in market_id_values if str(value).strip()]
        market_filter_sql = ""
        params: list[Any] = [cutoff_seconds]
        if market_id_values:
            placeholders = ", ".join(["?"] * len(market_id_values))
            market_filter_sql = f" AND s.market_id IN ({placeholders})"
            params.extend(market_id_values)
        with self._connect() as con:
            return con.execute(
                f"""
                SELECT
                    s.market_id,
                    s.selection_id,
                    s.snapshot_time,
                    s.seconds_to_start,
                    s.best_back_price,
                    s.best_back_size,
                    s.best_lay_price,
                    s.best_lay_size,
                    s.last_traded_price,
                    s.total_matched,
                    s.runner_status,
                    s.venue,
                    s.race_start_time,
                    s.race_name,
                    m.market_type
                FROM snapshots AS s
                LEFT JOIN markets AS m
                  ON m.market_id = s.market_id
                WHERE s.snapshot_time IS NOT NULL
                  AND s.race_start_time IS NOT NULL
                  AND s.snapshot_time < s.race_start_time
                  AND s.seconds_to_start >= ?
                  {market_filter_sql}
                """,
                params,
            ).df()

    def load_runner_metadata_for_cutoff(
        self,
        cutoff_minutes: int,
        market_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Load the latest pre-race metadata row per runner at/after cutoff for leakage-safe feature joins."""
        cutoff_seconds = max(0, int(cutoff_minutes) * 60)
        market_id_values = [str(value) for value in (market_ids or []) if str(value).strip()]
        market_filter_sql = ""
        params: list[Any] = [cutoff_seconds]
        if market_id_values:
            placeholders = ", ".join(["?"] * len(market_id_values))
            market_filter_sql = f" AND market_id IN ({placeholders})"
            params.extend(market_id_values)
        with self._connect() as con:
            return con.execute(
                f"""
                WITH eligible AS (
                    SELECT
                        market_id,
                        selection_id,
                        snapshot_time,
                        race_start_time,
                        seconds_to_start,
                        source,
                        runner_name,
                        jockey_name,
                        trainer_name,
                        age,
                        official_rating,
                        adjusted_rating,
                        days_since_last_run,
                        weight_value,
                        weight_units,
                        jockey_claim,
                        stall_draw,
                        form_string,
                        metadata,
                        ROW_NUMBER() OVER (
                            PARTITION BY market_id, selection_id
                            ORDER BY seconds_to_start ASC, snapshot_time DESC
                        ) AS row_num
                    FROM runner_metadata_snapshots
                    WHERE snapshot_time IS NOT NULL
                      AND race_start_time IS NOT NULL
                      AND snapshot_time < race_start_time
                      AND seconds_to_start >= ?
                      {market_filter_sql}
                )
                SELECT
                    market_id,
                    selection_id,
                    snapshot_time,
                    race_start_time,
                    seconds_to_start,
                    source,
                    runner_name,
                    jockey_name,
                    trainer_name,
                    age,
                    official_rating,
                    adjusted_rating,
                    days_since_last_run,
                    weight_value,
                    weight_units,
                    jockey_claim,
                    stall_draw,
                    form_string,
                    metadata
                FROM eligible
                WHERE row_num = 1
                """,
                params,
            ).df()

    def load_runner_metadata_completeness(self) -> pd.DataFrame:
        """Summarize per-day metadata coverage so drift in optional Betfair fields is easy to monitor."""
        with self._connect() as con:
            return con.execute(
                """
                SELECT
                    CAST(snapshot_time AS DATE) AS snapshot_date,
                    COUNT(*) AS row_count,
                    AVG(CASE WHEN jockey_name IS NOT NULL THEN 1.0 ELSE 0.0 END) AS jockey_name_coverage,
                    AVG(CASE WHEN trainer_name IS NOT NULL THEN 1.0 ELSE 0.0 END) AS trainer_name_coverage,
                    AVG(CASE WHEN age IS NOT NULL THEN 1.0 ELSE 0.0 END) AS age_coverage,
                    AVG(CASE WHEN official_rating IS NOT NULL THEN 1.0 ELSE 0.0 END) AS official_rating_coverage,
                    AVG(CASE WHEN adjusted_rating IS NOT NULL THEN 1.0 ELSE 0.0 END) AS adjusted_rating_coverage,
                    AVG(CASE WHEN days_since_last_run IS NOT NULL THEN 1.0 ELSE 0.0 END) AS days_since_last_run_coverage,
                    AVG(CASE WHEN weight_value IS NOT NULL THEN 1.0 ELSE 0.0 END) AS weight_value_coverage,
                    AVG(CASE WHEN jockey_claim IS NOT NULL THEN 1.0 ELSE 0.0 END) AS jockey_claim_coverage,
                    AVG(CASE WHEN stall_draw IS NOT NULL THEN 1.0 ELSE 0.0 END) AS stall_draw_coverage,
                    AVG(CASE WHEN form_string IS NOT NULL THEN 1.0 ELSE 0.0 END) AS form_string_coverage
                FROM runner_metadata_snapshots
                GROUP BY 1
                ORDER BY 1 DESC
                """
            ).df()

    def load_external_runner_form_for_cutoff(
        self,
        cutoff_minutes: int,
        market_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Load latest eligible external run-history rows per runner+run index at or before inference cutoff."""
        cutoff_seconds = max(0, int(cutoff_minutes) * 60)
        market_id_values = [str(value) for value in (market_ids or []) if str(value).strip()]
        market_filter_sql = ""
        params: list[Any] = [cutoff_seconds]
        if market_id_values:
            placeholders = ", ".join(["?"] * len(market_id_values))
            market_filter_sql = f" AND market_id IN ({placeholders})"
            params.extend(market_id_values)
        with self._connect() as con:
            return con.execute(
                f"""
                WITH eligible AS (
                    SELECT
                        source,
                        market_id,
                        selection_id,
                        snapshot_time,
                        race_start_time,
                        seconds_to_start,
                        run_index,
                        runner_name,
                        horse_name,
                        jockey_name,
                        trainer_name,
                        track,
                        surface,
                        distance_m,
                        class_label,
                        class_index,
                        track_condition,
                        run_date,
                        run_finish_pos,
                        run_field_size,
                        run_distance_m,
                        run_surface,
                        run_track,
                        run_class_label,
                        run_class_index,
                        run_track_condition,
                        run_sectional_time,
                        run_speed_rating,
                        run_weight_value,
                        run_barrier,
                        run_jockey_name,
                        run_trainer_name,
                        run_won,
                        run_placed,
                        metadata,
                        ROW_NUMBER() OVER (
                            PARTITION BY market_id, selection_id, run_index
                            ORDER BY seconds_to_start ASC, snapshot_time DESC
                        ) AS row_num
                    FROM external_runner_form_runs
                    WHERE snapshot_time IS NOT NULL
                      AND race_start_time IS NOT NULL
                      AND snapshot_time < race_start_time
                      AND seconds_to_start >= ?
                      {market_filter_sql}
                )
                SELECT
                    source,
                    market_id,
                    selection_id,
                    snapshot_time,
                    race_start_time,
                    seconds_to_start,
                    run_index,
                    runner_name,
                    horse_name,
                    jockey_name,
                    trainer_name,
                    track,
                    surface,
                    distance_m,
                    class_label,
                    class_index,
                    track_condition,
                    run_date,
                    run_finish_pos,
                    run_field_size,
                    run_distance_m,
                    run_surface,
                    run_track,
                    run_class_label,
                    run_class_index,
                    run_track_condition,
                    run_sectional_time,
                    run_speed_rating,
                    run_weight_value,
                    run_barrier,
                    run_jockey_name,
                    run_trainer_name,
                    run_won,
                    run_placed,
                    metadata
                FROM eligible
                WHERE row_num = 1
                """,
                params,
            ).df()

    def load_external_runner_form_completeness(self) -> pd.DataFrame:
        """Summarize per-day external-form coverage so upstream provider drift can be monitored."""
        with self._connect() as con:
            return con.execute(
                """
                SELECT
                    CAST(snapshot_time AS DATE) AS snapshot_date,
                    COUNT(*) AS row_count,
                    AVG(CASE WHEN run_date IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_date_coverage,
                    AVG(CASE WHEN run_finish_pos IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_finish_pos_coverage,
                    AVG(CASE WHEN run_distance_m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_distance_coverage,
                    AVG(CASE WHEN run_surface IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_surface_coverage,
                    AVG(CASE WHEN run_track IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_track_coverage,
                    AVG(CASE WHEN run_sectional_time IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_sectional_coverage,
                    AVG(CASE WHEN run_speed_rating IS NOT NULL THEN 1.0 ELSE 0.0 END) AS run_speed_rating_coverage
                FROM external_runner_form_runs
                GROUP BY 1
                ORDER BY 1 DESC
                """
            ).df()

    def load_results(self) -> pd.DataFrame:
        """Return stored results to build targets."""
        with self._connect() as con:
            return con.execute("SELECT * FROM results").df()

    def load_markets(self) -> pd.DataFrame:
        """Return stored markets for reporting and joins."""
        with self._connect() as con:
            return con.execute("SELECT * FROM markets").df()

    def load_runners(self) -> pd.DataFrame:
        """Return stored runners for reporting and joins."""
        with self._connect() as con:
            return con.execute("SELECT * FROM runners").df()

    def max_race_start_time(self) -> Optional[pd.Timestamp]:
        """Return the latest race_start_time from snapshots, falling back to markets."""
        with self._connect() as con:
            result = con.execute("SELECT MAX(race_start_time) FROM snapshots").fetchone()
            if result and result[0]:
                return result[0]
            result = con.execute("SELECT MAX(race_start_time) FROM markets").fetchone()
        if not result:
            return None
        return result[0]

    def max_snapshot_time(self) -> Optional[pd.Timestamp]:
        """Return the latest snapshot timestamp so CLI workflows can detect stale historic data."""
        with self._connect() as con:
            result = con.execute("SELECT MAX(snapshot_time) FROM snapshots").fetchone()
        if not result:
            return None
        return result[0]

    def table_row_count(self, table: str) -> int:
        """Return table row count so ingest steps can skip reprocessing when data already exists."""
        with self._connect() as con:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def has_data(self, table: str = "snapshots") -> bool:
        """Return True if the table has at least one row, to short-circuit expensive ingests."""
        return self.table_row_count(table) > 0
