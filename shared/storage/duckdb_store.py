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

    def record_bets(self, bets: Iterable[Dict[str, Any]]) -> None:
        """Persist simulated bets so backtests are auditable."""
        df = pd.DataFrame(list(bets))
        if df.empty:
            return
        cols = [
            "market_id",
            "selection_id",
            "bet_time",
            "stake",
            "price",
            "bet_type",
            "expected_value",
            "commission_rate",
            "result_profit",
        ]
        for col in cols:
            if col not in df.columns:
                df[col] = None
        df = df[cols]
        with self._connect() as con:
            con.register("df", df)
            con.execute(
                """
                INSERT INTO bets (
                    market_id,
                    selection_id,
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
    ) -> None:
        """Log model artefact locations and metrics for traceability."""
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO model_runs (model_path, calibrator_path, cutoff_minutes, metrics, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                [model_path, calibrator_path, cutoff_minutes, metrics or {}, notes],
            )

    def load_snapshots(self) -> pd.DataFrame:
        """Return all stored snapshots for downstream feature construction."""
        with self._connect() as con:
            return con.execute("SELECT * FROM snapshots").df()

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

    def load_runners(self) -> pd.DataFrame:
        """Return stored runners for joins."""
        with self._connect() as con:
            return con.execute("SELECT * FROM runners").df()

    def table_row_count(self, table: str) -> int:
        """Return table row count so ingest steps can skip reprocessing when data already exists."""
        with self._connect() as con:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def has_data(self, table: str = "snapshots") -> bool:
        """Return True if the table has at least one row, to short-circuit expensive ingests."""
        return self.table_row_count(table) > 0
