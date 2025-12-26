import argparse
import bz2
import concurrent.futures
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log

ALLOWED_EVENT_TYPE_IDS = {"7"}
ALLOWED_COUNTRIES = {"AU"}
ALLOWED_MARKET_TYPES = {"WIN"}


def _load_member_as_df(tar: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
    """Read a tar member into a DataFrame, supporting CSV and Parquet."""
    with tar.extractfile(member) as fh:  # type: ignore[arg-type]
        if fh is None:
            return pd.DataFrame()
        if member.name.endswith(".parquet"):
            return pd.read_parquet(fh)  # type: ignore[arg-type]
        return pd.read_csv(fh)


def _emit(store: DuckDBStore, table: str, df: pd.DataFrame) -> None:
    """Dispatch DataFrame rows to the appropriate store method."""
    if df.empty:
        return
    if table == "markets":
        store.upsert_markets(df.to_dict(orient="records"))
    elif table == "runners":
        store.upsert_runners(df.to_dict(orient="records"))
    elif table == "snapshots":
        store.append_snapshots(df.to_dict(orient="records"))
    elif table == "results":
        store.upsert_results(df.to_dict(orient="records"))


def _parse_iso(dt_str: str) -> datetime:
    """Parse ISO datetime with Z."""
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def _is_market_allowed(md: Dict) -> bool:
    """Return True when market matches the AU horse racing win filters."""
    event_type = str(md.get("eventTypeId") or "").strip()
    country = (md.get("countryCode") or "").upper()
    market_type = (md.get("marketType") or "").upper()
    return (
        event_type in ALLOWED_EVENT_TYPE_IDS
        and country in ALLOWED_COUNTRIES
        and market_type in ALLOWED_MARKET_TYPES
    )


def _market_definition_to_rows(
    market_id: str, md: Dict, filter_au_win: bool
) -> tuple[Dict | None, List[Dict], List[Dict], Dict]:
    """Convert marketDefinition into market/runner/result rows plus metadata for snapshots."""
    if filter_au_win and not _is_market_allowed(md):
        return None, [], [], {"eligible": False}
    market_time = _parse_iso(md["marketTime"]) if md.get("marketTime") else None
    market_row = {
        "market_id": market_id,
        "venue": md.get("venue") or md.get("eventName"),
        "race_start_time": market_time,
        "race_name": md.get("name"),
        "country_code": md.get("countryCode"),
        "event_type": md.get("eventTypeId"),
    }
    runners = [
        {
            "market_id": market_id,
            "selection_id": r["id"],
            "runner_name": r.get("name"),
            "stall_draw": r.get("stallDraw"),
        }
        for r in md.get("runners", [])
    ]
    results = []
    if md.get("status") == "CLOSED":
        for r in md.get("runners", []):
            results.append(
                {
                    "market_id": market_id,
                    "selection_id": r["id"],
                    "win_flag": r.get("status") == "WINNER",
                    "bsp": r.get("bsp"),
                    "place_position": 1 if r.get("status") == "WINNER" else None,
                }
            )
    meta = {
        "market_time": market_time,
        "venue": market_row["venue"],
        "race_name": market_row["race_name"],
        "country_code": market_row["country_code"],
        "eligible": True,
    }
    return market_row, runners, results, meta


def _handle_market_definition(market_id: str, md: Dict, store: DuckDBStore, filter_au_win: bool) -> Dict:
    """Convert marketDefinition into metadata and upsert market/runner records."""
    market_row, runners, results, meta = _market_definition_to_rows(market_id, md, filter_au_win)
    if market_row:
        store.upsert_markets([market_row])
        if runners:
            store.upsert_runners(runners)
        if results:
            store.upsert_results(results)
    return meta


def _append_snapshots_from_rc(
    market_id: str,
    rc_list: List[Dict],
    pt_ms: int,
    meta: Dict,
    buffer: List[Dict],
) -> None:
    """Transform runner changes into snapshot rows."""
    snapshot_time = datetime.fromtimestamp(pt_ms / 1000, tz=timezone.utc) if pt_ms else None
    market_time = meta.get("market_time")
    seconds_to_start = (
        (market_time - snapshot_time).total_seconds() if market_time and snapshot_time else None
    )
    if seconds_to_start is None:
        return
    for rc in rc_list:
        ex = rc.get("ex", {})
        atb = ex.get("availableToBack", [])
        atl = ex.get("availableToLay", [])
        best_back = atb[0] if atb else None
        best_lay = atl[0] if atl else None
        tv = rc.get("tv")
        if isinstance(tv, list):
            total_matched = sum(level.get("size", 0) for level in tv)
        else:
            total_matched = tv
        buffer.append(
            {
                "market_id": market_id,
                "selection_id": rc.get("id"),
                "snapshot_time": snapshot_time,
                "seconds_to_start": seconds_to_start,
                "best_back_price": best_back.get("price") if best_back else rc.get("ltp"),
                "best_back_size": best_back.get("size") if best_back else None,
                "best_lay_price": best_lay.get("price") if best_lay else rc.get("ltp"),
                "best_lay_size": best_lay.get("size") if best_lay else None,
                "last_traded_price": rc.get("ltp"),
                "total_matched": total_matched,
                "runner_status": rc.get("status"),
                "venue": meta.get("venue"),
                "race_start_time": market_time,
                "race_name": meta.get("race_name"),
            }
        )


def _parse_member_from_offset(task: tuple[str, int, int, str, bool]) -> Dict[str, List[Dict]]:
    """Parse a single .bz2 member from a tar file using byte offsets (worker-safe)."""
    archive_path, offset, size, name, filter_au_win = task
    market_meta: Dict[str, Dict] = {}
    market_rows: List[Dict] = []
    runner_rows: List[Dict] = []
    result_rows: List[Dict] = []
    snapshot_rows: List[Dict] = []

    with open(archive_path, "rb") as handle:
        handle.seek(offset)
        remaining = size
        decompressor = bz2.BZ2Decompressor()
        buffer = ""
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            data = decompressor.decompress(chunk)
            if not data:
                continue
            text = data.decode("utf-8", errors="ignore")
            buffer += text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                pt_ms = msg.get("pt")
                for mc in msg.get("mc", []):
                    market_id = mc.get("id")
                    if not market_id:
                        continue
                    if "marketDefinition" in mc:
                        market_row, runners, results, meta = _market_definition_to_rows(
                            market_id, mc["marketDefinition"], filter_au_win
                        )
                        if market_row:
                            market_rows.append(market_row)
                            runner_rows.extend(runners)
                            result_rows.extend(results)
                        market_meta[market_id] = meta
                    if "rc" in mc:
                        meta = market_meta.get(market_id, {})
                        if not meta.get("eligible"):
                            continue
                        _append_snapshots_from_rc(market_id, mc["rc"], pt_ms, meta, snapshot_rows)
        if buffer.strip():
            try:
                msg = json.loads(buffer)
                pt_ms = msg.get("pt")
                for mc in msg.get("mc", []):
                    market_id = mc.get("id")
                    if not market_id:
                        continue
                    if "marketDefinition" in mc:
                        market_row, runners, results, meta = _market_definition_to_rows(
                            market_id, mc["marketDefinition"], filter_au_win
                        )
                        if market_row:
                            market_rows.append(market_row)
                            runner_rows.extend(runners)
                            result_rows.extend(results)
                        market_meta[market_id] = meta
                    if "rc" in mc:
                        meta = market_meta.get(market_id, {})
                        if not meta.get("eligible"):
                            continue
                        _append_snapshots_from_rc(market_id, mc["rc"], pt_ms, meta, snapshot_rows)
            except json.JSONDecodeError:
                pass

    return {
        "market_rows": market_rows,
        "runner_rows": runner_rows,
        "result_rows": result_rows,
        "snapshot_rows": snapshot_rows,
        "name": name,
    }


def _ingest_bz_stream_archive(
    archive_path: Path,
    store: DuckDBStore,
    flush_every: int = 5000,
    progress_every: int = 100,
    workers: int | None = None,
    filter_au_win: bool = False,
) -> Dict[str, int]:
    """Parse Betfair historical stream (.bz2 inside tar) and write to DuckDB."""
    counts = {"snapshots": 0, "markets": 0, "runners": 0, "results": 0, "skipped_markets": 0}
    market_meta: Dict[str, Dict] = {}
    buffer: List[Dict] = []
    with tarfile.open(archive_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".bz2")]
        log(f"Found {len(members)} stream files in archive.")
        if workers is None:
            cpu_count = os.cpu_count() or 2
            workers = max(1, min(cpu_count - 1, len(members)))
        if workers > 1:
            log(f"Ingest: using {workers} workers (auto).")
            tasks = [(str(archive_path), m.offset_data, m.size, m.name, filter_au_win) for m in members]
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for idx, result in enumerate(executor.map(_parse_member_from_offset, tasks), start=1):
                    market_rows = result["market_rows"]
                    runner_rows = result["runner_rows"]
                    result_rows = result["result_rows"]
                    snapshot_rows = result["snapshot_rows"]
                    if market_rows:
                        store.upsert_markets(market_rows)
                        counts["markets"] += len(market_rows)
                    if runner_rows:
                        store.upsert_runners(runner_rows)
                        counts["runners"] += len(runner_rows)
                    if result_rows:
                        store.upsert_results(result_rows)
                        counts["results"] += len(result_rows)
                    if snapshot_rows:
                        for i in range(0, len(snapshot_rows), flush_every):
                            chunk = snapshot_rows[i : i + flush_every]
                            store.append_snapshots(chunk)
                            counts["snapshots"] += len(chunk)
                    if progress_every and idx % progress_every == 0:
                        log(
                            f"Processed {idx}/{len(tasks)} files; snapshots={counts['snapshots']} markets={counts['markets']}"
                        )
            return counts
        for idx, member in enumerate(members, start=1):
            if progress_every and idx % progress_every == 0:
                log(
                    f"Processing file {idx}/{len(members)}; snapshots={counts['snapshots']} markets={counts['markets']}"
                )
            with tar.extractfile(member) as fh:  # type: ignore[arg-type]
                if fh is None:
                    continue
                with bz2.open(fh, "rt") as bz:
                    for line in bz:
                        if not line.strip():
                            continue
                        msg = json.loads(line)
                        pt_ms = msg.get("pt")
                        for mc in msg.get("mc", []):
                            market_id = mc.get("id")
                            if not market_id:
                                continue
                            if "marketDefinition" in mc:
                                meta = _handle_market_definition(
                                    market_id, mc["marketDefinition"], store, filter_au_win
                                )
                                market_meta[market_id] = meta
                                if meta.get("eligible"):
                                    counts["markets"] += 1
                                    counts["runners"] += len(mc["marketDefinition"].get("runners", []))
                                    if mc["marketDefinition"].get("status") == "CLOSED":
                                        counts["results"] += len(mc["marketDefinition"].get("runners", []))
                                else:
                                    counts["skipped_markets"] += 1
                            if "rc" in mc:
                                meta = market_meta.get(market_id, {})
                                if filter_au_win and not meta.get("eligible"):
                                    continue
                                _append_snapshots_from_rc(market_id, mc["rc"], pt_ms, meta, buffer)
                                if len(buffer) >= flush_every:
                                    store.append_snapshots(buffer)
                                    counts["snapshots"] += len(buffer)
                                    log(f"Flushed {len(buffer)} snapshots (total {counts['snapshots']}).")
                                    buffer = []
    if buffer:
        store.append_snapshots(buffer)
        counts["snapshots"] += len(buffer)
        log(f"Final flush: {len(buffer)} snapshots (total {counts['snapshots']}).")
    return counts


def _ingest_tabular_archive(archive_path: Path, store: DuckDBStore, progress_every: int = 10) -> Dict[str, int]:
    """Load CSV/Parquet tables from tar into DuckDB (legacy path)."""
    ingested = {}
    with tarfile.open(archive_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        log(f"Found {len(members)} files in archive.")
        for idx, member in enumerate(members, start=1):
            if progress_every and idx % progress_every == 0:
                log(f"Processing file {idx}/{len(members)}; rows so far={ingested}")
            name = Path(member.name).name.lower()
            if name.startswith("markets"):
                table = "markets"
            elif name.startswith("runners"):
                table = "runners"
            elif name.startswith("snapshots"):
                table = "snapshots"
            elif name.startswith("results"):
                table = "results"
            else:
                continue
            df = _load_member_as_df(tar, member)
            _emit(store, table, df)
            ingested[table] = ingested.get(table, 0) + len(df)
    return ingested


def main() -> None:
    """Ingest historic Betfair data (either tabular snapshots/results or raw .bz2 stream files) into DuckDB."""
    parser = argparse.ArgumentParser(
        description="Ingest market data archive. Supports Betfair historical stream .bz2 inside tar, or CSV/Parquet tables."
    )
    parser.add_argument("--archive", type=str, default="data/data.tar", help="Path to tar archive.")
    parser.add_argument("--flush-every", type=int, default=5000, help="Number of snapshots to buffer before writing.")
    parser.add_argument("--progress-every", type=int, default=100, help="File progress print frequency.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers for stream ingest.")
    parser.add_argument(
        "--filter-au-win",
        action="store_true",
        help="If set, only keep AU horse racing WIN markets during ingest.",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-ingest even if snapshots already exist in DuckDB.",
    )
    args = parser.parse_args()

    log(
        f"Ingest: starting for {args.archive} (filter_au_win={args.filter_au_win})"
    )
    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    store = DuckDBStore()

    if not args.force_ingest and store.has_data("snapshots"):
        existing = store.table_row_count("snapshots")
        log(
            f"Ingest: snapshots already present ({existing} rows); skipping ingest. Use --force-ingest to re-run."
        )
        return

    counts = ingest_archive_file(
        archive_path,
        store,
        flush_every=args.flush_every,
        progress_every=args.progress_every,
        workers=args.workers,
        filter_au_win=args.filter_au_win,
    )
    log(f"Ingested rows: {counts}")
    log("Ingest: complete.")


def ingest_archive_file(
    archive_path: Path,
    store: DuckDBStore,
    flush_every: int = 5000,
    progress_every: int = 100,
    workers: int | None = None,
    filter_au_win: bool = False,
) -> Dict[str, int]:
    """Public helper to ingest a tar archive into DuckDB (stream or tabular)."""
    # Detect type by looking for .bz2 members
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        has_bz2 = any(m.name.endswith(".bz2") for m in members if m.isfile())

    if has_bz2:
        log("Detected Betfair stream archive (.bz2).")
        return _ingest_bz_stream_archive(
            archive_path,
            store,
            flush_every=flush_every,
            progress_every=progress_every,
            workers=workers,
            filter_au_win=filter_au_win,
        )
    log("Detected tabular archive (CSV/Parquet).")
    return _ingest_tabular_archive(archive_path, store, progress_every=progress_every)


if __name__ == "__main__":
    main()
