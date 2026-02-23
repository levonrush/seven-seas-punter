import argparse
import datetime as dt
from typing import List

from shared.betfair.client import BetfairClient
from shared.features.builder import SNAPSHOT_OFFSETS_MIN
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log


def capture_snapshots(client: BetfairClient, market_ids: List[str], offsets: List[int]) -> list[dict]:
    """Fetch market books at requested offsets (mocked in dry-run) and stamp timing metadata."""
    snapshots: list[dict] = []
    for offset in offsets:
        books = client.fetch_market_books(market_ids)
        for book in books:
            book["seconds_to_start"] = offset * 60
            book["snapshot_time"] = book["race_start_time"] - dt.timedelta(minutes=offset)
            snapshots.append(book)
    return snapshots


def _coerce_utc_datetime(value: object) -> dt.datetime | None:
    """Normalize date inputs so metadata snapshots use stable UTC timestamps across API payload variants."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    return None


def _to_int(value: object) -> int | None:
    """Convert optional numeric payloads to integers so schema writes are deterministic."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    """Convert optional numeric payloads to floats so schema writes are deterministic."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_runner_metadata_snapshots(
    markets: List[dict],
    source: str = "list_market_catalogue",
    captured_at: dt.datetime | None = None,
) -> list[dict]:
    """Extract point-in-time runner metadata rows so form features can be rebuilt exactly at inference cutoff."""
    snapshot_time = captured_at or dt.datetime.now(dt.timezone.utc)
    snapshot_time = _coerce_utc_datetime(snapshot_time) or dt.datetime.now(dt.timezone.utc)
    rows: list[dict] = []
    for market in markets:
        market_id = market.get("market_id")
        race_start_time = _coerce_utc_datetime(market.get("race_start_time"))
        if not market_id or race_start_time is None:
            continue
        seconds_to_start = int((race_start_time - snapshot_time).total_seconds())
        for runner in market.get("runners", []):
            metadata = runner.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            rows.append(
                {
                    "market_id": market_id,
                    "selection_id": runner.get("selection_id"),
                    "snapshot_time": snapshot_time,
                    "race_start_time": race_start_time,
                    "seconds_to_start": seconds_to_start,
                    "source": source,
                    "runner_name": runner.get("runner_name"),
                    "jockey_name": metadata.get("JOCKEY_NAME"),
                    "trainer_name": metadata.get("TRAINER_NAME"),
                    "age": _to_int(metadata.get("AGE")),
                    "official_rating": _to_float(metadata.get("OFFICIAL_RATING")),
                    "adjusted_rating": _to_float(metadata.get("ADJUSTED_RATING")),
                    "days_since_last_run": _to_int(metadata.get("DAYS_SINCE_LAST_RUN")),
                    "weight_value": _to_float(metadata.get("WEIGHT_VALUE")),
                    "weight_units": metadata.get("WEIGHT_UNITS"),
                    "jockey_claim": _to_float(metadata.get("JOCKEY_CLAIM")),
                    "stall_draw": _to_int(
                        metadata.get("STALL_DRAW")
                        if metadata.get("STALL_DRAW") is not None
                        else runner.get("stall_draw")
                    ),
                    "form_string": metadata.get("FORM"),
                    "metadata": metadata,
                }
            )
    return rows


def main() -> None:
    """Download markets, snapshots, and results for a specific date."""
    parser = argparse.ArgumentParser(description="Download Betfair markets and snapshots.")
    parser.add_argument("--date", required=True, help="ISO date, e.g. 2024-01-01")
    parser.add_argument("--dry-run", action="store_true", help="Force mock data even if credentials exist.")
    args = parser.parse_args()

    target_date = dt.date.fromisoformat(args.date)
    log(f"Download: starting for {target_date} (dry-run={args.dry_run}).")
    client = BetfairClient()
    if args.dry_run:
        client._dry_run = True  # type: ignore[attr-defined]

    store = DuckDBStore()

    markets = client.list_markets_for_date(target_date)
    store.upsert_markets(markets)
    runner_rows = []
    market_ids = []
    for market in markets:
        market_ids.append(market["market_id"])
        runner_rows.extend(market["runners"])
    store.upsert_runners(runner_rows)
    metadata_rows = extract_runner_metadata_snapshots(markets, source="list_market_catalogue")
    store.append_runner_metadata_snapshots(metadata_rows)

    snapshots = capture_snapshots(client, market_ids, SNAPSHOT_OFFSETS_MIN)
    store.append_snapshots(snapshots)

    results = client.fetch_market_results(market_ids)
    store.upsert_results(results)

    log(
        "Captured "
        f"{len(markets)} markets, {len(runner_rows)} runners, "
        f"{len(metadata_rows)} metadata snapshots, {len(snapshots)} price snapshots."
    )
    log("Download: complete.")


if __name__ == "__main__":
    main()
