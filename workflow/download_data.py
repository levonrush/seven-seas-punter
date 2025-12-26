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

    snapshots = capture_snapshots(client, market_ids, SNAPSHOT_OFFSETS_MIN)
    store.append_snapshots(snapshots)

    results = client.fetch_market_results(market_ids)
    store.upsert_results(results)

    log(f"Captured {len(markets)} markets, {len(runner_rows)} runners, {len(snapshots)} snapshots.")
    log("Download: complete.")


if __name__ == "__main__":
    main()
