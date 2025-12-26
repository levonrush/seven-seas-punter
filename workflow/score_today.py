import argparse
import datetime as dt
import pathlib

import pandas as pd

from shared.backtest.engine import compute_expected_value
from shared.betfair.client import BetfairClient
from shared.features.builder import SNAPSHOT_OFFSETS_MIN, build_features_from_store
from shared.model.training import load_model_and_calibrator, predict_probabilities
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log


def _capture_latest_snapshots(client: BetfairClient, market_ids: list[str], cutoff_minutes: int) -> list[dict]:
    """Capture snapshots around 'now', tagged to the requested cutoff for deterministic output."""
    now = dt.datetime.utcnow()
    snapshots = []
    books = client.fetch_market_books(market_ids)
    for book in books:
        book["seconds_to_start"] = cutoff_minutes * 60
        book["snapshot_time"] = now
        snapshots.append(book)
    return snapshots


def main() -> None:
    """Score today's markets and emit a CSV of the best value opportunities."""
    parser = argparse.ArgumentParser(description="Score today's markets.")
    parser.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    parser.add_argument("--dry-run", action="store_true", help="Use mock data even if credentials exist.")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    today = dt.date.today()
    log(f"Score: starting for {today} (dry-run={args.dry_run}).")
    client = BetfairClient()
    if args.dry_run:
        client._dry_run = True  # type: ignore[attr-defined]
    store = DuckDBStore()

    markets = client.list_markets_for_date(today)
    store.upsert_markets(markets)
    market_ids = [m["market_id"] for m in markets]
    runner_rows = []
    for market in markets:
        runner_rows.extend(market["runners"])
    store.upsert_runners(runner_rows)

    # Capture snapshots and build features
    snapshots = _capture_latest_snapshots(client, market_ids, cutoff_minutes=args.cutoff_minutes)
    store.append_snapshots(snapshots)

    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if features.empty:
        print("No features available for scoring.")
        return

    model_path = f"artifacts/model_cutoff_{args.cutoff_minutes}.joblib"
    calibrator_path = f"artifacts/calibrator_cutoff_{args.cutoff_minutes}.joblib"
    try:
        model, calibrator = load_model_and_calibrator(model_path, calibrator_path)
        probs = predict_probabilities(model, calibrator, features)
    except FileNotFoundError:
        price_col = f"back_price_t{args.cutoff_minutes}"
        implied = 1 / features[price_col]
        probs = implied.fillna(implied.mean())

    features["p_hat"] = probs
    price_col = f"back_price_t{args.cutoff_minutes}"
    features["ev"] = features.apply(
        lambda r: compute_expected_value(
            prob=r["p_hat"], price=r.get(price_col) or r.get("back_price_t10") or 0.0
        ),
        axis=1,
    )

    # Join runner names for readability
    runners = store.load_runners()
    if not runners.empty:
        features = features.merge(runners, on=["market_id", "selection_id"], how="left")

    best = (
        features.sort_values("ev", ascending=False)
        .groupby("market_id")
        .head(1)
        .reset_index(drop=True)
    )

    output_rows = []
    for _, row in best.iterrows():
        output_rows.append(
            {
                "date": today.isoformat(),
                "venue": row.get("venue"),
                "race_time": row.get("race_start_time"),
                "market_id": row["market_id"],
                "selection": row.get("runner_name", row["selection_id"]),
                "bet_type": "WIN",
                "price": row.get(price_col),
                "p_hat": row["p_hat"],
                "expected_value": row["ev"],
                "suggested_stake": 1.0,
                "notes": "dry-run" if client.dry_run else "",
            }
        )

    df = pd.DataFrame(output_rows)
    output_path = pathlib.Path(
        args.output or f"artifacts/score_today_{today.isoformat()}_cutoff_{args.cutoff_minutes}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"Wrote {len(df)} opportunities to {output_path}")
    log("Score: complete.")


if __name__ == "__main__":
    main()
