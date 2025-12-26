import argparse
import datetime as dt
import json
import pathlib

from shared.backtest.engine import run_backtest
from shared.betfair.client import BetfairClient
from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.training import load_model_and_calibrator, predict_probabilities, train_and_calibrate
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log
from workflow.ingest_archive import ingest_archive_file


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a tar archive (Betfair stream .bz2 or tabular CSV/Parquet) into DuckDB."""
    store = DuckDBStore()
    if not args.force_ingest and store.has_data("snapshots"):
        existing = store.table_row_count("snapshots")
        log(
            f"Ingest: snapshots already present ({existing} rows); skipping ingest. Use --force-ingest to re-run."
        )
        return
    counts = ingest_archive_file(
        pathlib.Path(args.archive),
        store,
        flush_every=args.flush_every,
        progress_every=args.progress_every,
        workers=args.workers,
        filter_au_win=args.filter_au_win,
    )
    log(f"Ingested rows: {counts}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download Betfair markets/snapshots/results for a date (mocked in dry-run)."""
    from workflow.download_data import capture_snapshots  # local import to reuse helper

    target_date = dt.date.fromisoformat(args.date)
    client = BetfairClient()
    if args.dry_run:
        client._dry_run = True  # type: ignore[attr-defined]
    store = DuckDBStore()
    markets = client.list_markets_for_date(target_date)
    store.upsert_markets(markets)
    market_ids = [m["market_id"] for m in markets]
    runner_rows = [r for m in markets for r in m["runners"]]
    store.upsert_runners(runner_rows)
    snapshots = capture_snapshots(client, market_ids, offsets=[60, 30, 10, 5, 2, 1])
    store.append_snapshots(snapshots)
    results = client.fetch_market_results(market_ids)
    store.upsert_results(results)
    log(f"Downloaded {len(markets)} markets, {len(runner_rows)} runners, {len(snapshots)} snapshots.")


def cmd_features(args: argparse.Namespace) -> None:
    """Build features at a cutoff minute and save Parquet."""
    print(f"Building features (cutoff T-{args.cutoff_minutes})...")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    output_path = pathlib.Path("data") / f"features_cutoff_{args.cutoff_minutes}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        features.to_parquet(output_path, index=False)
        log(f"Wrote features to {output_path} ({len(features)} rows).")
    except ImportError as exc:
        csv_path = output_path.with_suffix(".csv")
        features.to_csv(csv_path, index=False)
        log(f"Parquet engine missing ({exc}); wrote CSV to {csv_path} ({len(features)} rows).")


def cmd_train(args: argparse.Namespace) -> None:
    """Train LightGBM (Optuna tuned) + calibrator and persist artefacts."""
    print(f"Training model (cutoff T-{args.cutoff_minutes})...")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if getattr(args, "split_date", None):
        features, _ = split_features_by_race_time(features, args.split_date)
        log(f"Training: split-date {args.split_date} -> {len(features)} rows.")
    else:
        log("Training: no split-date set; training on full dataset (in-sample).")
    if features.empty:
        log("Training: no features available after split; skipping.")
        return
    model_path, calibrator_path, metrics = train_and_calibrate(
        features_df=features,
        cutoff_minutes=args.cutoff_minutes,
        store=store,
        split_date=getattr(args, "split_date", None),
    )
    log(f"Model saved: {model_path}, calibrator: {calibrator_path}, metrics: {metrics}")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest value strategy using trained model (or implied probs fallback)."""
    print(f"Backtesting (cutoff T-{args.cutoff_minutes})...")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if getattr(args, "split_date", None):
        _, features = split_features_by_race_time(features, args.split_date)
        log(f"Backtest: split-date {args.split_date} -> {len(features)} rows.")
    else:
        log("Backtest: no split-date set; backtest is in-sample.")
    if features.empty:
        log("Backtest: no features available after split; skipping.")
        return
    model_path = getattr(args, "model_path", None) or f"artifacts/model_cutoff_{args.cutoff_minutes}.joblib"
    calibrator_path = getattr(args, "calibrator_path", None) or f"artifacts/calibrator_cutoff_{args.cutoff_minutes}.joblib"
    try:
        model, calibrator = load_model_and_calibrator(model_path, calibrator_path)
        probs = predict_probabilities(model, calibrator, features)
    except FileNotFoundError:
        price_col = f"back_price_t{args.cutoff_minutes}"
        implied = 1 / features[price_col]
        probs = implied.fillna(implied.mean())
    bets, metrics = run_backtest(
        feature_df=features,
        probs=probs,
        cutoff_minutes=args.cutoff_minutes,
        commission=args.commission,
        min_ev=args.min_ev,
        max_spread=args.max_spread,
        stake=args.stake,
    )
    store.record_bets(bets.to_dict(orient="records"))
    log(f"Backtest metrics: {metrics}")


def cmd_score(args: argparse.Namespace) -> None:
    """Score today's markets and write CSV report."""
    print(f"Scoring today (cutoff T-{args.cutoff_minutes})...")
    import pandas as pd
    from workflow.score_today import _capture_latest_snapshots

    today = dt.date.today()
    client = BetfairClient()
    if args.dry_run:
        client._dry_run = True  # type: ignore[attr-defined]
    store = DuckDBStore()

    markets = client.list_markets_for_date(today)
    store.upsert_markets(markets)
    market_ids = [m["market_id"] for m in markets]
    runner_rows = [r for m in markets for r in m["runners"]]
    store.upsert_runners(runner_rows)

    snapshots = _capture_latest_snapshots(client, market_ids, cutoff_minutes=args.cutoff_minutes)
    store.append_snapshots(snapshots)

    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if features.empty:
        log("No features available for scoring.")
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

    from shared.backtest.engine import compute_expected_value

    features["p_hat"] = probs
    price_col = f"back_price_t{args.cutoff_minutes}"
    features["ev"] = features.apply(
        lambda r: compute_expected_value(
            prob=r["p_hat"], price=r.get(price_col) or r.get("back_price_t10") or 0.0
        ),
        axis=1,
    )

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


def cmd_status(args: argparse.Namespace) -> None:
    """Print simple row counts for sanity checks."""
    store = DuckDBStore()
    with store._connect() as con:  # type: ignore[attr-defined]
        tables = ["markets", "runners", "snapshots", "results", "bets", "model_runs"]
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    log(f"DB counts: {counts}")


def cmd_report(args: argparse.Namespace) -> None:
    """Print latest model metrics and backtest profit summary."""
    store = DuckDBStore()
    with store._connect() as con:  # type: ignore[attr-defined]
        model_row = con.execute(
            """
            SELECT created_at, cutoff_minutes, model_path, calibrator_path, metrics
            FROM model_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
        bet_row = con.execute(
            """
            SELECT
                COUNT(*) AS bets,
                SUM(stake) AS turnover,
                SUM(result_profit) AS profit,
                SUM(expected_value * stake) AS expected_profit,
                SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate
            FROM bets
            """
        ).fetchone()

    if model_row:
        created_at, cutoff_minutes, model_path, calibrator_path, metrics = model_row
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {"raw": metrics}
        log(f"Latest model run: {created_at} cutoff=T-{cutoff_minutes}")
        log(f"Model path: {model_path}")
        log(f"Calibrator path: {calibrator_path}")
        log(f"Model metrics: {metrics}")
    else:
        log("No model_runs found.")

    if bet_row and bet_row[0]:
        bets, turnover, profit, expected_profit, hit_rate = bet_row
        roi = (profit / turnover) if turnover else 0
        expected_roi = (expected_profit / turnover) if turnover else 0
        log(
            f"Backtest summary: bets={bets}, turnover={turnover:.2f}, profit={profit:.2f}, "
            f"roi={roi:.4f}, expected_profit={expected_profit:.2f}, expected_roi={expected_roi:.4f}, "
            f"hit_rate={hit_rate:.3f}"
        )
    else:
        log("No bets found.")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run ingest (optional), build features, train, backtest, and score in one go."""
    log("Starting pipeline...")
    if args.archive:
        log("Step: ingest archive")
        cmd_ingest(args)
    elif args.download_date:
        log("Step: download markets")
        dl_args = argparse.Namespace(date=args.download_date, dry_run=args.dry_run)
        cmd_download(dl_args)

    if not args.skip_features:
        log("Step: build features")
        cmd_features(args)
    if not args.skip_train:
        log("Step: train model")
        cmd_train(args)
    if not args.skip_backtest:
        log("Step: backtest")
        cmd_backtest(args)
    if not args.skip_score:
        log("Step: score")
        args.output = args.output or f"artifacts/pipeline_score_cutoff_{args.cutoff_minutes}.csv"
        cmd_score(args)
    log("Pipeline complete.")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with subcommands."""
    parser = argparse.ArgumentParser(description="Seven Seas Punter pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest tar archive (stream .bz2 or tabular CSV/Parquet)")
    p_ingest.add_argument("--archive", default="data/data.tar")
    p_ingest.add_argument("--flush-every", type=int, default=5000)
    p_ingest.add_argument("--progress-every", type=int, default=100)
    p_ingest.add_argument("--workers", type=int, default=None, help="Parallel workers for stream ingest")
    p_ingest.add_argument("--filter-au-win", action="store_true", help="Only keep AU WIN markets")
    p_ingest.add_argument("--force-ingest", action="store_true", help="Re-ingest even if data exists")
    p_ingest.set_defaults(func=cmd_ingest)

    p_dl = sub.add_parser("download", help="Download markets/snapshots for a date (or dry-run)")
    p_dl.add_argument("--date", required=True, help="ISO date, e.g., 2024-01-01")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_feat = sub.add_parser("features", help="Build features at cutoff")
    p_feat.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_feat.set_defaults(func=cmd_features)

    p_train = sub.add_parser("train", help="Train model at cutoff")
    p_train.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_train.add_argument(
        "--split-date",
        help="ISO date/time; training uses races strictly before this (leakage-safe split).",
    )
    p_train.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest", help="Backtest value strategy")
    p_bt.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_bt.add_argument("--commission", type=float, default=0.05)
    p_bt.add_argument("--min-ev", type=float, default=0.02)
    p_bt.add_argument("--max-spread", type=float, default=1.0)
    p_bt.add_argument("--stake", type=float, default=1.0)
    p_bt.add_argument("--model-path")
    p_bt.add_argument("--calibrator-path")
    p_bt.add_argument(
        "--split-date",
        help="ISO date/time; backtest uses races on/after this (holdout set).",
    )
    p_bt.set_defaults(func=cmd_backtest)

    p_score = sub.add_parser("score", help="Score today and output CSV")
    p_score.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--output")
    p_score.set_defaults(func=cmd_score)

    p_pipe = sub.add_parser("pipeline", help="Run ingest (optional), features, train, backtest, score")
    p_pipe.add_argument("--archive", help="Optional tar archive to ingest first")
    p_pipe.add_argument("--download-date", help="Optional date to download (dry-run friendly)")
    p_pipe.add_argument("--flush-every", type=int, default=5000, help="Flush size for ingest")
    p_pipe.add_argument("--progress-every", type=int, default=100, help="File progress print frequency")
    p_pipe.add_argument("--workers", type=int, default=None, help="Parallel workers for stream ingest")
    p_pipe.add_argument("--filter-au-win", action="store_true", help="Only keep AU WIN markets")
    p_pipe.add_argument("--force-ingest", action="store_true", help="Re-ingest even if data exists")
    p_pipe.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_pipe.add_argument("--commission", type=float, default=0.05)
    p_pipe.add_argument("--min-ev", type=float, default=0.02)
    p_pipe.add_argument("--max-spread", type=float, default=1.0)
    p_pipe.add_argument("--stake", type=float, default=1.0)
    p_pipe.add_argument(
        "--split-date",
        help="ISO date/time; train uses races before this, backtest uses on/after.",
    )
    p_pipe.add_argument("--output")
    p_pipe.add_argument("--dry-run", action="store_true", help="Applied to scoring step")
    p_pipe.add_argument("--skip-features", action="store_true")
    p_pipe.add_argument("--skip-train", action="store_true")
    p_pipe.add_argument("--skip-backtest", action="store_true")
    p_pipe.add_argument("--skip-score", action="store_true")
    p_pipe.set_defaults(func=cmd_pipeline)

    p_qs = sub.add_parser("quickstart", help="Run a dry-run end-to-end using mock download data")
    p_qs.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_qs.set_defaults(
        func=lambda args: cmd_pipeline(
            argparse.Namespace(
                archive=None,
                download_date="2024-01-01",
                model_path=None,
                calibrator_path=None,
                flush_every=5000,
                cutoff_minutes=args.cutoff_minutes,
                commission=0.05,
                min_ev=0.02,
                max_spread=1.0,
                stake=1.0,
                output=None,
                dry_run=True,
                skip_features=False,
                skip_train=False,
                skip_backtest=False,
                skip_score=False,
            )
        )
    )

    p_status = sub.add_parser("status", help="Show row counts in DuckDB tables")
    p_status.set_defaults(func=cmd_status)

    p_report = sub.add_parser("report", help="Show latest model metrics and backtest summary")
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    """Entry point for the unified CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
