import argparse
import datetime as dt
import json
import pathlib
import uuid

import pandas as pd

from shared.backtest.allocation import allocate_stakes_from_budget, summarize_budget_usage
from shared.backtest.engine import build_bet_preview, run_backtest
from shared.backtest.strategy_tuner import tune_strategy
from shared.betfair.client import BetfairClient
from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.predictions import build_prediction_preview
from shared.model.market_type import bucket_market_type
from shared.model.training import load_model_and_calibrator, predict_probabilities, train_and_calibrate
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.bet_explain import preview_legend_lines
from shared.utils.cli_presets import go_historic_args, go_pipeline_args
from shared.utils.progress import log
from workflow.ingest_archive import ingest_archive_file


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a tar archive (Betfair stream .bz2 or tabular CSV/Parquet) into DuckDB."""
    store = DuckDBStore()
    archive_path = pathlib.Path(args.archive)
    if not archive_path.exists():
        log(f"Ingest: archive not found at {archive_path}; skipping.")
        return
    incremental = getattr(args, "incremental", False)
    manifest_path = None
    manifest_value = getattr(args, "ingest_manifest", None) or "data/ingest_manifest.json"
    manifest_candidate = pathlib.Path(manifest_value)
    if incremental or getattr(args, "ingest_manifest", None):
        manifest_path = manifest_candidate
    snapshots_exist = store.has_data("snapshots")
    if snapshots_exist and not args.force_ingest and not incremental and manifest_candidate.exists():
        incremental = True
        manifest_path = manifest_candidate
        log(
            "Ingest: manifest found; "
            f"using incremental ingest ({manifest_candidate})."
        )
    if snapshots_exist:
        existing = store.table_row_count("snapshots")
        if not args.force_ingest and not incremental:
            log(
                "Ingest: snapshots already present "
                f"({existing} rows); skipping ingest. Use --force-ingest to re-run."
            )
            return
        if incremental:
            if manifest_path and not manifest_path.exists():
                log(
                    "Ingest: incremental manifest missing; "
                    "will seed from archive and skip ingest to avoid duplicates."
                )
            else:
                log(
                    "Ingest: snapshots already present "
                    f"({existing} rows); ingesting only new archive members."
                )
    elif incremental and manifest_path and not manifest_path.exists():
        log("Ingest: incremental manifest missing; full ingest will create it.")
    counts = ingest_archive_file(
        archive_path,
        store,
        flush_every=args.flush_every,
        progress_every=args.progress_every,
        workers=args.workers,
        filter_au_win=args.filter_au_win,
        incremental=incremental,
        manifest_path=manifest_path,
        force=args.force_ingest,
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


def cmd_download_historic(args: argparse.Namespace) -> None:
    """Download historic data packages using Betfair's historic API."""
    from workflow.download_historic import run_download_historic

    run_download_historic(args)


def _build_live_overrides(args: argparse.Namespace) -> dict:
    """Build sparse live-config overrides from CLI flags so users can tweak common knobs quickly."""
    overrides: dict = {}
    if getattr(args, "dry_run", None) is not None:
        overrides["dry_run"] = bool(args.dry_run)
    if getattr(args, "poll_interval_seconds", None) is not None:
        overrides["poll_interval_seconds"] = float(args.poll_interval_seconds)

    max_iterations = getattr(args, "max_iterations", None)
    if getattr(args, "once", False):
        max_iterations = 1
    if max_iterations is not None:
        overrides["max_iterations"] = int(max_iterations)

    strategy_overrides = {}
    if getattr(args, "min_edge", None) is not None:
        strategy_overrides["min_edge"] = float(args.min_edge)
    if strategy_overrides:
        overrides["strategy"] = strategy_overrides

    safety_overrides = {}
    if getattr(args, "max_stake_per_market", None) is not None:
        safety_overrides["max_stake_per_market"] = float(args.max_stake_per_market)
    if getattr(args, "max_daily_exposure", None) is not None:
        safety_overrides["max_daily_exposure"] = float(args.max_daily_exposure)
    if getattr(args, "ignore_within_minutes", None) is not None:
        safety_overrides["ignore_within_minutes"] = float(args.ignore_within_minutes)
    if safety_overrides:
        overrides["safety"] = safety_overrides
    return overrides


def cmd_live(args: argparse.Namespace) -> None:
    """Run live polling + inference + execution with optional CLI overrides for fast operational control."""
    from shared.live.betfair_live import run_live_loop, write_live_config_template

    if getattr(args, "write_config", None):
        try:
            output_path = write_live_config_template(args.write_config, overwrite=args.force)
        except FileExistsError as exc:
            log(str(exc))
            return
        log(f"Live: wrote starter config to {output_path}")
        return

    config_path = getattr(args, "config", None) or "config/live.yaml"
    overrides = _build_live_overrides(args)
    if overrides:
        log(f"Live: applying CLI overrides {overrides}")
    try:
        run_live_loop(config_path=config_path, overrides=overrides or None)
    except FileNotFoundError as exc:
        log(str(exc))
        log("Live: create a starter config with `punter live --write-config config/live.yaml`.")


def _ensure_run_id(args: argparse.Namespace) -> None:
    """Assign a run id so training/backtests can be tied to a single CLI execution."""
    if getattr(args, "run_id", None):
        return
    args.run_id = uuid.uuid4().hex
    log(f"Run id: {args.run_id}")

def _ensure_backtest_run_id(args: argparse.Namespace, store: DuckDBStore) -> None:
    """Reuse the latest training run id for backtests when one is not provided."""
    if getattr(args, "run_id", None):
        return
    cutoff = getattr(args, "cutoff_minutes", None)
    with store._connect() as con:  # type: ignore[attr-defined]
        row = con.execute(
            """
            SELECT run_id
            FROM model_runs
            WHERE run_id IS NOT NULL AND cutoff_minutes = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [cutoff],
        ).fetchone()
    if row and row[0]:
        args.run_id = row[0]
        log(f"Backtest: using run id from latest model run ({args.run_id}).")


def _resolve_min_prob(args: argparse.Namespace, store: DuckDBStore) -> float | None:
    """Load the tuned kappa threshold from model runs unless overridden."""
    if getattr(args, "min_prob", None) is not None:
        return args.min_prob
    cutoff = getattr(args, "cutoff_minutes", None)
    run_id = getattr(args, "run_id", None)
    with store._connect() as con:  # type: ignore[attr-defined]
        if run_id:
            row = con.execute(
                """
                SELECT metrics
                FROM model_runs
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [run_id],
            ).fetchone()
        else:
            row = con.execute(
                """
                SELECT metrics
                FROM model_runs
                WHERE cutoff_minutes = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [cutoff],
            ).fetchone()
    if not row:
        return None
    metrics = row[0]
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            return None
    if isinstance(metrics, dict):
        return metrics.get("kappa_threshold")
    return None

def _apply_split_from_days(args: argparse.Namespace, store: DuckDBStore) -> None:
    """Set split_date to the last N days of data when requested."""
    if getattr(args, "split_date", None):
        return
    days = getattr(args, "split_days", None)
    if days is None and getattr(args, "split_last_month", False):
        days = 30
    if days is None:
        return
    if days <= 0:
        log("Split date: split-days must be positive; skipping.")
        return
    max_time = store.max_race_start_time()
    if not max_time:
        log("Split date: no market data found; skipping split-days.")
        return
    split_date = (pd.to_datetime(max_time) - dt.timedelta(days=days)).isoformat()
    args.split_date = split_date
    label = "last month" if days == 30 else f"last {days} days"
    log(f"Split date set to {split_date} ({label}).")


def _filter_to_win_markets(feature_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Filter to WIN markets so win-target evaluation remains consistent."""
    if feature_df.empty or "market_type" not in feature_df.columns:
        return feature_df
    buckets = feature_df["market_type"].apply(bucket_market_type)
    win_mask = buckets == "WIN"
    if win_mask.any():
        filtered = feature_df.loc[win_mask].copy()
        log(f"{label}: filtered to WIN markets ({len(filtered)}/{len(feature_df)} rows).")
        return filtered
    log(f"{label}: no WIN markets found; keeping all rows.")
    return feature_df


def _load_oof_predictions_for_backtest(
    store: DuckDBStore,
    feature_df: pd.DataFrame,
    run_id: str,
    cutoff_minutes: int,
) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """Fetch stored out-of-fold predictions to avoid in-sample backtest bias."""
    oof = store.load_oof_predictions(run_id, cutoff_minutes)
    if oof.empty:
        return None, None
    oof = oof.drop_duplicates(subset=["market_id", "selection_id"], keep="last")
    merged = feature_df.merge(oof, on=["market_id", "selection_id"], how="left")
    missing = merged["p_hat"].isna().sum()
    if missing == len(merged):
        return None, None
    if missing:
        log(
            f"Backtest: {missing}/{len(merged)} rows missing OOF predictions; dropping them."
        )
    merged = merged.dropna(subset=["p_hat"])
    probs = merged["p_hat"].copy()
    features = merged.drop(columns=["p_hat"])
    return features, probs


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
    _ensure_run_id(args)
    _apply_split_from_days(args, store)
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    train_features = features
    test_features = None
    if getattr(args, "split_date", None):
        train_features, test_features = split_features_by_race_time(features, args.split_date)
        log(f"Training: split-date {args.split_date} -> {len(train_features)} train rows.")
    else:
        log("Training: no split-date set; training on full dataset (in-sample).")
    train_features = _filter_to_win_markets(train_features, "Training")
    if test_features is not None:
        test_features = _filter_to_win_markets(test_features, "Holdout")
    if train_features.empty:
        log("Training: no features available after split; skipping.")
        return
    model_path, calibrator_path, metrics, oof_predictions = train_and_calibrate(
        features_df=train_features,
        cutoff_minutes=args.cutoff_minutes,
        store=store,
        split_date=getattr(args, "split_date", None),
        run_id=getattr(args, "run_id", None),
    )
    log(f"Model saved: {model_path}, calibrator: {calibrator_path}, metrics: {metrics}")
    if getattr(args, "show_preds", True):
        tuned_min_prob = None
        if isinstance(metrics, dict):
            tuned_min_prob = metrics.get("kappa_threshold")
        preview_min_prob = (
            args.preds_min_prob if getattr(args, "preds_min_prob", None) is not None else tuned_min_prob
        )
        if preview_min_prob is not None:
            log(f"Prediction preview: using min_prob={preview_min_prob:.3f}.")
        if oof_predictions is not None:
            log("Prediction preview: using out-of-fold predictions from training set.")
            preview_source = train_features
            probs = oof_predictions
        elif test_features is not None:
            log("Prediction preview: using holdout predictions.")
            preview_source = test_features
            model, calibrator = load_model_and_calibrator(model_path, calibrator_path)
            probs = predict_probabilities(model, calibrator, test_features)
        else:
            preview_source = None
            probs = None
        if preview_source is not None and not preview_source.empty and probs is not None:
            preview = build_prediction_preview(
                preview_source,
                probs,
                cutoff_minutes=args.cutoff_minutes,
                runners=store.load_runners(),
                markets=store.load_markets(),
                limit=args.preds_limit,
                min_ev=args.preds_min_ev,
                min_edge=args.preds_min_edge,
                max_price=args.preds_max_price,
                max_edge_multiplier=args.preds_max_edge_mult,
                per_market_limit=args.preds_per_market,
                min_prob=preview_min_prob,
            )
            if preview.empty:
                log("Prediction preview: no rows to show.")
            else:
                log(f"Prediction preview: top {len(preview)} rows by expected value.")
                print(preview.to_string(index=False))
                for line in preview_legend_lines():
                    log(line)
        else:
            log("Prediction preview: no out-of-fold predictions available.")
    if getattr(args, "tune_strategy", False):
        if oof_predictions is None or oof_predictions.empty:
            log("Strategy tuning: no out-of-fold predictions available; skipping.")
        else:
            tuning_features = train_features.loc[oof_predictions.index]
            commission = getattr(args, "commission", 0.05)
            strategy_min_prob = getattr(args, "min_prob", None)
            if strategy_min_prob is None:
                strategy_min_prob = tuned_min_prob
            results, tradeoff = tune_strategy(
                feature_df=tuning_features,
                probs=oof_predictions,
                cutoff_minutes=args.cutoff_minutes,
                commission=commission,
                grid_profile=args.strategy_grid,
                objective=args.strategy_objective,
                min_hit_rate=args.strategy_min_hit_rate,
                min_bets=args.strategy_min_bets,
                stake=1.0,
                log_every=args.strategy_log_every,
                min_prob=strategy_min_prob,
            )
            if results.empty:
                log("Strategy tuning: no configs met constraints.")
            else:
                top_n = min(args.strategy_top_n, len(results))
                log(f"Strategy tuning: top {top_n} configs by {args.strategy_objective}.")
                print(results.head(top_n).to_string(index=False))
                if not tradeoff.empty:
                    log("Strategy tuning: hit-rate trade-off (best per bucket).")
                    print(tradeoff.to_string(index=False))
                output_path = pathlib.Path(
                    args.strategy_output
                    or f"artifacts/strategy_tuning_cutoff_{args.cutoff_minutes}.csv"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(output_path, index=False)
                best_path = output_path.with_name(
                    f"strategy_best_cutoff_{args.cutoff_minutes}.json"
                )
                with best_path.open("w", encoding="utf-8") as fh:
                    best_row = {
                        key: (value.item() if hasattr(value, "item") else value)
                        for key, value in results.iloc[0].to_dict().items()
                    }
                    json.dump(best_row, fh, indent=2)
                log(f"Strategy tuning: wrote {output_path} and {best_path}.")
    if getattr(args, "report", False):
        cmd_report(args)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Backtest value strategy using trained model (or implied probs fallback)."""
    print(f"Backtesting (cutoff T-{args.cutoff_minutes})...")
    store = DuckDBStore()
    _ensure_backtest_run_id(args, store)
    if not getattr(args, "run_id", None):
        _ensure_run_id(args)
    _apply_split_from_days(args, store)
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if getattr(args, "split_date", None):
        _, features = split_features_by_race_time(features, args.split_date)
        log(f"Backtest: split-date {args.split_date} -> {len(features)} rows.")
    else:
        log("Backtest: no split-date set; backtest is in-sample.")
    features = _filter_to_win_markets(features, "Backtest")
    if features.empty:
        log("Backtest: no features available after split; skipping.")
        return
    min_prob = _resolve_min_prob(args, store)
    if min_prob is not None:
        log(f"Backtest: using min_prob={min_prob:.3f} from kappa tuning.")
    model_path = getattr(args, "model_path", None) or f"artifacts/model_cutoff_{args.cutoff_minutes}.joblib"
    calibrator_path = getattr(args, "calibrator_path", None) or f"artifacts/calibrator_cutoff_{args.cutoff_minutes}.joblib"
    probs = None
    if not getattr(args, "split_date", None) and getattr(args, "run_id", None):
        features_oof, probs_oof = _load_oof_predictions_for_backtest(
            store, features, args.run_id, args.cutoff_minutes
        )
        if probs_oof is not None:
            log("Backtest: using out-of-fold predictions from stored training run.")
            features = features_oof
            probs = probs_oof
    if probs is None:
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
        min_edge=args.min_edge,
        max_spread=args.max_spread,
        max_price=args.max_price,
        max_edge_multiplier=args.max_edge_mult,
        min_prob=min_prob,
        stake=args.stake,
    )
    store.record_bets(bets.to_dict(orient="records"), run_id=getattr(args, "run_id", None))
    log(f"Backtest metrics: {metrics}")
    if getattr(args, "show_bets", True):
        preview = build_bet_preview(
            bets, runners=store.load_runners(), markets=store.load_markets(), limit=args.bets_limit
        )
        if preview.empty:
            log("Bet preview: no bets to show.")
        else:
            log(f"Bet preview: top {len(preview)} bets by expected value.")
            print(preview.to_string(index=False))
            for line in preview_legend_lines():
                log(line)


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
    features = _filter_to_win_markets(features, "Score")
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
    min_prob = _resolve_min_prob(args, store)
    if min_prob is not None:
        before = len(features)
        features = features[features["p_hat"] >= min_prob].copy()
        log(f"Score: applied min_prob={min_prob:.3f} ({len(features)}/{before} rows).")
        if features.empty:
            log("No rows remain after min_prob filter.")
            return
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

    best["price"] = best.get(price_col)
    budget = getattr(args, "budget", None)
    if budget is not None:
        if float(budget) <= 0:
            raise ValueError("--budget must be > 0.")
        best = allocate_stakes_from_budget(
            best,
            budget=float(budget),
            commission=0.05,
            method=str(getattr(args, "allocation_method", "fractional_kelly")),
            kelly_fraction=float(getattr(args, "kelly_fraction", 0.25)),
            max_bet_pct=float(getattr(args, "max_bet_pct", 0.2)),
        )
        before = len(best)
        best = best[best["suggested_stake"] > 0].copy()
        log(f"Score: budget allocation kept {len(best)}/{before} rows with non-zero stake.")
        used, remaining = summarize_budget_usage(best, float(budget))
        log(
            "Score: "
            f"allocated {used:.2f}/{float(budget):.2f} budget; "
            f"remaining={remaining:.2f}."
        )
    else:
        best["suggested_stake"] = 1.0
        best["allocation_method"] = "flat"
        best["kelly_fraction_full"] = None
        best["stake_fraction"] = None
    if best.empty:
        log("Score: no candidates received a positive suggested stake.")
        return

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
                "suggested_stake": row.get("suggested_stake", 1.0),
                "stake_fraction": row.get("stake_fraction"),
                "allocation_method": row.get("allocation_method"),
                "kelly_fraction_full": row.get("kelly_fraction_full"),
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


def cmd_pub(args: argparse.Namespace) -> None:
    """Generate a manual-betting day sheet from live Betfair prices without placing any orders.

    Why: provide a simple pub-friendly entrypoint so users can get today's shortlist without
    remembering the full `score` command flags.
    """
    if not getattr(args, "output", None):
        args.output = f"artifacts/pub_sheet_{dt.date.today().isoformat()}.csv"
    log(f"Pub sheet: writing to {args.output}")
    cmd_score(args)


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
        run_id = getattr(args, "run_id", None)
        if run_id:
            model_row = con.execute(
                """
                SELECT created_at, cutoff_minutes, model_path, calibrator_path, metrics, run_id
                FROM model_runs
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [run_id],
            ).fetchone()
        else:
            model_row = con.execute(
                """
                SELECT created_at, cutoff_minutes, model_path, calibrator_path, metrics, run_id
                FROM model_runs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            run_id = model_row[5] if model_row else None
        if run_id:
            bet_row = con.execute(
                """
                SELECT
                    COUNT(*) AS bets,
                    SUM(stake) AS turnover,
                    SUM(result_profit) AS profit,
                    SUM(expected_value * stake) AS expected_profit,
                    SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate
                FROM bets
                WHERE run_id = ?
                """,
                [run_id],
            ).fetchone()
        else:
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

    training_metrics = {}
    if model_row:
        created_at, cutoff_minutes, model_path, calibrator_path, metrics, model_run_id = model_row
        if model_run_id:
            log(f"Run id: {model_run_id}")
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {"raw": metrics}
        training_metrics = metrics if isinstance(metrics, dict) else {}
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

    if model_row:
        split_date = training_metrics.get("split_date")
        cv_folds = training_metrics.get("cv_folds")
        cv_gap = training_metrics.get("cv_gap_days")
        cv_strategy = training_metrics.get("cv_strategy")
        train_rows = training_metrics.get("train_rows")
        train_markets = training_metrics.get("train_markets")
        calib_rows = training_metrics.get("calib_rows")
        kappa_threshold = training_metrics.get("kappa_threshold")
        sample_label = "out-of-sample" if split_date else "in-sample"
        cv_bits = []
        if cv_folds:
            cv_bits.append(f"{cv_folds} folds")
        if cv_gap is not None:
            cv_bits.append(f"gap={cv_gap}d")
        if cv_strategy:
            cv_bits.append(f"strategy={cv_strategy}")
        cv_text = ", ".join(cv_bits) if cv_bits else "n/a"
        threshold_text = (
            f"{kappa_threshold:.3f}" if isinstance(kappa_threshold, (float, int)) else "n/a"
        )
        log("Training vs backtest summary:")
        log(
            "Training: "
            f"rows={train_rows or 'n/a'}, markets={train_markets or 'n/a'}, "
            f"calib_rows={calib_rows or 'n/a'}, split_date={split_date or 'none'}, "
            f"cv={cv_text}, kappa_threshold={threshold_text}."
        )
        if bet_row and bet_row[0]:
            log(
                "Backtest: "
                f"bets={bets}, roi={roi:.4f}, expected_roi={expected_roi:.4f}, "
                f"hit_rate={hit_rate:.3f}, sample={sample_label}."
            )

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run ingest (optional), build features, train, backtest, and score in one go."""
    log("Starting pipeline...")
    _ensure_run_id(args)
    if getattr(args, "ingest_new", False):
        args.incremental = True
    if args.archive:
        archive_path = pathlib.Path(args.archive)
        if not archive_path.exists():
            log(f"Step: ingest archive skipped (missing {archive_path}).")
            args.archive = None
    if not args.archive:
        default_archive = pathlib.Path("data/data.tar")
        if default_archive.exists():
            args.archive = str(default_archive)
            log(f"Step: ingest archive (default {default_archive}).")
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
        report_flag = getattr(args, "report", False)
        args.report = False
        cmd_train(args)
        args.report = report_flag
    if not args.skip_backtest:
        log("Step: backtest")
        cmd_backtest(args)
    if not args.skip_score:
        log("Step: score")
        args.output = args.output or f"artifacts/pipeline_score_cutoff_{args.cutoff_minutes}.csv"
        cmd_score(args)
    if getattr(args, "report", False):
        log("Step: report")
        cmd_report(args)
    log("Pipeline complete.")


def cmd_go(_args: argparse.Namespace) -> None:
    """Run the default optimized end-to-end flow with no tuning flags required.

    Why: provide a single CLI command for users who want the recommended path without
    memorizing command options. Advanced users can still tune each subcommand directly.
    """
    parser = build_parser()

    historic_tokens = ["download-historic", *go_historic_args()]
    log(f"Go: running `punter {' '.join(historic_tokens)}`")
    historic_args = parser.parse_args(historic_tokens)
    historic_args.func(historic_args)

    pipeline_tokens = ["pipeline", *go_pipeline_args()]
    log(f"Go: running `punter {' '.join(pipeline_tokens)}`")
    pipeline_args = parser.parse_args(pipeline_tokens)
    pipeline_args.func(pipeline_args)


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
    p_ingest.add_argument(
        "--incremental",
        action="store_true",
        help="Only ingest new archive members (uses ingest manifest; auto-enabled when manifest exists).",
    )
    p_ingest.add_argument(
        "--ingest-manifest",
        help="Path to ingest manifest JSON (default data/ingest_manifest.json).",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_dl = sub.add_parser("download", help="Download markets/snapshots for a date (or dry-run)")
    p_dl.add_argument("--date", required=True, help="ISO date, e.g., 2024-01-01")
    p_dl.add_argument("--dry-run", action="store_true")
    p_dl.set_defaults(func=cmd_download)

    p_hist = sub.add_parser("download-historic", help="Download historic data via Betfair Historic API")
    p_hist.add_argument("--auto", action="store_true", help="Download all purchased months for the sport.")
    p_hist.add_argument("--from-date", help="Start date (YYYY-MM-DD).")
    p_hist.add_argument("--to-date", help="End date (YYYY-MM-DD).")
    p_hist.add_argument("--sport", default="Horse Racing", help="Sport name (e.g., Horse Racing).")
    p_hist.add_argument("--plan", help="Plan name (e.g., Basic Plan). Omit for all plans in --auto.")
    p_hist.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types. Use ALL (default) for no market-type filter.",
    )
    p_hist.add_argument("--countries", default="AU", help="Comma-separated country codes.")
    p_hist.add_argument("--file-types", default="M", help="Comma-separated file types (M/E).")
    p_hist.add_argument("--event-id", type=int)
    p_hist.add_argument("--event-name")
    p_hist.add_argument(
        "--show-market-types",
        action="store_true",
        help="Print available market types (with counts) for the selected window and exit.",
    )
    p_hist.add_argument("--show-options", action="store_true", help="Show available filters.")
    p_hist.add_argument("--size-only", action="store_true", help="Only show file count and size.")
    p_hist.add_argument("--list-only", action="store_true", help="Only list available files.")
    p_hist.add_argument("--list-output", help="Optional path to write the file list JSON.")
    p_hist.add_argument(
        "--list-cache-dir",
        default="data/historic_lists",
        help="Cache list_files responses (set empty to disable).",
    )
    p_hist.add_argument(
        "--refresh-list-cache",
        action="store_true",
        help="Ignore cached file lists and refresh from the API.",
    )
    p_hist.add_argument("--download", action="store_true", help="Download the files.")
    p_hist.add_argument("--output", help="Output tar path (defaults to data/historic_*.tar).")
    p_hist.add_argument("--output-dir", help="Directory to store downloads before tarring.")
    p_hist.add_argument("--keep-files", action="store_true", help="Keep downloaded files on disk.")
    p_hist.add_argument("--clean-temp", action="store_true", help="Remove temp download dir before starting.")
    p_hist.add_argument("--max-files", type=int, help="Limit number of files (useful for testing).")
    p_hist.add_argument("--workers", type=int, default=None, help="Parallel download workers (default: auto).")
    p_hist.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Override request cap per window (defaults to BETFAIR_HISTORIC_MAX_REQUESTS or 90).",
    )
    p_hist.add_argument(
        "--request-window-seconds",
        type=float,
        default=None,
        help="Override rate-limit window seconds (defaults to BETFAIR_HISTORIC_REQUEST_WINDOW or 10).",
    )
    p_hist.add_argument("--progress-every", type=int, default=50, help="Progress logging frequency.")
    p_hist.add_argument("--manifest-path", help="Path for download manifest JSON.")
    p_hist.add_argument("--force", action="store_true", help="Re-download files even if seen before.")
    p_hist.add_argument("--retries", type=int, default=5, help="Retry count for historic API failures.")
    p_hist.add_argument("--retry-wait", type=float, default=3.0, help="Seconds to wait between retries.")
    p_hist.add_argument(
        "--target-basket-mb",
        type=float,
        default=1000.0,
        help="Shard date windows when basket size exceeds this MB threshold (set <=0 to disable).",
    )
    p_hist.add_argument(
        "--max-shard-depth",
        type=int,
        default=5,
        help="Maximum recursive split depth for oversized windows.",
    )
    p_hist.add_argument(
        "--no-auto-shard",
        dest="auto_shard",
        action="store_false",
        help="Disable automatic basket-size based date sharding.",
    )
    p_hist.add_argument(
        "--no-adaptive-workers",
        dest="adaptive_workers",
        action="store_false",
        help="Disable worker backoff when transient failures are detected.",
    )
    p_hist.add_argument("--max-download-rounds", type=int, default=3, help="Retry rounds for transient failures.")
    p_hist.add_argument("--adaptive-cooldown", type=float, default=2.0, help="Seconds to wait between rounds.")
    p_hist.set_defaults(auto_shard=True, adaptive_workers=True)
    p_hist.set_defaults(func=cmd_download_historic)

    p_feat = sub.add_parser("features", help="Build features at cutoff")
    p_feat.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_feat.set_defaults(func=cmd_features)

    p_train = sub.add_parser("train", help="Train model at cutoff")
    p_train.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_train.add_argument(
        "--split-date",
        help="ISO date/time; training uses races strictly before this (leakage-safe split).",
    )
    p_train.add_argument(
        "--split-last-month",
        action="store_true",
        help="Set split-date to the last 30 days of available data.",
    )
    p_train.add_argument(
        "--split-last-180",
        dest="split_days",
        action="store_const",
        const=180,
        help="Set split-date to the last 180 days of available data.",
    )
    p_train.add_argument(
        "--split-days",
        type=int,
        help="Set split-date to the last N days of available data.",
    )
    p_train.add_argument("--run-id", help="Optional run id to align training/backtests/reports.")
    p_train.add_argument(
        "--show-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a preview of top predictions after training.",
    )
    p_train.add_argument(
        "--preds-limit",
        type=int,
        default=20,
        help="Rows to show in the prediction preview.",
    )
    p_train.add_argument("--preds-min-ev", type=float, default=0.02, help="Min EV for preview rows.")
    p_train.add_argument(
        "--preds-min-edge", type=float, default=0.1, help="Min edge (relative) for preview rows."
    )
    p_train.add_argument(
        "--preds-min-prob",
        type=float,
        help="Optional min probability for preview rows (defaults to tuned kappa threshold).",
    )
    p_train.add_argument("--preds-max-price", type=float, default=200.0, help="Max price to show.")
    p_train.add_argument(
        "--preds-max-edge-mult",
        type=float,
        default=5.0,
        help="Max multiple of market implied prob to show.",
    )
    p_train.add_argument(
        "--preds-per-market",
        type=int,
        default=1,
        help="Max preview rows per market.",
    )
    p_train.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a report summary after training.",
    )
    p_train.add_argument(
        "--tune-strategy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune betting filters on out-of-fold predictions (use --no-tune-strategy to disable).",
    )
    p_train.add_argument(
        "--strategy-grid",
        choices=["small", "medium", "large"],
        default="small",
        help="Grid size for strategy tuning.",
    )
    p_train.add_argument(
        "--strategy-objective",
        choices=["expected_roi", "roi", "expected_profit", "profit"],
        default="expected_roi",
        help="Metric to maximize during strategy tuning.",
    )
    p_train.add_argument(
        "--strategy-min-hit-rate",
        type=float,
        default=0.05,
        help="Minimum hit rate required for tuned configs.",
    )
    p_train.add_argument(
        "--strategy-min-bets",
        type=int,
        default=200,
        help="Minimum number of bets required for tuned configs.",
    )
    p_train.add_argument(
        "--strategy-top-n",
        type=int,
        default=10,
        help="Top configs to display after tuning.",
    )
    p_train.add_argument(
        "--strategy-log-every",
        type=int,
        default=20,
        help="Progress logging frequency during strategy tuning.",
    )
    p_train.add_argument(
        "--strategy-output",
        help="Optional CSV path to write strategy tuning results.",
    )
    p_train.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest", help="Backtest value strategy")
    p_bt.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_bt.add_argument("--commission", type=float, default=0.05)
    p_bt.add_argument("--min-ev", type=float, default=0.02)
    p_bt.add_argument("--min-edge", type=float, default=0.1)
    p_bt.add_argument("--max-spread", type=float, default=1.0)
    p_bt.add_argument("--max-price", type=float, default=200.0)
    p_bt.add_argument("--max-edge-mult", type=float, default=5.0)
    p_bt.add_argument(
        "--min-prob",
        type=float,
        help="Optional min probability to bet (defaults to tuned kappa threshold).",
    )
    p_bt.add_argument("--stake", type=float, default=1.0)
    p_bt.add_argument("--model-path")
    p_bt.add_argument("--calibrator-path")
    p_bt.add_argument(
        "--split-date",
        help="ISO date/time; backtest uses races on/after this (holdout set).",
    )
    p_bt.add_argument(
        "--split-last-month",
        action="store_true",
        help="Set split-date to the last 30 days of available data.",
    )
    p_bt.add_argument(
        "--split-last-180",
        dest="split_days",
        action="store_const",
        const=180,
        help="Set split-date to the last 180 days of available data.",
    )
    p_bt.add_argument(
        "--split-days",
        type=int,
        help="Set split-date to the last N days of available data.",
    )
    p_bt.add_argument("--run-id", help="Optional run id to align backtests/reports.")
    p_bt.add_argument(
        "--show-bets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a preview of top bets after backtest.",
    )
    p_bt.add_argument("--bets-limit", type=int, default=20, help="Rows to show in bet preview.")
    p_bt.set_defaults(func=cmd_backtest)

    p_score = sub.add_parser("score", help="Score today and output CSV")
    p_score.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--output")
    p_score.add_argument(
        "--min-prob",
        type=float,
        help="Optional min probability to include in scoring (defaults to tuned kappa threshold).",
    )
    p_score.add_argument("--budget", type=float, help="Optional day budget for stake allocation.")
    p_score.add_argument(
        "--allocation-method",
        choices=["fractional_kelly", "equal"],
        default="fractional_kelly",
        help="Stake allocation method when --budget is set.",
    )
    p_score.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (used by fractional_kelly method).",
    )
    p_score.add_argument(
        "--max-bet-pct",
        type=float,
        default=0.2,
        help="Max stake per bet as a fraction of daily budget.",
    )
    p_score.set_defaults(func=cmd_score)

    p_pub = sub.add_parser(
        "pub",
        aliases=["sheet"],
        help="Create today's manual betting sheet from live markets (no auto execution).",
    )
    p_pub.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_pub.add_argument("--dry-run", action="store_true", help="Use mock data even if credentials exist.")
    p_pub.add_argument(
        "--output",
        help="Output CSV path (defaults to artifacts/pub_sheet_<YYYY-MM-DD>.csv).",
    )
    p_pub.add_argument(
        "--min-prob",
        type=float,
        help="Optional min probability to include in the pub sheet.",
    )
    p_pub.add_argument("--budget", type=float, help="Optional day budget for stake allocation.")
    p_pub.add_argument(
        "--allocation-method",
        choices=["fractional_kelly", "equal"],
        default="fractional_kelly",
        help="Stake allocation method when --budget is set.",
    )
    p_pub.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fractional Kelly multiplier (used by fractional_kelly method).",
    )
    p_pub.add_argument(
        "--max-bet-pct",
        type=float,
        default=0.2,
        help="Max stake per bet as a fraction of daily budget.",
    )
    p_pub.set_defaults(func=cmd_pub)

    p_live = sub.add_parser(
        "live",
        help="Run live Betfair polling + model inference + dry-run/live execution.",
    )
    p_live.add_argument(
        "--config",
        default="config/live.yaml",
        help="Path to live YAML config (default: config/live.yaml).",
    )
    p_live.add_argument(
        "--write-config",
        help="Write a starter live config to this path and exit.",
    )
    p_live.add_argument(
        "--wirte-config",
        dest="write_config",
        help=argparse.SUPPRESS,
    )
    p_live.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file when used with --write-config.",
    )
    p_live_mode = p_live.add_mutually_exclusive_group()
    p_live_mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Override config and simulate orders without placing them.",
    )
    p_live_mode.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Override config and enable real order placement.",
    )
    p_live.set_defaults(dry_run=None)
    p_live.add_argument(
        "--once",
        action="store_true",
        help="Run one polling iteration and exit.",
    )
    p_live.add_argument(
        "--max-iterations",
        type=int,
        help="Override max polling iterations from config.",
    )
    p_live.add_argument(
        "--poll-interval-seconds",
        type=float,
        help="Override poll interval from config.",
    )
    p_live.add_argument(
        "--min-edge",
        type=float,
        help="Override strategy.min_edge from config.",
    )
    p_live.add_argument(
        "--max-stake-per-market",
        type=float,
        help="Override safety.max_stake_per_market from config.",
    )
    p_live.add_argument(
        "--max-daily-exposure",
        type=float,
        help="Override safety.max_daily_exposure from config.",
    )
    p_live.add_argument(
        "--ignore-within-minutes",
        type=float,
        help="Override safety.ignore_within_minutes from config.",
    )
    p_live.set_defaults(func=cmd_live)

    p_pipe = sub.add_parser("pipeline", help="Run ingest (optional), features, train, backtest, score")
    p_pipe.add_argument("--archive", help="Optional tar archive to ingest first")
    p_pipe.add_argument("--download-date", help="Optional date to download (dry-run friendly)")
    p_pipe.add_argument("--flush-every", type=int, default=5000, help="Flush size for ingest")
    p_pipe.add_argument("--progress-every", type=int, default=100, help="File progress print frequency")
    p_pipe.add_argument("--workers", type=int, default=None, help="Parallel workers for stream ingest")
    p_pipe.add_argument("--filter-au-win", action="store_true", help="Only keep AU WIN markets")
    p_pipe.add_argument("--force-ingest", action="store_true", help="Re-ingest even if data exists")
    p_pipe.add_argument(
        "--ingest-new",
        action="store_true",
        help="Only ingest new archive members (uses ingest manifest).",
    )
    p_pipe.add_argument(
        "--ingest-manifest",
        help="Path to ingest manifest JSON (default data/ingest_manifest.json).",
    )
    p_pipe.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_pipe.add_argument("--commission", type=float, default=0.05)
    p_pipe.add_argument("--min-ev", type=float, default=0.02)
    p_pipe.add_argument("--min-edge", type=float, default=0.1)
    p_pipe.add_argument("--max-spread", type=float, default=1.0)
    p_pipe.add_argument("--max-price", type=float, default=200.0)
    p_pipe.add_argument("--max-edge-mult", type=float, default=5.0)
    p_pipe.add_argument(
        "--min-prob",
        type=float,
        help="Optional min probability to bet (defaults to tuned kappa threshold).",
    )
    p_pipe.add_argument("--stake", type=float, default=1.0)
    p_pipe.add_argument(
        "--split-date",
        help="ISO date/time; train uses races before this, backtest uses on/after.",
    )
    p_pipe.add_argument(
        "--split-last-month",
        action="store_true",
        help="Set split-date to the last 30 days of available data.",
    )
    p_pipe.add_argument(
        "--split-last-180",
        dest="split_days",
        action="store_const",
        const=180,
        help="Set split-date to the last 180 days of available data.",
    )
    p_pipe.add_argument(
        "--split-days",
        type=int,
        help="Set split-date to the last N days of available data.",
    )
    p_pipe.add_argument("--run-id", help="Optional run id to align pipeline/reports.")
    p_pipe.add_argument(
        "--show-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a preview of top predictions after training.",
    )
    p_pipe.add_argument(
        "--preds-limit",
        type=int,
        default=20,
        help="Rows to show in the prediction preview.",
    )
    p_pipe.add_argument("--preds-min-ev", type=float, default=0.02, help="Min EV for preview rows.")
    p_pipe.add_argument(
        "--preds-min-edge", type=float, default=0.1, help="Min edge (relative) for preview rows."
    )
    p_pipe.add_argument(
        "--preds-min-prob",
        type=float,
        help="Optional min probability for preview rows (defaults to tuned kappa threshold).",
    )
    p_pipe.add_argument("--preds-max-price", type=float, default=200.0, help="Max price to show.")
    p_pipe.add_argument(
        "--preds-max-edge-mult",
        type=float,
        default=5.0,
        help="Max multiple of market implied prob to show.",
    )
    p_pipe.add_argument(
        "--preds-per-market",
        type=int,
        default=1,
        help="Max preview rows per market.",
    )
    p_pipe.add_argument(
        "--show-bets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a preview of top bets after backtest.",
    )
    p_pipe.add_argument("--bets-limit", type=int, default=20, help="Rows to show in bet preview.")
    p_pipe.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a report summary after the pipeline completes.",
    )
    p_pipe.add_argument(
        "--tune-strategy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune betting filters on out-of-fold predictions (use --no-tune-strategy to disable).",
    )
    p_pipe.add_argument(
        "--strategy-grid",
        choices=["small", "medium", "large"],
        default="small",
        help="Grid size for strategy tuning.",
    )
    p_pipe.add_argument(
        "--strategy-objective",
        choices=["expected_roi", "roi", "expected_profit", "profit"],
        default="expected_roi",
        help="Metric to maximize during strategy tuning.",
    )
    p_pipe.add_argument(
        "--strategy-min-hit-rate",
        type=float,
        default=0.05,
        help="Minimum hit rate required for tuned configs.",
    )
    p_pipe.add_argument(
        "--strategy-min-bets",
        type=int,
        default=200,
        help="Minimum number of bets required for tuned configs.",
    )
    p_pipe.add_argument(
        "--strategy-top-n",
        type=int,
        default=10,
        help="Top configs to display after tuning.",
    )
    p_pipe.add_argument(
        "--strategy-log-every",
        type=int,
        default=20,
        help="Progress logging frequency during strategy tuning.",
    )
    p_pipe.add_argument(
        "--strategy-output",
        help="Optional CSV path to write strategy tuning results.",
    )
    p_pipe.add_argument("--output")
    p_pipe.add_argument("--dry-run", action="store_true", help="Applied to scoring step")
    p_pipe.add_argument("--skip-features", action="store_true")
    p_pipe.add_argument("--skip-train", action="store_true")
    p_pipe.add_argument("--skip-backtest", action="store_true")
    p_pipe.add_argument("--skip-score", action="store_true")
    p_pipe.set_defaults(func=cmd_pipeline)

    p_go = sub.add_parser(
        "go",
        aliases=["auto"],
        help="Run the default optimized flow (historic download -> incremental pipeline).",
    )
    p_go.set_defaults(func=cmd_go)

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
                min_edge=0.1,
                max_spread=1.0,
                max_price=200.0,
                max_edge_mult=5.0,
                stake=1.0,
                output=None,
                dry_run=True,
                skip_features=False,
                skip_train=False,
                skip_backtest=False,
                skip_score=False,
                show_preds=True,
                preds_limit=20,
                preds_min_ev=0.02,
                preds_min_edge=0.1,
                preds_max_price=200.0,
                preds_max_edge_mult=5.0,
                preds_per_market=1,
                show_bets=True,
                bets_limit=20,
                report=True,
                tune_strategy=True,
                strategy_grid="small",
                strategy_objective="expected_roi",
                strategy_min_hit_rate=0.05,
                strategy_min_bets=200,
                strategy_top_n=10,
                strategy_log_every=20,
                strategy_output=None,
            )
        )
    )

    p_status = sub.add_parser("status", help="Show row counts in DuckDB tables")
    p_status.set_defaults(func=cmd_status)

    p_report = sub.add_parser("report", help="Show latest model metrics and backtest summary")
    p_report.add_argument("--run-id", help="Optional run id to report on.")
    p_report.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    """Entry point for the unified CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
