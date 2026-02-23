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
from shared.form.ingest import load_external_form_rows
from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.market_type import bucket_market_type
from shared.model.predictions import build_prediction_preview
from shared.model.tab_translation import estimate_tab_odds_quantiles, quantile_to_column
from shared.model.training import (
    DEFAULT_CALIBRATION_RANDOMIZE_WITHIN_WINDOWS,
    DEFAULT_CALIBRATION_RANDOM_STATE,
    DEFAULT_CALIBRATION_WINDOW_SAMPLE_FRACTION,
    load_model_and_calibrator,
    predict_probabilities,
    train_and_calibrate,
)
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.bet_explain import preview_legend_lines
from shared.utils.cli_presets import (
    go_historic_args,
    go_historic_stale_days_default,
    go_pipeline_args,
)
from shared.utils.manifest_repair import repair_manifests
from shared.utils.market_types import (
    api_market_type_codes,
    market_type_matches,
    normalize_market_type_tokens,
    tokens_to_filter_set,
)
from shared.utils.progress import log
from workflow.ingest_archive import ingest_archive_file

DEFAULT_EXTERNAL_FORM_SOURCE = "external_form_provider"
DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES = 10
DEFAULT_EXTERNAL_FORM_INPUT_CANDIDATES = (
    pathlib.Path("data/external_form_runs.parquet"),
    pathlib.Path("data/external_form_runs.csv"),
    pathlib.Path("data/external_form_runs.json"),
    pathlib.Path("data/external_form_runs.jsonl"),
    pathlib.Path("data/external_form.json"),
)


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
    bad_members_output_value = getattr(args, "bad_members_output", "artifacts/ingest_bad_stream_members.txt")
    bad_members_output_path = pathlib.Path(bad_members_output_value) if bad_members_output_value else None
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
        bad_members_output=bad_members_output_path,
    )
    log(f"Ingested rows: {counts}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download Betfair markets/snapshots/results for a date (mocked in dry-run)."""
    from workflow.download_data import (  # local import to reuse helpers
        capture_snapshots,
        extract_runner_metadata_snapshots,
    )

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
    metadata_rows = extract_runner_metadata_snapshots(markets, source="list_market_catalogue")
    store.append_runner_metadata_snapshots(metadata_rows)
    snapshots = capture_snapshots(client, market_ids, offsets=[60, 30, 10, 5, 2, 1])
    store.append_snapshots(snapshots)
    results = client.fetch_market_results(market_ids)
    store.upsert_results(results)
    log(
        "Downloaded "
        f"{len(markets)} markets, {len(runner_rows)} runners, "
        f"{len(metadata_rows)} metadata snapshots, {len(snapshots)} price snapshots."
    )


def cmd_ingest_form(args: argparse.Namespace) -> None:
    """Ingest licensed external form-run snapshots so P1 features are reproducible at inference cutoff."""
    store = DuckDBStore()
    rows = load_external_form_rows(
        args.input,
        source=args.source,
        default_cutoff_minutes=args.default_cutoff_minutes,
        default_snapshot_time=args.default_snapshot_time,
    )
    if not rows:
        log("Ingest form: no valid rows found in input payload.")
        return
    store.append_external_runner_form_runs(rows)
    log(f"Ingest form: wrote {len(rows)} rows from {args.input}.")


def cmd_download_historic(args: argparse.Namespace) -> None:
    """Download historic data packages using Betfair's historic API."""
    from workflow.download_historic import run_download_historic

    run_download_historic(args)


def cmd_repair_manifests(args: argparse.Namespace) -> None:
    """Repair ingest/download manifests by pruning known-bad basenames so retries can recover data holes."""
    try:
        summary = repair_manifests(
            bad_list_path=pathlib.Path(args.bad_list),
            historic_manifest_path=pathlib.Path(args.historic_manifest),
            ingest_manifest_path=pathlib.Path(args.ingest_manifest),
            backup=bool(args.backup),
        )
    except FileNotFoundError as exc:
        log(str(exc))
        return
    log(
        "Repair manifests: "
        f"bad_basenames={summary['bad_basenames']} "
        f"historic_removed={summary['historic_removed']} "
        f"ingest_removed={summary['ingest_removed']}."
    )
    for backup_path in summary.get("backup_paths", []):
        log(f"Repair manifests: backup -> {backup_path}")


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

    markets_overrides = {}
    if getattr(args, "market_types", None) is not None:
        markets_overrides["market_type_codes"] = normalize_market_type_tokens(args.market_types)
    if markets_overrides:
        overrides["markets"] = markets_overrides

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


def _load_model_metrics(args: argparse.Namespace, store: DuckDBStore) -> dict | None:
    """Load metrics JSON for the target run/cutoff so threshold routing can stay centralized."""
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
    return metrics if isinstance(metrics, dict) else None


def _resolve_min_prob(args: argparse.Namespace, store: DuckDBStore) -> float | None:
    """Resolve the active min-prob threshold, only using tuned kappa when explicitly requested."""
    if getattr(args, "min_prob", None) is not None:
        return float(args.min_prob)
    if not getattr(args, "use_kappa_thresholds", False):
        return None
    metrics = _load_model_metrics(args, store)
    if not metrics:
        return None
    value = metrics.get("kappa_threshold")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_market_type_min_probs(args: argparse.Namespace, store: DuckDBStore) -> dict[str, float]:
    """Resolve per-market-type min-prob thresholds when explicitly requested."""
    if getattr(args, "min_prob", None) is not None:
        return {}
    if not getattr(args, "use_market_type_kappa_thresholds", False):
        return {}
    metrics = _load_model_metrics(args, store)
    if not metrics:
        return {}
    bucket_metrics = metrics.get("market_type_models")
    if not isinstance(bucket_metrics, dict):
        return {}
    thresholds: dict[str, float] = {}
    for bucket, payload in bucket_metrics.items():
        if not isinstance(payload, dict):
            continue
        threshold = payload.get("kappa_threshold")
        if isinstance(threshold, (int, float)):
            thresholds[str(bucket).upper()] = float(threshold)
    return thresholds


def _apply_probability_thresholds(
    frame: pd.DataFrame,
    label: str,
    global_min_prob: float | None,
    bucket_min_probs: dict[str, float],
) -> pd.DataFrame:
    """Apply either one global min-prob or per-bucket thresholds to prediction frames."""
    if frame.empty:
        return frame
    if "p_hat" not in frame.columns:
        return frame

    before = len(frame)
    filtered = frame[frame["p_hat"].notna()].copy()
    dropped_nan = before - len(filtered)
    if dropped_nan:
        log(f"{label}: dropped {dropped_nan} rows with missing predictions.")
    if filtered.empty:
        return filtered

    if global_min_prob is not None and not bucket_min_probs:
        kept = filtered[filtered["p_hat"] >= global_min_prob].copy()
        log(f"{label}: applied min_prob={global_min_prob:.3f} ({len(kept)}/{len(filtered)} rows).")
        return kept

    if bucket_min_probs:
        if "market_type" in filtered.columns:
            bucket_series = filtered["market_type"].apply(bucket_market_type).str.upper()
            threshold_series = bucket_series.map(bucket_min_probs)
        else:
            threshold_series = pd.Series(index=filtered.index, dtype=float)
        if global_min_prob is not None:
            threshold_series = threshold_series.fillna(global_min_prob)
        match_count = int(threshold_series.notna().sum())
        keep_mask = threshold_series.isna() | (filtered["p_hat"] >= threshold_series)
        kept = filtered.loc[keep_mask].copy()
        log(
            f"{label}: applied market-type min_prob thresholds "
            f"({len(kept)}/{len(filtered)} rows; matched={match_count})."
        )
        return kept

    return filtered

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


def _filter_to_market_types(feature_df: pd.DataFrame, label: str, market_tokens: list[str]) -> pd.DataFrame:
    """Filter rows by market type tokens so train/backtest/score can align to user scope."""
    selected = tokens_to_filter_set(market_tokens)
    if selected is None:
        return feature_df
    if feature_df.empty or "market_type" not in feature_df.columns:
        return feature_df.iloc[0:0].copy()
    mask = feature_df["market_type"].apply(lambda value: market_type_matches(value, selected))
    filtered = feature_df.loc[mask].copy()
    log(f"{label}: market-type filter kept {len(filtered)}/{len(feature_df)} rows.")
    return filtered


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


def _select_score_price(feature_row: pd.Series, cutoff_minutes: int) -> float | None:
    """Pick a usable Betfair price for scoring so sparse offsets do not drop otherwise valid rows."""
    candidate = feature_row.get(f"back_price_t{cutoff_minutes}")
    if pd.notnull(candidate):
        return float(candidate)
    for offset in [60, 30, 10, 5, 2, 1]:
        fallback = feature_row.get(f"back_price_t{offset}")
        if pd.notnull(fallback):
            return float(fallback)
    return None


def _apply_score_strategy_filters(
    frame: pd.DataFrame,
    cutoff_minutes: int,
    min_ev: float | None,
    min_edge: float | None,
    max_price: float | None,
    max_edge_multiplier: float | None,
    max_spread: float | None,
) -> pd.DataFrame:
    """Apply backtest-style EV/edge/price/spread filters so score output matches strategy assumptions."""
    if frame.empty:
        return frame
    filtered = frame.copy()
    filtered["implied_prob"] = 1.0 / filtered["decision_price"]
    filtered["edge_pct"] = (filtered["p_hat"] - filtered["implied_prob"]) / filtered["implied_prob"]
    filtered["edge_multiplier"] = filtered["p_hat"] / filtered["implied_prob"]

    before = len(filtered)
    if min_ev is not None:
        filtered = filtered[filtered["ev"] >= min_ev]
    if min_edge is not None:
        filtered = filtered[filtered["edge_pct"] >= min_edge]
    if max_price is not None:
        filtered = filtered[filtered["decision_price"] <= max_price]
    if max_edge_multiplier is not None:
        filtered = filtered[filtered["edge_multiplier"] <= max_edge_multiplier]
    spread_col = f"spread_t{cutoff_minutes}"
    if max_spread is not None and spread_col in filtered.columns:
        filtered = filtered[filtered[spread_col].isna() | (filtered[spread_col] <= max_spread)]
    log(f"Score: strategy filters kept {len(filtered)}/{before} rows.")
    return filtered


def _resolve_execution_domain(args: argparse.Namespace) -> str:
    """Resolve scoring domain defaults so pub can stay TAB-first while score stays Betfair-first."""
    explicit = getattr(args, "execution_domain", None)
    if explicit is not None:
        return str(explicit).strip().lower()
    command = str(getattr(args, "command", "")).strip().lower()
    if command in {"pub", "sheet"}:
        return "tab"
    return "betfair"


def _parse_tab_quantile(raw_value: float | None) -> float:
    """Validate TAB decision quantile so conservative EV gating always uses a valid probability quantile."""
    quantile = 0.10 if raw_value is None else float(raw_value)
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError("--tab-odds-quantile must be between 0 and 1 (exclusive).")
    return quantile


def _resolve_external_form_input_path(explicit_input: str | None) -> pathlib.Path | None:
    """Resolve the external-form input path so default-on form features work without manual file flags."""
    if explicit_input:
        candidate = pathlib.Path(explicit_input).expanduser()
        return candidate if candidate.exists() else None
    for candidate in DEFAULT_EXTERNAL_FORM_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _maybe_auto_ingest_external_form(args: argparse.Namespace, store: DuckDBStore) -> None:
    """Auto-ingest external form data once per command run so form features are enabled by default."""
    if getattr(args, "_external_form_auto_ingest_attempted", False):
        return
    setattr(args, "_external_form_auto_ingest_attempted", True)

    if not getattr(args, "external_form_ingest", True):
        log("External form ingest: disabled by CLI flag (--no-external-form-ingest).")
        return

    explicit_input = getattr(args, "external_form_input", None)
    input_path = _resolve_external_form_input_path(explicit_input)
    if input_path is None:
        if explicit_input:
            log(f"External form ingest: input not found at {explicit_input}; skipping.")
        return

    existing_rows = store.table_row_count("external_runner_form_runs")
    if existing_rows > 0:
        return

    default_cutoff = getattr(
        args,
        "external_form_default_cutoff_minutes",
        DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES,
    )
    source = getattr(args, "external_form_source", DEFAULT_EXTERNAL_FORM_SOURCE)
    rows = load_external_form_rows(
        str(input_path),
        source=str(source),
        default_cutoff_minutes=int(default_cutoff),
        default_snapshot_time=getattr(args, "external_form_default_snapshot_time", None),
    )
    if not rows:
        log(f"External form ingest: no valid rows found in {input_path}; skipping.")
        return
    store.append_external_runner_form_runs(rows)
    log(f"External form ingest: wrote {len(rows)} rows from {input_path}.")


def cmd_features(args: argparse.Namespace) -> None:
    """Build features at a cutoff minute and save Parquet."""
    print(f"Building features (cutoff T-{args.cutoff_minutes})...")
    store = DuckDBStore()
    _maybe_auto_ingest_external_form(args, store)
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
    _maybe_auto_ingest_external_form(args, store)
    market_tokens = normalize_market_type_tokens(getattr(args, "market_types", "ALL"))
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
    train_features = _filter_to_market_types(train_features, "Training", market_tokens)
    if test_features is not None:
        test_features = _filter_to_market_types(test_features, "Holdout", market_tokens)
    if train_features.empty:
        log("Training: no features available after split; skipping.")
        return
    sample_fraction = float(
        getattr(
            args,
            "calibration_window_sample_fraction",
            DEFAULT_CALIBRATION_WINDOW_SAMPLE_FRACTION,
        )
    )
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError("--calibration-window-sample-fraction must be in the range (0, 1].")
    model_path, calibrator_path, metrics, oof_predictions = train_and_calibrate(
        features_df=train_features,
        cutoff_minutes=args.cutoff_minutes,
        store=store,
        split_date=getattr(args, "split_date", None),
        calibration_randomize_within_windows=getattr(
            args,
            "calibration_randomize_within_windows",
            DEFAULT_CALIBRATION_RANDOMIZE_WITHIN_WINDOWS,
        ),
        calibration_window_sample_fraction=sample_fraction,
        calibration_random_state=getattr(
            args,
            "calibration_random_state",
            DEFAULT_CALIBRATION_RANDOM_STATE,
        ),
        run_id=getattr(args, "run_id", None),
    )
    log(f"Model saved: {model_path}, calibrator: {calibrator_path}, metrics: {metrics}")
    if getattr(args, "show_preds", True):
        tuned_min_prob = None
        if isinstance(metrics, dict):
            tuned_min_prob = metrics.get("kappa_threshold")
        preview_min_prob = getattr(args, "preds_min_prob", None)
        if preview_min_prob is None and getattr(args, "preds_use_kappa_threshold", False):
            preview_min_prob = tuned_min_prob
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
            strategy_min_prob = getattr(args, "strategy_min_prob", None)
            if strategy_min_prob is None:
                strategy_min_prob = getattr(args, "min_prob", None)
            if strategy_min_prob is None and getattr(args, "use_kappa_thresholds", False):
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
    _maybe_auto_ingest_external_form(args, store)
    market_tokens = normalize_market_type_tokens(getattr(args, "market_types", "ALL"))
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
    features = _filter_to_market_types(features, "Backtest", market_tokens)
    if features.empty:
        log("Backtest: no features available after split; skipping.")
        return
    global_min_prob = _resolve_min_prob(args, store)
    bucket_min_probs = _resolve_market_type_min_probs(args, store)
    if global_min_prob is not None and not bucket_min_probs:
        source = "CLI override" if getattr(args, "min_prob", None) is not None else "kappa tuning"
        log(f"Backtest: using min_prob={global_min_prob:.3f} ({source}).")
    elif bucket_min_probs:
        log(
            "Backtest: using market-type min_prob thresholds "
            f"for {sorted(bucket_min_probs.keys())}."
        )
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
    prediction_frame = features.copy()
    prediction_frame["p_hat"] = probs
    prediction_frame = _apply_probability_thresholds(
        prediction_frame,
        label="Backtest",
        global_min_prob=global_min_prob,
        bucket_min_probs=bucket_min_probs,
    )
    if prediction_frame.empty:
        log("Backtest: no rows remain after probability thresholds.")
        return
    features = prediction_frame.drop(columns=["p_hat"])
    probs = prediction_frame["p_hat"]
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
        min_prob=None,
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
    from shared.backtest.engine import compute_expected_value
    from workflow.score_today import _capture_latest_snapshots

    today = dt.date.today()
    execution_domain = _resolve_execution_domain(args)
    if execution_domain not in {"betfair", "tab"}:
        raise ValueError("--execution-domain must be either 'betfair' or 'tab'.")
    tab_decision_quantile = _parse_tab_quantile(getattr(args, "tab_odds_quantile", None))
    tab_decision_column = quantile_to_column(tab_decision_quantile)
    tab_model_path = getattr(args, "tab_translation_model", None)
    tab_fallback_haircut = float(getattr(args, "tab_fallback_haircut", 0.08))
    tab_fallback_spread = float(getattr(args, "tab_fallback_spread", 0.10))

    market_tokens = normalize_market_type_tokens(getattr(args, "market_types", "ALL"))
    selected = tokens_to_filter_set(market_tokens)
    client = BetfairClient()
    if args.dry_run:
        client._dry_run = True  # type: ignore[attr-defined]
    store = DuckDBStore()
    _maybe_auto_ingest_external_form(args, store)

    markets = client.list_markets_for_date(today, market_types=api_market_type_codes(market_tokens))
    if selected is not None:
        markets = [market for market in markets if market_type_matches(market.get("market_type"), selected)]
        log(f"Score: market-type filter kept {len(markets)} markets.")
    if not markets:
        log("Score: no markets returned for selected market types.")
        return
    store.upsert_markets(markets)
    market_ids = [m["market_id"] for m in markets]
    runner_rows = [r for m in markets for r in m["runners"]]
    store.upsert_runners(runner_rows)

    snapshots = _capture_latest_snapshots(client, market_ids, cutoff_minutes=args.cutoff_minutes)
    store.append_snapshots(snapshots)

    features = build_features_from_store(
        store,
        cutoff_minutes=args.cutoff_minutes,
        market_ids=market_ids,
    )
    before_market_scope = len(features)
    features = features[features["market_id"].astype(str).isin(set(market_ids))].copy()
    log(f"Score: market-id scope kept {len(features)}/{before_market_scope} rows.")
    features = _filter_to_market_types(features, "Score", market_tokens)
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

    features["p_hat"] = probs
    global_min_prob = _resolve_min_prob(args, store)
    bucket_min_probs = _resolve_market_type_min_probs(args, store)
    features = _apply_probability_thresholds(
        features,
        label="Score",
        global_min_prob=global_min_prob,
        bucket_min_probs=bucket_min_probs,
    )
    if features.empty:
        log("No rows remain after min_prob filter.")
        return
    features["betfair_price"] = features.apply(
        lambda row: _select_score_price(row, args.cutoff_minutes),
        axis=1,
    )
    before_price = len(features)
    features = features[features["betfair_price"].notna()].copy()
    features = features[features["betfair_price"] > 0].copy()
    if features.empty:
        log(f"Score: no rows with a usable price ({len(features)}/{before_price} rows).")
        return

    tab_q10_col = quantile_to_column(0.10)
    tab_q50_col = quantile_to_column(0.50)
    tab_q90_col = quantile_to_column(0.90)
    commission_rate = 0.05
    if execution_domain == "tab":
        tab_prices = estimate_tab_odds_quantiles(
            feature_df=features,
            cutoff_minutes=args.cutoff_minutes,
            quantiles=[0.10, 0.50, 0.90, tab_decision_quantile],
            model_path=tab_model_path,
            fallback_haircut=tab_fallback_haircut,
            fallback_spread=tab_fallback_spread,
        )
        features = features.join(tab_prices)
        features["decision_price"] = features.get(tab_decision_column)
        commission_rate = 0.0
        log(
            "Score: "
            f"execution-domain=tab, decision-quantile={tab_decision_quantile:.2f}, "
            f"model={tab_model_path or 'fallback-only'}."
        )
    else:
        features["decision_price"] = features["betfair_price"]
        features["tab_price_source"] = "betfair"
        features[tab_q10_col] = pd.NA
        features[tab_q50_col] = pd.NA
        features[tab_q90_col] = pd.NA

    before_decision = len(features)
    features = features[features["decision_price"].notna()].copy()
    features = features[features["decision_price"] > 0].copy()
    if features.empty:
        log(f"Score: no rows with decision prices ({len(features)}/{before_decision} rows).")
        return

    features["ev"] = features.apply(
        lambda row: compute_expected_value(
            prob=float(row["p_hat"]),
            price=float(row["decision_price"]),
            commission=commission_rate,
        ),
        axis=1,
    )
    features = _apply_score_strategy_filters(
        frame=features,
        cutoff_minutes=args.cutoff_minutes,
        min_ev=getattr(args, "min_ev", None),
        min_edge=getattr(args, "min_edge", None),
        max_price=getattr(args, "max_price", None),
        max_edge_multiplier=getattr(args, "max_edge_mult", None),
        max_spread=getattr(args, "max_spread", None),
    )
    if features.empty:
        log("Score: no rows remain after strategy filters.")
        return

    runners = store.load_runners()
    if not runners.empty:
        features = features.merge(runners, on=["market_id", "selection_id"], how="left")

    best = (
        features.sort_values("ev", ascending=False)
        .groupby("market_id")
        .head(1)
        .reset_index(drop=True)
    )

    best["price"] = best.get("decision_price")
    budget = getattr(args, "budget", None)
    if budget is not None:
        if float(budget) <= 0:
            raise ValueError("--budget must be > 0.")
        best = allocate_stakes_from_budget(
            best,
            budget=float(budget),
            commission=commission_rate,
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
                "execution_domain": execution_domain,
                "price": row.get("decision_price"),
                "betfair_price": row.get("betfair_price"),
                "tab_decision_quantile": tab_decision_quantile if execution_domain == "tab" else None,
                "tab_price_source": row.get("tab_price_source"),
                "tab_price_q10": row.get(tab_q10_col),
                "tab_price_q50": row.get(tab_q50_col),
                "tab_price_q90": row.get(tab_q90_col),
                "p_hat": row["p_hat"],
                "expected_value": row["ev"],
                "commission_rate": commission_rate,
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
    """Generate a TAB-focused manual-betting day sheet with conservative domain-adapted pricing.

    Why: pub mode targets manual TAB execution, so it defaults to conservative TAB pricing assumptions
    while keeping CLI usage simple for everyday operation.
    """
    if not getattr(args, "command", None):
        args.command = "pub"
    if getattr(args, "execution_domain", None) is None:
        args.execution_domain = "tab"
    if getattr(args, "tab_odds_quantile", None) is None:
        args.tab_odds_quantile = 0.10
    if getattr(args, "tab_fallback_haircut", None) is None:
        args.tab_fallback_haircut = 0.08
    if getattr(args, "tab_fallback_spread", None) is None:
        args.tab_fallback_spread = 0.10
    if getattr(args, "tab_translation_model", None) is None:
        default_model = pathlib.Path(f"artifacts/tab_translation_cutoff_{args.cutoff_minutes}.joblib")
        if default_model.exists():
            args.tab_translation_model = str(default_model)

    if not getattr(args, "output", None):
        args.output = f"artifacts/pub_sheet_{dt.date.today().isoformat()}.csv"
    log(
        "Pub sheet: "
        f"writing to {args.output} "
        f"(execution-domain={args.execution_domain}, q={float(args.tab_odds_quantile):.2f})."
    )
    cmd_score(args)


def cmd_status(args: argparse.Namespace) -> None:
    """Print simple row counts for sanity checks."""
    store = DuckDBStore()
    with store._connect() as con:  # type: ignore[attr-defined]
        tables = [
            "markets",
            "runners",
            "runner_metadata_snapshots",
            "external_runner_form_runs",
            "snapshots",
            "results",
            "bets",
            "model_runs",
            "tab_quotes",
            "tab_executions",
        ]
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
                WITH dedup AS (
                    SELECT DISTINCT
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
                    FROM bets
                    WHERE run_id = ?
                )
                SELECT
                    COUNT(*) AS bets,
                    SUM(stake) AS turnover,
                    SUM(result_profit) AS profit,
                    SUM(expected_value * stake) AS expected_profit,
                    SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate
                FROM dedup
                """,
                [run_id],
            ).fetchone()
        else:
            bet_row = con.execute(
                """
                WITH dedup AS (
                    SELECT DISTINCT
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
                    FROM bets
                )
                SELECT
                    COUNT(*) AS bets,
                    SUM(stake) AS turnover,
                    SUM(result_profit) AS profit,
                    SUM(expected_value * stake) AS expected_profit,
                    SUM(CASE WHEN result_profit > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate
                FROM dedup
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
    if hasattr(store, "load_runner_metadata_completeness"):
        completeness = store.load_runner_metadata_completeness()
        if not completeness.empty:
            latest = completeness.iloc[0]
            log(
                "Runner metadata completeness: "
                f"date={latest['snapshot_date']} rows={int(latest['row_count'])} "
                f"jockey={latest['jockey_name_coverage']:.2%} "
                f"trainer={latest['trainer_name_coverage']:.2%} "
                f"official_rating={latest['official_rating_coverage']:.2%} "
                f"stall_draw={latest['stall_draw_coverage']:.2%}."
            )
    if hasattr(store, "load_external_runner_form_completeness"):
        external_form = store.load_external_runner_form_completeness()
        if not external_form.empty:
            latest = external_form.iloc[0]
            log(
                "External form completeness: "
                f"date={latest['snapshot_date']} rows={int(latest['row_count'])} "
                f"run_date={latest['run_date_coverage']:.2%} "
                f"finish_pos={latest['run_finish_pos_coverage']:.2%} "
                f"sectional={latest['run_sectional_coverage']:.2%} "
                f"speed_rating={latest['run_speed_rating_coverage']:.2%}."
            )

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run ingest (optional), build features, train, backtest, and score in one go."""
    log("Starting pipeline...")
    _ensure_run_id(args)
    decision_market_types = (
        getattr(args, "decision_market_types", None) or getattr(args, "market_types", "ALL")
    )
    if str(decision_market_types).strip().upper() == "ALL":
        log(
            "Pipeline: decision-market-types=ALL applies one strategy across all market buckets; "
            "use --decision-market-types WIN (recommended) for cleaner comparability."
        )
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
        backtest_args = argparse.Namespace(**vars(args))
        backtest_args.market_types = decision_market_types
        cmd_backtest(backtest_args)
    if not args.skip_score:
        log("Step: score")
        score_args = argparse.Namespace(**vars(args))
        score_args.market_types = decision_market_types
        score_args.output = score_args.output or f"artifacts/pipeline_score_cutoff_{args.cutoff_minutes}.csv"
        cmd_score(score_args)
    if getattr(args, "report", False):
        log("Step: report")
        cmd_report(args)
    log("Pipeline complete.")


def _as_utc_datetime(value: object) -> dt.datetime | None:
    """Normalize timestamp-like values to timezone-aware UTC datetimes for stale-data checks."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    return None


def cmd_go(args: argparse.Namespace) -> None:
    """Run the default optimized end-to-end flow with no tuning flags required.

    Why: provide a single CLI command for users who want the recommended path without
    memorizing command options. Advanced users can still tune each subcommand directly.
    """
    run_historic_download = True
    stale_days = int(getattr(args, "refresh_historic_if_stale_days", go_historic_stale_days_default()))
    if not getattr(args, "refresh_historic", False):
        try:
            store = DuckDBStore()
            snapshot_rows = store.table_row_count("snapshots")
            if snapshot_rows > 0:
                if stale_days < 0:
                    run_historic_download = False
                    log(
                        "Go: snapshots already present "
                        f"({snapshot_rows} rows); stale-data auto-refresh disabled. "
                        "Use --refresh-historic to force download."
                    )
                else:
                    latest_snapshot = _as_utc_datetime(store.max_snapshot_time())
                    if latest_snapshot is None:
                        log(
                            "Go: snapshots exist but latest snapshot timestamp is unavailable; "
                            "running historic download to avoid stale gaps."
                        )
                    else:
                        now_utc = dt.datetime.now(dt.timezone.utc)
                        age_days = (now_utc - latest_snapshot).total_seconds() / 86400.0
                        if age_days > float(stale_days):
                            log(
                                "Go: latest snapshot age "
                                f"{age_days:.1f} days exceeds stale threshold ({stale_days} days); "
                                "running historic download."
                            )
                        else:
                            run_historic_download = False
                            log(
                                "Go: snapshots already present "
                                f"({snapshot_rows} rows, latest {latest_snapshot.date()}); "
                                f"skipping historic download (stale threshold {stale_days} days)."
                            )
        except Exception as exc:  # pragma: no cover - defensive fallback around local DB state
            log(f"Go: could not inspect snapshots ({exc}); proceeding with historic download.")
    parser = build_parser()

    if run_historic_download:
        historic_tokens = ["download-historic", *go_historic_args()]
        log(f"Go: running `punter {' '.join(historic_tokens)}`")
        historic_args = parser.parse_args(historic_tokens)
        historic_args.func(historic_args)
    else:
        log("Go: skipping `punter download-historic`.")

    pipeline_tokens = ["pipeline", *go_pipeline_args()]
    if not getattr(args, "external_form_ingest", True):
        pipeline_tokens.append("--no-external-form-ingest")

    external_form_input = getattr(args, "external_form_input", None)
    if external_form_input:
        pipeline_tokens.extend(["--external-form-input", str(external_form_input)])

    external_form_source = getattr(args, "external_form_source", DEFAULT_EXTERNAL_FORM_SOURCE)
    if external_form_source != DEFAULT_EXTERNAL_FORM_SOURCE:
        pipeline_tokens.extend(["--external-form-source", str(external_form_source)])

    external_form_cutoff = getattr(
        args,
        "external_form_default_cutoff_minutes",
        DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES,
    )
    if int(external_form_cutoff) != DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES:
        pipeline_tokens.extend(["--external-form-default-cutoff-minutes", str(int(external_form_cutoff))])

    external_form_snapshot_time = getattr(args, "external_form_default_snapshot_time", None)
    if external_form_snapshot_time:
        pipeline_tokens.extend(["--external-form-default-snapshot-time", str(external_form_snapshot_time)])

    log(f"Go: running `punter {' '.join(pipeline_tokens)}`")
    pipeline_args = parser.parse_args(pipeline_tokens)
    pipeline_args.func(pipeline_args)


def _add_execution_domain_arguments(
    parser: argparse.ArgumentParser,
    default_execution_domain: str,
) -> None:
    """Attach domain-adaptation flags so pub/score can share one consistent TAB pricing interface."""
    parser.add_argument(
        "--execution-domain",
        choices=["betfair", "tab"],
        default=default_execution_domain,
        help=(
            "Price domain used for EV and stake sizing. "
            "Use tab for manual TAB execution and betfair for exchange-style scoring."
        ),
    )
    parser.add_argument(
        "--tab-translation-model",
        help=(
            "Optional path to a TAB translation model bundle. "
            "If omitted or missing, pub mode uses a conservative haircut fallback."
        ),
    )
    parser.add_argument(
        "--tab-odds-quantile",
        type=float,
        default=0.10,
        help="Conservative TAB odds quantile used for EV gating when --execution-domain=tab.",
    )
    parser.add_argument(
        "--tab-fallback-haircut",
        type=float,
        default=0.08,
        help="Fallback percentage haircut vs Betfair odds when TAB translation model is unavailable.",
    )
    parser.add_argument(
        "--tab-fallback-spread",
        type=float,
        default=0.10,
        help="Fallback uncertainty spread used to derive TAB odds quantile bands around the median.",
    )


def _add_external_form_auto_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach default-on external form ingest flags so new form layers are active unless explicitly disabled."""
    parser.add_argument(
        "--external-form-ingest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically ingest external form data when a default file is found and the table is empty "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--external-form-input",
        help=(
            "Optional explicit path for auto-ingest input. "
            "If omitted, common defaults in data/ are auto-detected."
        ),
    )
    parser.add_argument(
        "--external-form-source",
        default=DEFAULT_EXTERNAL_FORM_SOURCE,
        help="Source label used when auto-ingesting external form rows.",
    )
    parser.add_argument(
        "--external-form-default-cutoff-minutes",
        type=int,
        default=DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES,
        help="Fallback cutoff used by auto external-form ingestion when rows omit seconds_to_start.",
    )
    parser.add_argument(
        "--external-form-default-snapshot-time",
        help="Fallback snapshot timestamp (ISO) used by auto external-form ingestion when missing in rows.",
    )


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
    p_ingest.add_argument(
        "--bad-members-output",
        default="artifacts/ingest_bad_stream_members.txt",
        help="Path to append invalid stream-member basenames detected during ingest (set empty to disable).",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_form = sub.add_parser(
        "ingest-form",
        help="Ingest external licensed form-run exports (CSV/JSON/JSONL/Parquet).",
    )
    p_form.add_argument("--input", required=True, help="Path to external form export file.")
    p_form.add_argument(
        "--source",
        default=DEFAULT_EXTERNAL_FORM_SOURCE,
        help="Source label stored with rows (e.g., punting_form).",
    )
    p_form.add_argument(
        "--default-cutoff-minutes",
        type=int,
        default=DEFAULT_EXTERNAL_FORM_DEFAULT_CUTOFF_MINUTES,
        help="Fallback cutoff used when snapshot times are missing in the export.",
    )
    p_form.add_argument(
        "--default-snapshot-time",
        help="Optional fallback snapshot timestamp (ISO) when input rows omit snapshot_time.",
    )
    p_form.set_defaults(func=cmd_ingest_form)

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
    _add_external_form_auto_ingest_arguments(p_feat)
    p_feat.set_defaults(func=cmd_features)

    p_train = sub.add_parser("train", help="Train model at cutoff")
    p_train.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_train.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types/buckets (e.g., WIN,PLACE,EXOTIC,EXACTA). ALL disables filtering.",
    )
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
    p_train.add_argument(
        "--preds-min-ev",
        type=float,
        default=None,
        help="Optional min EV for preview rows (disabled by default).",
    )
    p_train.add_argument(
        "--preds-min-edge",
        type=float,
        default=None,
        help="Optional min edge (relative) for preview rows (disabled by default).",
    )
    p_train.add_argument(
        "--preds-min-prob",
        type=float,
        help="Optional min probability for preview rows.",
    )
    p_train.add_argument(
        "--preds-use-kappa-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the tuned kappa threshold for preview min-prob when --preds-min-prob "
            "is not set (disabled by default)."
        ),
    )
    p_train.add_argument(
        "--preds-max-price",
        type=float,
        default=None,
        help="Optional max price to show (disabled by default).",
    )
    p_train.add_argument(
        "--preds-max-edge-mult",
        type=float,
        default=None,
        help="Optional max multiple of market implied prob to show (disabled by default).",
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
        default=False,
        help="Tune betting filters on out-of-fold predictions (disabled by default).",
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
        "--strategy-min-prob",
        type=float,
        help="Optional min probability filter applied while strategy tuning.",
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
    p_train.add_argument(
        "--calibration-randomize-within-windows",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CALIBRATION_RANDOMIZE_WITHIN_WINDOWS,
        help="When time-aware calibration is available, randomly subsample within each past time window.",
    )
    p_train.add_argument(
        "--calibration-window-sample-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_WINDOW_SAMPLE_FRACTION,
        help=(
            "Fraction of rows to sample inside each calibration time window "
            "(0..1, only used when --calibration-randomize-within-windows)."
        ),
    )
    p_train.add_argument(
        "--calibration-random-state",
        type=int,
        default=DEFAULT_CALIBRATION_RANDOM_STATE,
        help="Random seed for within-window calibration sampling.",
    )
    _add_external_form_auto_ingest_arguments(p_train)
    p_train.set_defaults(func=cmd_train)

    p_bt = sub.add_parser("backtest", help="Backtest value strategy")
    p_bt.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_bt.add_argument(
        "--market-types",
        default="WIN",
        help=(
            "Comma-separated market types/buckets (e.g., WIN,PLACE,EXOTIC,EXACTA). "
            "Defaults to WIN for strategy comparability."
        ),
    )
    p_bt.add_argument("--commission", type=float, default=0.05)
    p_bt.add_argument("--min-ev", type=float, default=None)
    p_bt.add_argument("--min-edge", type=float, default=None)
    p_bt.add_argument("--max-spread", type=float, default=None)
    p_bt.add_argument("--max-price", type=float, default=None)
    p_bt.add_argument("--max-edge-mult", type=float, default=None)
    p_bt.add_argument(
        "--min-prob",
        type=float,
        help="Optional global min probability to bet.",
    )
    p_bt.add_argument(
        "--use-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the latest tuned global kappa threshold as min-prob (disabled by default).",
    )
    p_bt.add_argument(
        "--use-market-type-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use latest per-market-type tuned kappa thresholds as min-prob routing "
            "(disabled by default)."
        ),
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
    _add_external_form_auto_ingest_arguments(p_bt)
    p_bt.set_defaults(func=cmd_backtest)

    p_score = sub.add_parser("score", help="Score today and output CSV")
    p_score.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_score.add_argument("--dry-run", action="store_true")
    p_score.add_argument("--output")
    p_score.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types/buckets (e.g., WIN,PLACE,EXOTIC,EXACTA). ALL disables filtering.",
    )
    p_score.add_argument(
        "--min-prob",
        type=float,
        help="Optional global min probability to include in scoring.",
    )
    p_score.add_argument(
        "--use-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the latest tuned global kappa threshold as min-prob (disabled by default).",
    )
    p_score.add_argument(
        "--use-market-type-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use latest per-market-type tuned kappa thresholds as min-prob routing "
            "(disabled by default)."
        ),
    )
    p_score.add_argument(
        "--min-ev",
        type=float,
        default=None,
        help="Optional min expected value filter for scored rows.",
    )
    p_score.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="Optional min edge filter for scored rows.",
    )
    p_score.add_argument(
        "--max-spread",
        type=float,
        default=None,
        help="Optional max spread filter for scored rows.",
    )
    p_score.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Optional max decision price for scored rows.",
    )
    p_score.add_argument(
        "--max-edge-mult",
        type=float,
        default=None,
        help="Optional max implied-probability multiplier for scored rows.",
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
    _add_external_form_auto_ingest_arguments(p_score)
    _add_execution_domain_arguments(p_score, default_execution_domain="betfair")
    p_score.set_defaults(func=cmd_score)

    p_pub = sub.add_parser(
        "pub",
        aliases=["sheet"],
        help="Create today's manual betting sheet from live markets (no auto execution).",
    )
    p_pub.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_pub.add_argument("--dry-run", action="store_true", help="Use mock data even if credentials exist.")
    p_pub.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types/buckets (e.g., WIN,PLACE,EXOTIC,EXACTA). ALL disables filtering.",
    )
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
    _add_external_form_auto_ingest_arguments(p_pub)
    _add_execution_domain_arguments(p_pub, default_execution_domain="tab")
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
        "--market-types",
        help="Override live market types (comma-separated). Use ALL to disable market-type filtering.",
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
    p_pipe.add_argument(
        "--bad-members-output",
        default="artifacts/ingest_bad_stream_members.txt",
        help="Path to append invalid stream-member basenames detected during ingest (set empty to disable).",
    )
    p_pipe.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    p_pipe.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types/buckets for train/backtest/score. ALL disables filtering.",
    )
    p_pipe.add_argument(
        "--decision-market-types",
        default="WIN",
        help=(
            "Market types for backtest/score decisions inside pipeline. "
            "Train scope still uses --market-types."
        ),
    )
    p_pipe.add_argument("--commission", type=float, default=0.05)
    p_pipe.add_argument("--min-ev", type=float, default=None)
    p_pipe.add_argument("--min-edge", type=float, default=None)
    p_pipe.add_argument("--max-spread", type=float, default=None)
    p_pipe.add_argument("--max-price", type=float, default=None)
    p_pipe.add_argument("--max-edge-mult", type=float, default=None)
    p_pipe.add_argument(
        "--min-prob",
        type=float,
        help="Optional global min probability to bet.",
    )
    p_pipe.add_argument(
        "--use-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the latest tuned global kappa threshold as min-prob (disabled by default).",
    )
    p_pipe.add_argument(
        "--use-market-type-kappa-thresholds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use latest per-market-type tuned kappa thresholds as min-prob routing "
            "(disabled by default)."
        ),
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
    p_pipe.add_argument(
        "--preds-min-ev",
        type=float,
        default=None,
        help="Optional min EV for preview rows (disabled by default).",
    )
    p_pipe.add_argument(
        "--preds-min-edge",
        type=float,
        default=None,
        help="Optional min edge (relative) for preview rows (disabled by default).",
    )
    p_pipe.add_argument(
        "--preds-min-prob",
        type=float,
        help="Optional min probability for preview rows.",
    )
    p_pipe.add_argument(
        "--preds-use-kappa-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the tuned kappa threshold for preview min-prob when --preds-min-prob "
            "is not set (disabled by default)."
        ),
    )
    p_pipe.add_argument(
        "--preds-max-price",
        type=float,
        default=None,
        help="Optional max price to show (disabled by default).",
    )
    p_pipe.add_argument(
        "--preds-max-edge-mult",
        type=float,
        default=None,
        help="Optional max multiple of market implied prob to show (disabled by default).",
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
        default=False,
        help="Tune betting filters on out-of-fold predictions (disabled by default).",
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
        "--strategy-min-prob",
        type=float,
        help="Optional min probability filter applied while strategy tuning.",
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
    p_pipe.add_argument(
        "--calibration-randomize-within-windows",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CALIBRATION_RANDOMIZE_WITHIN_WINDOWS,
        help="When time-aware calibration is available, randomly subsample within each past time window.",
    )
    p_pipe.add_argument(
        "--calibration-window-sample-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_WINDOW_SAMPLE_FRACTION,
        help=(
            "Fraction of rows to sample inside each calibration time window "
            "(0..1, only used when --calibration-randomize-within-windows)."
        ),
    )
    p_pipe.add_argument(
        "--calibration-random-state",
        type=int,
        default=DEFAULT_CALIBRATION_RANDOM_STATE,
        help="Random seed for within-window calibration sampling.",
    )
    p_pipe.add_argument("--output")
    p_pipe.add_argument("--dry-run", action="store_true", help="Applied to scoring step")
    p_pipe.add_argument("--skip-features", action="store_true")
    p_pipe.add_argument("--skip-train", action="store_true")
    p_pipe.add_argument("--skip-backtest", action="store_true")
    p_pipe.add_argument("--skip-score", action="store_true")
    _add_external_form_auto_ingest_arguments(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    p_go = sub.add_parser(
        "go",
        aliases=["auto"],
        help="Run the default optimized flow (historic download -> incremental pipeline).",
    )
    p_go.add_argument(
        "--refresh-historic",
        action="store_true",
        help="Force historic download even when snapshots already exist in DuckDB.",
    )
    p_go.add_argument(
        "--refresh-historic-if-stale-days",
        type=int,
        default=go_historic_stale_days_default(),
        help=(
            "Auto-refresh historic data when the latest snapshot is older than this many days "
            "(set negative to disable stale-data auto-refresh)."
        ),
    )
    _add_external_form_auto_ingest_arguments(p_go)
    p_go.set_defaults(func=cmd_go)

    p_repair = sub.add_parser(
        "repair-manifests",
        help="Prune bad member basenames from historic/ingest manifests so retries can refill corrupted data.",
    )
    p_repair.add_argument(
        "--bad-list",
        default="artifacts/ingest_bad_stream_members.txt",
        help="Newline-delimited bad basenames list (default artifacts/ingest_bad_stream_members.txt).",
    )
    p_repair.add_argument(
        "--historic-manifest",
        default="data/historic_manifest.json",
        help="Path to historic download manifest.",
    )
    p_repair.add_argument(
        "--ingest-manifest",
        default="data/ingest_manifest.json",
        help="Path to ingest manifest.",
    )
    p_repair.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create timestamped backups before modifying manifests.",
    )
    p_repair.set_defaults(func=cmd_repair_manifests)

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
