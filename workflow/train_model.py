import argparse
import json
import pathlib

from shared.backtest.strategy_tuner import tune_strategy
from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.predictions import build_prediction_preview
from shared.model.training import load_model_and_calibrator, predict_probabilities, train_and_calibrate
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.bet_explain import preview_legend_lines
from shared.utils.progress import log


def main() -> None:
    """Train baseline model + calibrator using stored snapshots/features."""
    parser = argparse.ArgumentParser(description="Train and calibrate a baseline model.")
    parser.add_argument(
        "--cutoff-minutes",
        type=int,
        default=10,
        choices=[60, 30, 10, 5, 2, 1],
        help="Snapshot cutoff in minutes.",
    )
    parser.add_argument(
        "--split-date",
        help="ISO date/time; training uses races strictly before this (leakage-safe split).",
    )
    parser.add_argument(
        "--show-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a preview of top predictions after training.",
    )
    parser.add_argument(
        "--preds-limit",
        type=int,
        default=20,
        help="Rows to show in the prediction preview.",
    )
    parser.add_argument("--preds-min-ev", type=float, default=0.02, help="Min EV for preview rows.")
    parser.add_argument(
        "--preds-min-edge", type=float, default=0.1, help="Min edge (relative) for preview rows."
    )
    parser.add_argument("--preds-max-price", type=float, default=200.0, help="Max price to show.")
    parser.add_argument(
        "--preds-max-edge-mult",
        type=float,
        default=5.0,
        help="Max multiple of market implied prob to show.",
    )
    parser.add_argument(
        "--preds-per-market",
        type=int,
        default=1,
        help="Max preview rows per market.",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a report summary after training.",
    )
    parser.add_argument(
        "--tune-strategy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune betting filters on out-of-fold predictions (use --no-tune-strategy to disable).",
    )
    parser.add_argument(
        "--strategy-grid",
        choices=["small", "medium", "large"],
        default="small",
        help="Grid size for strategy tuning.",
    )
    parser.add_argument(
        "--strategy-objective",
        choices=["expected_roi", "roi", "expected_profit", "profit"],
        default="expected_roi",
        help="Metric to maximize during strategy tuning.",
    )
    parser.add_argument(
        "--strategy-min-hit-rate",
        type=float,
        default=0.05,
        help="Minimum hit rate required for tuned configs.",
    )
    parser.add_argument(
        "--strategy-min-bets",
        type=int,
        default=200,
        help="Minimum number of bets required for tuned configs.",
    )
    parser.add_argument(
        "--strategy-top-n",
        type=int,
        default=10,
        help="Top configs to display after tuning.",
    )
    parser.add_argument(
        "--strategy-log-every",
        type=int,
        default=20,
        help="Progress logging frequency during strategy tuning.",
    )
    parser.add_argument(
        "--strategy-output",
        help="Optional CSV path to write strategy tuning results.",
    )
    args = parser.parse_args()

    log(f"Train: cutoff T-{args.cutoff_minutes}")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    train_features = features
    test_features = None
    if args.split_date:
        train_features, test_features = split_features_by_race_time(features, args.split_date)
        log(f"Train: split-date {args.split_date} -> {len(train_features)} train rows.")
    else:
        log("Train: no split-date set; training on full dataset (in-sample).")
    if train_features.empty:
        log("Train: no features available after split; skipping.")
        return
    model_path, calibrator_path, metrics, oof_predictions = train_and_calibrate(
        features_df=train_features,
        cutoff_minutes=args.cutoff_minutes,
        store=store,
        split_date=args.split_date,
    )
    log(f"Model saved to {model_path}, calibrator to {calibrator_path}, metrics={metrics}")
    if args.show_preds:
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
    if args.tune_strategy:
        if oof_predictions is None or oof_predictions.empty:
            log("Strategy tuning: no out-of-fold predictions available; skipping.")
        else:
            tuning_features = train_features.loc[oof_predictions.index]
            results, tradeoff = tune_strategy(
                feature_df=tuning_features,
                probs=oof_predictions,
                cutoff_minutes=args.cutoff_minutes,
                commission=0.05,
                grid_profile=args.strategy_grid,
                objective=args.strategy_objective,
                min_hit_rate=args.strategy_min_hit_rate,
                min_bets=args.strategy_min_bets,
                stake=1.0,
                log_every=args.strategy_log_every,
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
    if args.report:
        from workflow.cli import cmd_report

        cmd_report(argparse.Namespace())


if __name__ == "__main__":
    main()
