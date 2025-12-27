import argparse

from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.predictions import build_prediction_preview
from shared.model.training import load_model_and_calibrator, predict_probabilities, train_and_calibrate
from shared.storage.duckdb_store import DuckDBStore
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
        else:
            log("Prediction preview: no out-of-fold predictions available.")
    if args.report:
        from workflow.cli import cmd_report

        cmd_report(argparse.Namespace())


if __name__ == "__main__":
    main()
