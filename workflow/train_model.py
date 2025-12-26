import argparse

from shared.features.builder import build_features_from_store, split_features_by_race_time
from shared.model.training import train_and_calibrate
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
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print a report summary after training.",
    )
    args = parser.parse_args()

    log(f"Train: cutoff T-{args.cutoff_minutes}")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    if args.split_date:
        features, _ = split_features_by_race_time(features, args.split_date)
        log(f"Train: split-date {args.split_date} -> {len(features)} rows.")
    else:
        log("Train: no split-date set; training on full dataset (in-sample).")
    if features.empty:
        log("Train: no features available after split; skipping.")
        return
    model_path, calibrator_path, metrics = train_and_calibrate(
        features_df=features,
        cutoff_minutes=args.cutoff_minutes,
        store=store,
        split_date=args.split_date,
    )
    log(f"Model saved to {model_path}, calibrator to {calibrator_path}, metrics={metrics}")
    if args.report:
        from workflow.cli import cmd_report

        cmd_report(argparse.Namespace())


if __name__ == "__main__":
    main()
