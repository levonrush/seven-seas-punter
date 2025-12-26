import argparse

from shared.features.builder import build_features_from_store
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
    args = parser.parse_args()

    log(f"Train: cutoff T-{args.cutoff_minutes}")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    model_path, calibrator_path, metrics = train_and_calibrate(
        features_df=features, cutoff_minutes=args.cutoff_minutes, store=store
    )
    log(f"Model saved to {model_path}, calibrator to {calibrator_path}, metrics={metrics}")


if __name__ == "__main__":
    main()
