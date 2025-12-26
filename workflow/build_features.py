import argparse
import pathlib

from shared.features.builder import build_features_from_store
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log


def main() -> None:
    """Load snapshots from DuckDB and materialise leakage-safe training features."""
    parser = argparse.ArgumentParser(description="Build training features from stored snapshots.")
    parser.add_argument(
        "--cutoff-minutes",
        type=int,
        default=10,
        choices=[60, 30, 10, 5, 2, 1],
        help="Only use snapshots before this minute.",
    )
    args = parser.parse_args()

    log(f"Build features: cutoff T-{args.cutoff_minutes}")
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


if __name__ == "__main__":
    main()
