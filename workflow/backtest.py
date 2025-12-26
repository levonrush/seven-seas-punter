import argparse

from shared.backtest.engine import run_backtest
from shared.features.builder import build_features_from_store
from shared.model.training import load_model_and_calibrator, predict_probabilities
from shared.storage.duckdb_store import DuckDBStore
from shared.utils.progress import log


def main() -> None:
    """Backtest a value-betting strategy at the chosen cutoff."""
    parser = argparse.ArgumentParser(description="Run value strategy backtest.")
    parser.add_argument("--cutoff-minutes", type=int, default=10, choices=[60, 30, 10, 5, 2, 1])
    parser.add_argument("--commission", type=float, default=0.05)
    parser.add_argument("--min-ev", type=float, default=0.02)
    parser.add_argument("--max-spread", type=float, default=1.0)
    parser.add_argument("--stake", type=float, default=1.0)
    parser.add_argument("--model-path", type=str, default=None, help="Optional explicit model path.")
    parser.add_argument("--calibrator-path", type=str, default=None, help="Optional calibrator path.")
    args = parser.parse_args()

    log(f"Backtest: cutoff T-{args.cutoff_minutes}")
    store = DuckDBStore()
    features = build_features_from_store(store, cutoff_minutes=args.cutoff_minutes)
    model_path = args.model_path or f"artifacts/model_cutoff_{args.cutoff_minutes}.joblib"
    calibrator_path = args.calibrator_path or f"artifacts/calibrator_cutoff_{args.cutoff_minutes}.joblib"

    try:
        model, calibrator = load_model_and_calibrator(model_path, calibrator_path)
        probs = predict_probabilities(model, calibrator, features)
    except FileNotFoundError:
        # Fallback: implied probabilities
        price_col = f"back_price_t{args.cutoff_minutes}"
        implied = 1 / features[price_col]
        probs = implied.fillna(implied.mean())

    bet_df, metrics = run_backtest(
        feature_df=features,
        probs=probs,
        cutoff_minutes=args.cutoff_minutes,
        commission=args.commission,
        min_ev=args.min_ev,
        max_spread=args.max_spread,
        stake=args.stake,
    )
    log(f"Backtest metrics: {metrics}")

    # Persist bets for auditing
    store.record_bets(
        bet_df.to_dict(orient="records") if hasattr(bet_df, "to_dict") else [],
    )


if __name__ == "__main__":
    main()
