import argparse

from shared.live.betfair_live import run_live_loop
from shared.utils.progress import log


def main() -> None:
    """Run live Betfair polling/inference/execution from a YAML config file."""
    parser = argparse.ArgumentParser(description="Live Betfair model execution loop.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to live YAML config.",
    )
    args = parser.parse_args()
    log(f"Live: starting with config {args.config}")
    run_live_loop(args.config)


if __name__ == "__main__":
    main()
