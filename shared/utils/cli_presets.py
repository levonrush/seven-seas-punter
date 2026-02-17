from __future__ import annotations

import os
from typing import List


def recommended_historic_workers(cpu_count: int | None = None) -> int:
    """Return a fast but stable worker count for historic downloads.

    Why: the historic API throttles bursty request patterns; this cap keeps concurrency high
    enough for throughput while remaining manageable with the built-in request limiter.
    """
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 4)
    return max(2, min(8, int(cores)))


def go_historic_args() -> List[str]:
    """Return CLI args for the no-tune historic download step used by `punter go`.

    Why: encode a single safe preset so users can run a full refresh without memorizing flags.
    """
    return [
        "--auto",
        "--market-types",
        "ALL",
        "--workers",
        str(recommended_historic_workers()),
        "--clean-temp",
    ]


def go_pipeline_args() -> List[str]:
    """Return CLI args for the no-tune pipeline step used by `punter go`.

    Why: keep a single default profile for end-to-end operation while preserving advanced
    command-specific tuning flags elsewhere in the CLI.
    """
    return [
        "--ingest-new",
        "--cutoff-minutes",
        "10",
    ]

