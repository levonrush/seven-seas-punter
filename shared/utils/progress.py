import time

_START = time.monotonic()


def log(message: str) -> None:
    """Print a progress message with elapsed time since process start."""
    elapsed = time.monotonic() - _START
    if elapsed < 60:
        display = f"{elapsed:.1f}s"
    elif elapsed < 3600:
        display = f"{elapsed / 60:.1f}m"
    else:
        display = f"{elapsed / 3600:.2f}h"
    print(f"[+{display}] {message}", flush=True)
