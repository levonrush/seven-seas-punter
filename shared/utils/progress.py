import time

_START = time.monotonic()


def log(message: str) -> None:
    """Print a progress message with elapsed time since process start."""
    elapsed = time.monotonic() - _START
    print(f"[+{elapsed:.1f}s] {message}", flush=True)
