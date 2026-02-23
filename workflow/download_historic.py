from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

from shared.betfair.historic import HistoricDataClient, build_filter, dump_json
from shared.utils.cli_presets import recommended_historic_workers
from shared.utils.progress import log


def _parse_date(date_str: str) -> dt.date:
    """Parse YYYY-MM-DD strings for historic API filters."""
    return dt.date.fromisoformat(date_str)


def _split_csv(value: str | None) -> List[str]:
    """Split comma-separated strings into a list of trimmed values."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_market_types(values: List[str]) -> List[str]:
    """Normalize market-type filters and treat ALL/* as no filter."""
    if not values:
        return []
    normalized = []
    seen = set()
    for value in values:
        token = value.strip().upper()
        if not token:
            continue
        if token in {"ALL", "*"}:
            return []
        if token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _print_market_types(options: dict) -> None:
    """Print available market types in a compact format for quick filter selection."""
    items = options.get("marketTypesCollection") or []
    if not items:
        print("No market types returned for this filter.")
        return
    rows = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            count = item.get("count")
        else:
            name = str(item).strip()
            count = None
        if not name:
            continue
        rows.append((name, count))
    rows = sorted(rows, key=lambda row: (-(row[1] or 0), row[0]))
    print("Available market types:")
    for name, count in rows:
        suffix = f" ({count})" if isinstance(count, int) else ""
        print(f"- {name}{suffix}")


def _resolve_workers(value: int | None) -> tuple[int, bool]:
    """Choose a fast but stable default worker count for historic downloads."""
    if value and value > 0:
        return value, False
    return recommended_historic_workers(), True


def _default_output(from_date: dt.date, to_date: dt.date) -> Path:
    """Build a stable tar output path for the date window."""
    name = f"data/historic_{from_date.isoformat()}_{to_date.isoformat()}.tar"
    return Path(name)


def _month_bounds(date_value: dt.date) -> tuple[dt.date, dt.date]:
    """Return first/last day for the month containing the provided date."""
    first = date_value.replace(day=1)
    last_day = calendar.monthrange(date_value.year, date_value.month)[1]
    return first, date_value.replace(day=last_day)


def _build_payload(
    args: argparse.Namespace,
    from_date: dt.date,
    to_date: dt.date,
    plan_name: str,
    market_types: list[str],
    countries: list[str],
    file_types: list[str],
) -> dict:
    """Build the historic API payload once so list/size/shard calls stay in sync."""
    return build_filter(
        sport=args.sport,
        plan=plan_name,
        from_day=from_date.day,
        from_month=from_date.month,
        from_year=from_date.year,
        to_day=to_date.day,
        to_month=to_date.month,
        to_year=to_date.year,
        market_types=market_types,
        countries=countries,
        file_types=file_types,
        event_id=args.event_id,
        event_name=args.event_name,
    ).to_payload()


def _iter_numeric_fields(value: object, path: str = "") -> Iterable[Tuple[str, float]]:
    """Yield numeric leaf values with their dotted paths to simplify flexible response parsing."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            yield from _iter_numeric_fields(child, child_path)
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{path}[{idx}]"
            yield from _iter_numeric_fields(child, child_path)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)


def _extract_basket_size_mb(size_payload: dict) -> float | None:
    """Return basket size in MB from `GetAdvBasketDataSize`, handling observed response variants."""
    candidates: list[float] = []
    for path, value in _iter_numeric_fields(size_payload):
        key = path.lower()
        if "size" not in key:
            continue
        if "filesize" in key:
            continue
        if "byte" in key:
            candidates.append(value / (1024 * 1024))
            continue
        if "kb" in key:
            candidates.append(value / 1024)
            continue
        if "gb" in key:
            candidates.append(value * 1024)
            continue
        if "mb" in key:
            candidates.append(value)
            continue
        # Historic API docs/examples usually communicate basket size as MB.
        # If we get a very large unlabeled number, treat it as bytes.
        if value > 100_000:
            candidates.append(value / (1024 * 1024))
        else:
            candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _extract_basket_file_count(size_payload: dict) -> int | None:
    """Return file count from `GetAdvBasketDataSize` payload for progress logging."""
    for path, value in _iter_numeric_fields(size_payload):
        key = path.lower()
        if "count" in key and "size" not in key:
            return int(value)
        if "files" in key and "size" not in key:
            return int(value)
    return None


def _split_window(from_date: dt.date, to_date: dt.date) -> tuple[tuple[dt.date, dt.date], tuple[dt.date, dt.date]]:
    """Split a date window in half so oversized baskets can be retried with smaller API requests."""
    span_days = (to_date - from_date).days
    if span_days <= 0:
        return (from_date, to_date), (to_date, to_date)
    midpoint = from_date + dt.timedelta(days=span_days // 2)
    left = (from_date, midpoint)
    right = (midpoint + dt.timedelta(days=1), to_date)
    return left, right


def _plan_sharded_windows(
    client: HistoricDataClient,
    args: argparse.Namespace,
    plan_name: str,
    from_date: dt.date,
    to_date: dt.date,
    market_types: list[str],
    countries: list[str],
    file_types: list[str],
) -> list[tuple[dt.date, dt.date]]:
    """Recursively shard large windows to keep each list/download basket at a manageable size."""
    target_mb = float(args.target_basket_mb or 0)
    if target_mb <= 0:
        return [(from_date, to_date)]
    max_depth = max(0, int(args.max_shard_depth or 0))

    def recurse(window_from: dt.date, window_to: dt.date, depth: int) -> list[tuple[dt.date, dt.date]]:
        if window_from >= window_to:
            return [(window_from, window_to)]
        payload = _build_payload(args, window_from, window_to, plan_name, market_types, countries, file_types)
        try:
            size_payload = client.get_basket_size(payload)
        except requests.RequestException as exc:
            log(
                "Historic download: "
                f"basket-size probe failed for {window_from.isoformat()}->{window_to.isoformat()} "
                f"({plan_name}); using unsplit window ({exc})."
            )
            return [(window_from, window_to)]
        size_mb = _extract_basket_size_mb(size_payload)
        if size_mb is None:
            log(
                "Historic download: "
                f"basket-size probe returned unknown shape for {window_from.isoformat()}->{window_to.isoformat()} "
                f"({plan_name}); using unsplit window."
            )
            return [(window_from, window_to)]
        if size_mb <= target_mb:
            return [(window_from, window_to)]
        if depth >= max_depth:
            log(
                "Historic download: "
                f"window {window_from.isoformat()}->{window_to.isoformat()} ({plan_name}) "
                f"is {size_mb:.1f}MB but max shard depth reached ({max_depth}); keeping as-is."
            )
            return [(window_from, window_to)]
        left, right = _split_window(window_from, window_to)
        if left == right:
            return [(window_from, window_to)]
        file_count = _extract_basket_file_count(size_payload)
        file_note = f", files={file_count}" if file_count is not None else ""
        log(
            "Historic download: "
            f"sharding {window_from.isoformat()}->{window_to.isoformat()} ({plan_name}, {size_mb:.1f}MB{file_note}) "
            f"to stay under {target_mb:.0f}MB."
        )
        return recurse(*left, depth + 1) + recurse(*right, depth + 1)

    return recurse(from_date, to_date, 0)


def _parse_package_month(value: str) -> dt.date:
    """Parse Betfair GetMyData forDate values into a date."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _default_manifest_payload() -> dict:
    """Provide a stable empty manifest payload so recoverable parse failures do not crash download runs."""
    return {"files": {}, "version": 1}


def _backup_invalid_manifest(path: Path) -> Path | None:
    """Copy a corrupted manifest to a timestamped sidecar path so operators can inspect bad state later."""
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bad_{timestamp}")
    try:
        shutil.copy2(path, backup_path)
    except OSError as exc:
        log(
            "Historic download: "
            f"failed to back up invalid manifest {path} ({exc}); continuing with a fresh manifest."
        )
        return None
    return backup_path


def _load_manifest(path: Path) -> dict:
    """Load manifest state while tolerating empty/corrupt files so resumed downloads are resilient."""
    data = _default_manifest_payload()
    if path.exists():
        raw_payload = path.read_text(encoding="utf-8")
        if raw_payload.strip():
            try:
                loaded = json.loads(raw_payload)
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    log(
                        "Historic download: "
                        f"manifest {path} contains non-object JSON; using a fresh manifest."
                    )
            except json.JSONDecodeError as exc:
                backup_path = _backup_invalid_manifest(path)
                if backup_path is not None:
                    log(
                        "Historic download: "
                        f"manifest {path} is invalid JSON ({exc}); backed up to {backup_path} and reset."
                    )
                else:
                    log(
                        "Historic download: "
                        f"manifest {path} is invalid JSON ({exc}); using a fresh manifest."
                    )
        else:
            log(f"Historic download: manifest {path} is empty; using a fresh manifest.")
    files = data.get("files")
    if not isinstance(files, dict):
        log(
            "Historic download: "
            f"manifest {path} has non-object 'files'; using an empty files map."
        )
        files = {}
    data["files"] = files
    try:
        data["version"] = int(data.get("version", 1) or 1)
    except (TypeError, ValueError):
        data["version"] = 1
    basenames = set()
    for entry in files.values():
        if not isinstance(entry, dict):
            continue
        basename = entry.get("basename")
        if basename:
            basenames.add(str(basename))
    data["_basenames"] = basenames
    return data


def _save_manifest(path: Path, manifest: dict) -> None:
    """Persist the historic download manifest to disk."""
    manifest = {k: v for k, v in manifest.items() if not k.startswith("_")}
    manifest["updated_at"] = dt.datetime.utcnow().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _list_cache_path(cache_dir: Path, from_date: dt.date, to_date: dt.date, plan: str, payload: dict) -> Path:
    """Build a deterministic cache path for a list_files payload."""
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    plan_slug = plan.lower().replace(" ", "-")
    name = f"list_{plan_slug}_{from_date.isoformat()}_{to_date.isoformat()}_{digest}.json"
    return cache_dir / name


def _load_cached_list(path: Path) -> list[str] | None:
    """Load cached list_files output when present and valid."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return [str(item) for item in data]
    return None


def _save_cached_list(path: Path, files: list[str]) -> None:
    """Persist list_files output for reuse in later runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(files), indent=2), encoding="utf-8")


def _mark_downloaded(manifest: dict, file_path: str, dest: Path) -> None:
    """Record a downloaded file in the manifest."""
    manifest["files"][file_path] = {
        "basename": dest.name,
        "downloaded_at": dt.datetime.utcnow().isoformat(),
    }
    manifest["_basenames"].add(dest.name)


def _is_downloaded(manifest: dict, file_path: str) -> bool:
    """Return True when the file path or basename is already recorded."""
    if file_path in manifest["files"]:
        return True
    basename = Path(file_path).name
    return basename in manifest.get("_basenames", set())


def _seed_manifest_from_tar(manifest: dict, tar_path: Path) -> None:
    """Preload manifest basenames from an existing tar so repeats can be skipped."""
    if manifest.get("files"):
        return
    if not tar_path.exists():
        return
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile():
                manifest["_basenames"].add(Path(member.name).name)


def _is_retryable_download_error(exc: Exception) -> bool:
    """Classify transient download failures so adaptive retries focus on likely recoverable issues."""
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status in {408, 425, 429} or (status is not None and status >= 500)
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    text = str(exc).lower()
    return "timed out" in text or "temporarily unavailable" in text


def _download_many(
    client: HistoricDataClient,
    file_paths: Iterable[str],
    output_dir: Path,
    workers: int,
    progress_every: int,
    manifest: dict,
    force: bool,
    retries: int,
    retry_wait: float,
    adaptive_workers: bool,
    max_rounds: int,
    adaptive_cooldown: float,
) -> List[Path]:
    """Download files with optional adaptive worker backoff for transient throttling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for file_path in file_paths:
        if not force and _is_downloaded(manifest, file_path):
            continue
        files.append(file_path)
    saved: List[Path] = []
    saved_names: set[str] = set()
    total = len(files)
    if total == 0:
        return saved
    pending = list(files)
    current_workers = max(1, int(workers))
    total_rounds = max(1, int(max_rounds))
    done = 0
    for round_idx in range(1, total_rounds + 1):
        if not pending:
            break
        if round_idx == 1:
            log(
                "Historic download: "
                f"starting round {round_idx}/{total_rounds} with {current_workers} workers ({len(pending)} files)."
            )
        else:
            log(
                "Historic download: "
                f"retry round {round_idx}/{total_rounds} with {current_workers} workers ({len(pending)} files)."
            )
        round_failures: list[str] = []
        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            futures = {}
            for idx, file_path in enumerate(pending, start=1):
                dest = output_dir / Path(file_path).name
                if dest.exists() and not force:
                    _mark_downloaded(manifest, file_path, dest)
                    if dest.name not in saved_names:
                        saved.append(dest)
                        saved_names.add(dest.name)
                        done += 1
                    continue
                futures[executor.submit(client.download_file, file_path, str(dest), retries, retry_wait)] = (
                    dest,
                    file_path,
                )
                if progress_every and idx % progress_every == 0:
                    log(f"Historic download: queued {idx}/{len(pending)} files in round {round_idx}.")
            for future in as_completed(futures):
                dest, file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    retryable = _is_retryable_download_error(exc)
                    if retryable and round_idx < total_rounds:
                        round_failures.append(file_path)
                        log(f"Historic download: transient failure for {file_path}; queued for retry.")
                    else:
                        log(f"Historic download: failed {file_path} ({exc}).")
                    if dest.exists():
                        dest.unlink(missing_ok=True)
                    continue
                if dest.name not in saved_names:
                    saved.append(dest)
                    saved_names.add(dest.name)
                    done += 1
                if progress_every and done % progress_every == 0:
                    log(f"Historic download: completed {done}/{total} files.")
        if not round_failures:
            break
        pending = round_failures
        if adaptive_workers and current_workers > 1:
            next_workers = max(1, current_workers // 2)
            if next_workers < current_workers:
                log(
                    "Historic download: "
                    f"reducing workers {current_workers}->{next_workers} after transient failures."
                )
            current_workers = next_workers
        if adaptive_cooldown > 0:
            time.sleep(adaptive_cooldown)
    if pending and len(saved) < total:
        log(f"Historic download: finished with {len(saved)}/{total} files downloaded.")
    return saved


def _tar_files(output_tar: Path, files: Iterable[Path]) -> None:
    """Bundle downloaded files into a tar archive."""
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_tar.exists() else "w"
    with tarfile.open(output_tar, mode) as tar:
        for path in files:
            tar.add(path, arcname=path.name)


def run_download_historic(args: argparse.Namespace) -> None:
    """Execute a historic download using pre-parsed CLI arguments."""
    if args.list_output and not args.list_only:
        args.list_only = True
    if not args.auto and not args.from_date and not args.to_date:
        log("Historic download: no date range supplied; defaulting to --auto.")
        args.auto = True
    if not (
        args.download
        or args.list_only
        or args.size_only
        or args.show_options
        or args.show_market_types
    ):
        args.download = True
    workers, auto_workers = _resolve_workers(args.workers)
    args.workers = workers
    if auto_workers:
        label = "worker" if workers == 1 else "workers"
        log(f"Historic download: using {workers} {label} (default).")
    market_types = _normalize_market_types(_split_csv(args.market_types))
    countries = _split_csv(args.countries)
    file_types = _split_csv(args.file_types)
    if market_types:
        log(f"Historic download: market type filter -> {','.join(market_types)}.")
    else:
        log("Historic download: market type filter disabled (ALL market types).")

    client = HistoricDataClient(
        max_requests=args.max_requests,
        request_window_seconds=args.request_window_seconds,
    )
    log(
        "Historic download: "
        f"request throttle set to {client.max_requests} requests per {client.request_window_seconds:.1f}s."
    )
    if args.auto_shard:
        log(
            "Historic download: "
            f"auto-shard enabled (target basket <= {float(args.target_basket_mb):.0f}MB, "
            f"max depth {int(args.max_shard_depth)})."
        )
    else:
        log("Historic download: auto-shard disabled.")

    manifest_path = Path(args.manifest_path or "data/historic_manifest.json")
    manifest = _load_manifest(manifest_path)
    list_cache_dir = Path(args.list_cache_dir) if getattr(args, "list_cache_dir", None) else None
    if list_cache_dir:
        list_cache_dir.mkdir(parents=True, exist_ok=True)

    def handle_window(from_date: dt.date, to_date: dt.date, plan: str) -> List[str]:
        plan_name = plan or args.plan or "Basic Plan"
        payload = _build_payload(args, from_date, to_date, plan_name, market_types, countries, file_types)

        if args.show_options or args.show_market_types:
            options = client.get_collection_options(payload)
            if args.show_market_types:
                _print_market_types(options)
            if args.show_options:
                print(dump_json(options))
            return []
        if args.size_only:
            size = client.get_basket_size(payload)
            print(dump_json(size))
            return []
        files = []
        cache_path = None
        if list_cache_dir:
            cache_path = _list_cache_path(list_cache_dir, from_date, to_date, plan_name, payload)
            if not args.refresh_list_cache:
                cached = _load_cached_list(cache_path)
                if cached is not None:
                    files = cached
                    log(f"Historic download: using cached file list {cache_path.name}.")
        last_error = None
        if not files:
            for attempt in range(1, args.retries + 1):
                try:
                    files = client.list_files(payload)
                    last_error = None
                    break
                except requests.HTTPError as exc:
                    last_error = exc
                    status = exc.response.status_code if exc.response else "unknown"
                    body = (exc.response.text or "").strip().replace("\n", " ") if exc.response else ""
                    snippet = body[:200] if body else str(exc)
                    log(
                        "Historic download: "
                        f"{from_date.isoformat()} to {to_date.isoformat()} ({plan_name}) "
                        f"list_files failed (HTTP {status}). {snippet}"
                    )
                    if attempt < args.retries and exc.response and exc.response.status_code >= 500:
                        sleep_for = args.retry_wait * attempt
                        log(f"Historic download: retrying in {sleep_for}s (attempt {attempt}/{args.retries}).")
                        time.sleep(sleep_for)
                        continue
                    if args.auto:
                        return []
                    raise
                except requests.RequestException as exc:
                    last_error = exc
                    log(
                        "Historic download: "
                        f"{from_date.isoformat()} to {to_date.isoformat()} ({plan_name}) "
                        f"list_files failed ({exc})."
                    )
                    if attempt < args.retries:
                        sleep_for = args.retry_wait * attempt
                        log(f"Historic download: retrying in {sleep_for}s (attempt {attempt}/{args.retries}).")
                        time.sleep(sleep_for)
                        continue
                    if args.auto:
                        return []
                    raise
        if last_error and not files:
            return []
        if files and list_cache_dir and cache_path:
            _save_cached_list(cache_path, files)
        if args.max_files:
            files = files[: args.max_files]
        if args.list_only or not args.download:
            if args.list_output:
                Path(args.list_output).write_text(dump_json(files), encoding="utf-8")
                log(f"Wrote file list to {args.list_output}.")
            else:
                print(dump_json(files))
            return []
        return files

    def iter_windows(from_date: dt.date, to_date: dt.date, plan_name: str) -> list[tuple[dt.date, dt.date]]:
        """Return one or more windows, optionally sharded by basket size to avoid oversized API calls."""
        if args.show_options or args.show_market_types or args.size_only or args.list_only or not args.download:
            return [(from_date, to_date)]
        if not args.auto_shard:
            return [(from_date, to_date)]
        return _plan_sharded_windows(
            client=client,
            args=args,
            plan_name=plan_name,
            from_date=from_date,
            to_date=to_date,
            market_types=market_types,
            countries=countries,
            file_types=file_types,
        )

    if args.auto:
        packages = client.get_my_data(retries=args.retries, retry_wait=args.retry_wait)
        sport_filter = args.sport.lower()
        plan_filter = None
        if args.plan and args.plan.lower() not in {"all", "*"}:
            plan_filter = args.plan.lower()
        from_date = _parse_date(args.from_date) if args.from_date else None
        to_date = _parse_date(args.to_date) if args.to_date else None
        package_months = []
        for pkg in packages:
            if pkg.get("sport", "").lower() != sport_filter:
                continue
            if plan_filter and pkg.get("plan", "").lower() != plan_filter:
                continue
            for_date = _parse_package_month(pkg["forDate"])
            if from_date and for_date < from_date:
                continue
            if to_date and for_date > to_date:
                continue
            package_months.append((for_date, pkg.get("plan")))
        if not package_months:
            log("Historic download: no matching packages found.")
            return
        if args.show_market_types:
            latest_for_date, latest_plan = sorted(package_months)[-1]
            from_date, to_date = _month_bounds(latest_for_date)
            plan_name = latest_plan or args.plan or "Basic Plan"
            log(
                "Historic download: showing market types for "
                f"{from_date.isoformat()} to {to_date.isoformat()} ({plan_name})."
            )
            handle_window(from_date, to_date, plan_name)
            return
        output_tar = Path(args.output) if args.output else Path("data/data.tar")
        _seed_manifest_from_tar(manifest, output_tar)
        output_dir = Path(args.output_dir) if args.output_dir else output_tar.with_suffix("")
        if args.clean_temp and output_dir.exists():
            log(f"Historic download: cleaning temp dir {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        for for_date, plan in sorted(package_months):
            from_date, to_date = _month_bounds(for_date)
            plan_name = plan or args.plan or "Basic Plan"
            for window_from, window_to in iter_windows(from_date, to_date, plan_name):
                files = handle_window(window_from, window_to, plan_name)
                if not files:
                    continue
                log(
                    "Historic download: "
                    f"{window_from.isoformat()} to {window_to.isoformat()} ({plan_name})"
                )
                saved = _download_many(
                    client,
                    files,
                    output_dir=output_dir,
                    workers=args.workers,
                    progress_every=args.progress_every,
                    manifest=manifest,
                    force=args.force,
                    retries=args.retries,
                    retry_wait=args.retry_wait,
                    adaptive_workers=args.adaptive_workers,
                    max_rounds=args.max_download_rounds,
                    adaptive_cooldown=args.adaptive_cooldown,
                )
                file_lookup = {Path(file_path).name: file_path for file_path in files}
                for path in saved:
                    file_path = file_lookup.get(path.name)
                    if file_path:
                        _mark_downloaded(manifest, file_path, path)
                if saved:
                    log("Historic download: appending to tar archive.")
                    _tar_files(output_tar, saved)
                    if not args.keep_files:
                        for file_path in saved:
                            file_path.unlink(missing_ok=True)
                _save_manifest(manifest_path, manifest)
            _save_manifest(manifest_path, manifest)
            time.sleep(0.2)
        if not args.keep_files:
            shutil.rmtree(output_dir, ignore_errors=True)
            log("Historic download: cleaned temporary files.")
        log(f"Historic download: complete -> {output_tar}")
        return

    if not args.from_date or not args.to_date:
        raise ValueError("--from-date and --to-date are required unless --auto is set.")
    plan_name = args.plan or "Basic Plan"
    if plan_name.lower() in {"all", "*"}:
        raise ValueError("--plan must be a concrete plan when not using --auto.")

    from_date = _parse_date(args.from_date)
    to_date = _parse_date(args.to_date)
    if to_date < from_date:
        raise ValueError("to-date must be on or after from-date.")

    output_tar = Path(args.output) if args.output else _default_output(from_date, to_date)
    _seed_manifest_from_tar(manifest, output_tar)
    output_dir = Path(args.output_dir) if args.output_dir else output_tar.with_suffix("")
    if args.clean_temp and output_dir.exists():
        log(f"Historic download: cleaning temp dir {output_dir}.")
        shutil.rmtree(output_dir, ignore_errors=True)
    windows = iter_windows(from_date, to_date, plan_name)
    for window_from, window_to in windows:
        files = handle_window(window_from, window_to, plan_name)
        if not files:
            continue
        log(f"Historic download: {len(files)} files -> {output_tar}")
        saved = _download_many(
            client,
            files,
            output_dir=output_dir,
            workers=args.workers,
            progress_every=args.progress_every,
            manifest=manifest,
            force=args.force,
            retries=args.retries,
            retry_wait=args.retry_wait,
            adaptive_workers=args.adaptive_workers,
            max_rounds=args.max_download_rounds,
            adaptive_cooldown=args.adaptive_cooldown,
        )
        file_lookup = {Path(file_path).name: file_path for file_path in files}
        for path in saved:
            file_path = file_lookup.get(path.name)
            if file_path:
                _mark_downloaded(manifest, file_path, path)
        if saved:
            log("Historic download: building tar archive.")
            _tar_files(output_tar, saved)
    _save_manifest(manifest_path, manifest)
    if not args.keep_files:
        shutil.rmtree(output_dir, ignore_errors=True)
        log("Historic download: cleaned temporary files.")
    log(f"Historic download: complete -> {output_tar}")


def main() -> None:
    """Download Betfair historic data using the official historic API."""
    parser = argparse.ArgumentParser(description="Download Betfair historic data via the historic API.")
    parser.add_argument("--auto", action="store_true", help="Download all available purchased months.")
    parser.add_argument("--from-date", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--to-date", help="End date (YYYY-MM-DD).")
    parser.add_argument("--sport", default="Horse Racing", help="Sport name (e.g., Horse Racing).")
    parser.add_argument("--plan", default="Basic Plan", help="Plan name (e.g., Basic Plan).")
    parser.add_argument(
        "--market-types",
        default="ALL",
        help="Comma-separated market types. Use ALL (default) for no market-type filter.",
    )
    parser.add_argument("--countries", default="AU", help="Comma-separated country codes.")
    parser.add_argument("--file-types", default="M", help="Comma-separated file types (M/E).")
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--event-name")
    parser.add_argument(
        "--show-market-types",
        action="store_true",
        help="Print available market types (with counts) for the selected window and exit.",
    )
    parser.add_argument("--show-options", action="store_true", help="Show available filters.")
    parser.add_argument("--size-only", action="store_true", help="Only show file count and size.")
    parser.add_argument("--list-only", action="store_true", help="Only list available files.")
    parser.add_argument("--list-output", help="Optional path to write the file list JSON.")
    parser.add_argument(
        "--list-cache-dir",
        default="data/historic_lists",
        help="Cache list_files responses (set empty to disable).",
    )
    parser.add_argument(
        "--refresh-list-cache",
        action="store_true",
        help="Ignore cached file lists and refresh from the API.",
    )
    parser.add_argument("--download", action="store_true", help="Download the files.")
    parser.add_argument("--output", help="Output tar path (defaults to data/historic_*.tar).")
    parser.add_argument("--output-dir", help="Directory to store downloads before tarring.")
    parser.add_argument("--keep-files", action="store_true", help="Keep downloaded files on disk.")
    parser.add_argument("--clean-temp", action="store_true", help="Remove temp download dir before starting.")
    parser.add_argument("--max-files", type=int, help="Limit number of files (useful for testing).")
    parser.add_argument("--workers", type=int, default=None, help="Parallel download workers (default: auto).")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Override request cap per window (defaults to BETFAIR_HISTORIC_MAX_REQUESTS or 90).",
    )
    parser.add_argument(
        "--request-window-seconds",
        type=float,
        default=None,
        help="Override rate-limit window seconds (defaults to BETFAIR_HISTORIC_REQUEST_WINDOW or 10).",
    )
    parser.add_argument("--progress-every", type=int, default=50, help="Progress logging frequency.")
    parser.add_argument("--manifest-path", help="Path for download manifest JSON.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if seen before.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for historic API failures.")
    parser.add_argument("--retry-wait", type=float, default=3.0, help="Seconds to wait between retries.")
    parser.add_argument(
        "--target-basket-mb",
        type=float,
        default=1000.0,
        help="Shard date windows when basket size exceeds this MB threshold (set <=0 to disable).",
    )
    parser.add_argument(
        "--max-shard-depth",
        type=int,
        default=5,
        help="Maximum recursive split depth for oversized windows.",
    )
    parser.add_argument(
        "--no-auto-shard",
        dest="auto_shard",
        action="store_false",
        help="Disable automatic basket-size based date sharding.",
    )
    parser.add_argument(
        "--no-adaptive-workers",
        dest="adaptive_workers",
        action="store_false",
        help="Disable worker backoff when transient failures are detected.",
    )
    parser.add_argument(
        "--max-download-rounds",
        type=int,
        default=3,
        help="Retry rounds for transient download failures.",
    )
    parser.add_argument(
        "--adaptive-cooldown",
        type=float,
        default=2.0,
        help="Seconds to wait between adaptive retry rounds.",
    )
    parser.set_defaults(auto_shard=True, adaptive_workers=True)
    args = parser.parse_args()
    run_download_historic(args)


if __name__ == "__main__":
    main()
