from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import requests

from shared.betfair.historic import HistoricDataClient, build_filter, dump_json
from shared.utils.progress import log


def _parse_date(date_str: str) -> dt.date:
    """Parse YYYY-MM-DD strings for historic API filters."""
    return dt.date.fromisoformat(date_str)


def _split_csv(value: str | None) -> List[str]:
    """Split comma-separated strings into a list of trimmed values."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_workers(value: int | None) -> tuple[int, bool]:
    """Choose a safe default worker count for downloads."""
    if value and value > 0:
        return value, False
    cpu_count = os.cpu_count() or 4
    auto = max(2, min(8, cpu_count // 2))
    return auto, True


def _default_output(from_date: dt.date, to_date: dt.date) -> Path:
    """Build a stable tar output path for the date window."""
    name = f"data/historic_{from_date.isoformat()}_{to_date.isoformat()}.tar"
    return Path(name)


def _month_bounds(date_value: dt.date) -> tuple[dt.date, dt.date]:
    """Return first/last day for the month containing the provided date."""
    first = date_value.replace(day=1)
    last_day = calendar.monthrange(date_value.year, date_value.month)[1]
    return first, date_value.replace(day=last_day)


def _parse_package_month(value: str) -> dt.date:
    """Parse Betfair GetMyData forDate values into a date."""
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _load_manifest(path: Path) -> dict:
    """Load or initialize the historic download manifest."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"files": {}, "version": 1}
    files = data.get("files") or {}
    data["files"] = files
    data["_basenames"] = {entry.get("basename") for entry in files.values() if entry.get("basename")}
    return data


def _save_manifest(path: Path, manifest: dict) -> None:
    """Persist the historic download manifest to disk."""
    manifest = {k: v for k, v in manifest.items() if not k.startswith("_")}
    manifest["updated_at"] = dt.datetime.utcnow().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


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
) -> List[Path]:
    """Download many files in parallel to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for file_path in file_paths:
        if not force and _is_downloaded(manifest, file_path):
            continue
        files.append(file_path)
    saved: List[Path] = []
    total = len(files)
    if total == 0:
        return saved
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, file_path in enumerate(files, start=1):
            dest = output_dir / Path(file_path).name
            if dest.exists() and not force:
                _mark_downloaded(manifest, file_path, dest)
                continue
            futures[executor.submit(client.download_file, file_path, str(dest), retries, retry_wait)] = (
                dest,
                file_path,
            )
            if progress_every and idx % progress_every == 0:
                log(f"Historic download: queued {idx}/{total} files.")
        done = 0
        for future in as_completed(futures):
            dest, file_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                log(f"Historic download: failed {file_path} ({exc}).")
                if dest.exists():
                    dest.unlink(missing_ok=True)
                continue
            saved.append(dest)
            done += 1
            if progress_every and done % progress_every == 0:
                log(f"Historic download: completed {done}/{total} files.")
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
    if not (args.download or args.list_only or args.size_only or args.show_options):
        args.download = True
    workers, auto_workers = _resolve_workers(args.workers)
    args.workers = workers
    if auto_workers:
        log(f"Historic download: using {workers} workers (auto).")
    market_types = _split_csv(args.market_types)
    countries = _split_csv(args.countries)
    file_types = _split_csv(args.file_types)

    client = HistoricDataClient()

    manifest_path = Path(args.manifest_path or "data/historic_manifest.json")
    manifest = _load_manifest(manifest_path)

    def handle_window(from_date: dt.date, to_date: dt.date, plan: str) -> List[str]:
        payload = build_filter(
            sport=args.sport,
            plan=plan,
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

        if args.show_options:
            options = client.get_collection_options(payload)
            print(dump_json(options))
            return []
        if args.size_only:
            size = client.get_basket_size(payload)
            print(dump_json(size))
            return []
        files = []
        last_error = None
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
                    f"{from_date.isoformat()} to {to_date.isoformat()} ({plan}) "
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
                    f"{from_date.isoformat()} to {to_date.isoformat()} ({plan}) "
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

    if args.auto:
        packages = client.get_my_data()
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
        output_tar = Path(args.output) if args.output else Path("data/data.tar")
        _seed_manifest_from_tar(manifest, output_tar)
        output_dir = Path(args.output_dir) if args.output_dir else output_tar.with_suffix("")
        if args.clean_temp and output_dir.exists():
            log(f"Historic download: cleaning temp dir {output_dir}.")
            shutil.rmtree(output_dir, ignore_errors=True)
        for for_date, plan in sorted(package_months):
            from_date, to_date = _month_bounds(for_date)
            files = handle_window(from_date, to_date, plan or args.plan or "Basic Plan")
            if not files:
                continue
            log(
                "Historic download: "
                f"{from_date.isoformat()} to {to_date.isoformat()} ({plan})"
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
            )
            for path in saved:
                file_path = next(
                    (fp for fp in files if Path(fp).name == path.name), None
                )
                if file_path:
                    _mark_downloaded(manifest, file_path, path)
            if saved:
                log("Historic download: appending to tar archive.")
                _tar_files(output_tar, saved)
                if not args.keep_files:
                    for file_path in saved:
                        file_path.unlink(missing_ok=True)
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

    files = handle_window(from_date, to_date, plan_name)
    if not files:
        return

    output_tar = Path(args.output) if args.output else _default_output(from_date, to_date)
    _seed_manifest_from_tar(manifest, output_tar)
    output_dir = Path(args.output_dir) if args.output_dir else output_tar.with_suffix("")
    if args.clean_temp and output_dir.exists():
        log(f"Historic download: cleaning temp dir {output_dir}.")
        shutil.rmtree(output_dir, ignore_errors=True)
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
    )
    for path in saved:
        file_path = next((fp for fp in files if Path(fp).name == path.name), None)
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
    parser.add_argument("--market-types", default="WIN", help="Comma-separated market types.")
    parser.add_argument("--countries", default="AU", help="Comma-separated country codes.")
    parser.add_argument("--file-types", default="M", help="Comma-separated file types (M/E).")
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--event-name")
    parser.add_argument("--show-options", action="store_true", help="Show available filters.")
    parser.add_argument("--size-only", action="store_true", help="Only show file count and size.")
    parser.add_argument("--list-only", action="store_true", help="Only list available files.")
    parser.add_argument("--list-output", help="Optional path to write the file list JSON.")
    parser.add_argument("--download", action="store_true", help="Download the files.")
    parser.add_argument("--output", help="Output tar path (defaults to data/historic_*.tar).")
    parser.add_argument("--output-dir", help="Directory to store downloads before tarring.")
    parser.add_argument("--keep-files", action="store_true", help="Keep downloaded files on disk.")
    parser.add_argument("--clean-temp", action="store_true", help="Remove temp download dir before starting.")
    parser.add_argument("--max-files", type=int, help="Limit number of files (useful for testing).")
    parser.add_argument("--workers", type=int, default=None, help="Parallel download workers.")
    parser.add_argument("--progress-every", type=int, default=50, help="Progress logging frequency.")
    parser.add_argument("--manifest-path", help="Path for download manifest JSON.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if seen before.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count for historic API failures.")
    parser.add_argument("--retry-wait", type=float, default=3.0, help="Seconds to wait between retries.")
    args = parser.parse_args()
    run_download_historic(args)


if __name__ == "__main__":
    main()
