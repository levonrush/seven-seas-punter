from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

from shared.utils.progress import log

load_dotenv()


HISTORIC_BASE_URL = "https://historicdata.betfair.com/api"
SSO_LOGIN_URL = "https://identitysso.betfair.com/api/login"
SSO_LOGIN_URL_AU = "https://identitysso.betfair.com.au/api/login"
SSO_CERT_LOGIN_URL = "https://identitysso.betfair.com/api/certlogin"
SSO_CERT_LOGIN_URL_AU = "https://identitysso.betfair.com.au/api/certlogin"


@dataclass
class HistoricFilter:
    """Filter payload for Betfair historic data endpoints."""

    sport: str
    plan: str
    from_day: int
    from_month: int
    from_year: int
    to_day: int
    to_month: int
    to_year: int
    event_id: Optional[int]
    event_name: Optional[str]
    market_types: List[str]
    countries: List[str]
    file_types: List[str]

    def to_payload(self) -> Dict[str, Any]:
        """Return API-ready filter payload for historic data endpoints."""
        return {
            "sport": self.sport,
            "plan": self.plan,
            "fromDay": self.from_day,
            "fromMonth": self.from_month,
            "fromYear": self.from_year,
            "toDay": self.to_day,
            "toMonth": self.to_month,
            "toYear": self.to_year,
            "eventId": self.event_id,
            "eventName": self.event_name,
            "marketTypesCollection": self.market_types,
            "countriesCollection": self.countries,
            "fileTypeCollection": self.file_types,
        }


class _RequestRateLimiter:
    """Apply a simple sliding-window cap so multi-threaded downloads do not burst into API throttles."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        """Store limits used to gate outbound historic API requests."""
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(0.1, float(window_seconds))
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the current request can proceed within the configured request budget."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return
                wait_for = self.window_seconds - (now - self._timestamps[0]) + 0.01
            if wait_for > 0:
                time.sleep(min(wait_for, 0.5))


def _env_int(name: str, default: int) -> int:
    """Read integer env vars with a fallback so malformed local env files do not crash startup."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read float env vars with a fallback so malformed local env files do not crash startup."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class HistoricDataClient:
    """Client for Betfair historic data API using SSO session tokens."""

    def __init__(
        self,
        app_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        max_requests: Optional[int] = None,
        request_window_seconds: Optional[float] = None,
    ) -> None:
        """Authenticate via Betfair SSO and prepare a session for historic API calls."""
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY")
        self.username = username or os.getenv("BETFAIR_USERNAME")
        self.password = password or os.getenv("BETFAIR_PASSWORD")
        self.cert_file = cert_file or os.getenv("BETFAIR_CERT_FILE")
        self.key_file = key_file or os.getenv("BETFAIR_KEY_FILE")
        if self.cert_file and not os.path.isfile(self.cert_file):
            log(f"HistoricDataClient: cert file missing, ignoring {self.cert_file}.")
            self.cert_file = None
        if self.key_file and not os.path.isfile(self.key_file):
            log(f"HistoricDataClient: key file missing, ignoring {self.key_file}.")
            self.key_file = None
        if not (self.cert_file and self.key_file):
            self.cert_file = None
            self.key_file = None
        if not all([self.app_key, self.username, self.password]):
            raise ValueError("Missing BETFAIR_APP_KEY/BETFAIR_USERNAME/BETFAIR_PASSWORD.")
        self.session = requests.Session()
        self.historic_timeout = _env_float("BETFAIR_HISTORIC_TIMEOUT", 60.0)
        # Keep a small safety margin under published limits so retry bursts stay compliant.
        env_max_requests = _env_int("BETFAIR_HISTORIC_MAX_REQUESTS", 90)
        env_window = _env_float("BETFAIR_HISTORIC_REQUEST_WINDOW", 10.0)
        self.max_requests = max_requests if max_requests is not None else env_max_requests
        self.request_window_seconds = (
            request_window_seconds if request_window_seconds is not None else env_window
        )
        self._rate_limiter = _RequestRateLimiter(self.max_requests, self.request_window_seconds)
        self.ssoid = self._login()

    def _login(self) -> str:
        """Login to Betfair SSO using app key and return the session token."""
        use_cert = bool(self.cert_file and self.key_file)
        override_url = os.getenv("BETFAIR_SSO_CERT_URL" if use_cert else "BETFAIR_SSO_URL")
        urls = [override_url] if override_url else []
        if use_cert:
            urls += [SSO_CERT_LOGIN_URL, SSO_CERT_LOGIN_URL_AU]
        else:
            urls += [SSO_LOGIN_URL, SSO_LOGIN_URL_AU]
        headers = {
            "X-Application": self.app_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"username": self.username, "password": self.password}
        cert = (self.cert_file, self.key_file) if use_cert else None
        errors: List[str] = []
        retries = _env_int("BETFAIR_SSO_RETRIES", 5)
        retry_wait = _env_float("BETFAIR_SSO_RETRY_WAIT", 3.0)
        for url in dict.fromkeys(urls):
            if not url:
                continue
            for attempt in range(1, retries + 1):
                log(
                    "HistoricDataClient: "
                    f"logging in ({'cert' if use_cert else 'password'} login) via {url} "
                    f"(attempt {attempt}/{retries})."
                )
                try:
                    resp = requests.post(url, headers=headers, data=data, cert=cert, timeout=30)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    if attempt < retries:
                        sleep_for = retry_wait * attempt
                        log(f"HistoricDataClient: login failed ({exc}); retrying in {sleep_for}s.")
                        time.sleep(sleep_for)
                        continue
                    errors.append(f"{url} -> request failed: {exc}")
                    break
                try:
                    payload = resp.json()
                except requests.JSONDecodeError:
                    snippet = resp.text.strip().replace("\n", " ")[:200]
                    errors.append(f"{url} -> non-JSON response: {snippet}")
                    break
                if payload.get("status") != "SUCCESS":
                    raise RuntimeError(f"SSO login failed: {payload}")
                token = payload.get("sessionToken") or payload.get("token")
                if not token:
                    raise RuntimeError(f"SSO login missing sessionToken: {payload}")
                return token
        error_text = "; ".join(errors) if errors else "no login URLs available"
        raise RuntimeError(
            "SSO login did not return JSON. Set BETFAIR_SSO_URL to the correct endpoint "
            f"(e.g., {SSO_LOGIN_URL_AU} for AU accounts). Details: {error_text}"
        )

    def _headers(self) -> Dict[str, str]:
        """Return headers required for historic data endpoints."""
        return {"ssoid": self.ssoid}

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Rate-limit requests centrally so all API calls share one throttle budget."""
        limiter = getattr(self, "_rate_limiter", None)
        if limiter is not None:
            limiter.acquire()
        request_fn = getattr(self.session, "request", None)
        if callable(request_fn):
            return request_fn(method, url, **kwargs)
        fallback = getattr(self.session, method.lower())
        return fallback(url, **kwargs)

    def get_my_data(
        self,
        retries: int = 3,
        retry_wait: float = 2.0,
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return purchased historic packages with retries because this call can timeout on large accounts."""
        attempts = max(1, int(retries))
        read_timeout = float(timeout if timeout is not None else self.historic_timeout)
        for attempt in range(1, attempts + 1):
            try:
                resp = self._request(
                    "GET",
                    f"{HISTORIC_BASE_URL}/GetMyData",
                    headers=self._headers(),
                    timeout=read_timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt < attempts:
                    wait_for = retry_wait * attempt
                    log(
                        "HistoricDataClient: "
                        f"GetMyData failed ({exc}); retrying in {wait_for}s "
                        f"(attempt {attempt}/{attempts})."
                    )
                    time.sleep(wait_for)
                    continue
                raise

    def get_collection_options(self, filter_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return available market/country/file filters for a given request window."""
        resp = self._request(
            "POST",
            f"{HISTORIC_BASE_URL}/GetCollectionOptions",
            headers=self._headers(),
            json=filter_payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_basket_size(self, filter_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return file count and total size for the given filter."""
        resp = self._request(
            "POST",
            f"{HISTORIC_BASE_URL}/GetAdvBasketDataSize",
            headers=self._headers(),
            json=filter_payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def list_files(self, filter_payload: Dict[str, Any]) -> List[str]:
        """Return a list of downloadable file paths for the given filter."""
        resp = self._request(
            "POST",
            f"{HISTORIC_BASE_URL}/DownloadListOfFiles",
            headers=self._headers(),
            json=filter_payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def download_file(self, file_path: str, output_path: str, retries: int = 3, retry_wait: float = 2.0) -> None:
        """Download a single historic data file to disk, retrying on transient errors."""
        encoded = urllib.parse.quote(file_path, safe="")
        url = f"{HISTORIC_BASE_URL}/DownloadFile?filePath={encoded}"
        for attempt in range(1, retries + 1):
            try:
                resp = self._request("GET", url, headers=self._headers(), stream=True, timeout=120)
                resp.raise_for_status()
                with open(output_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                return
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                retryable = status in {429} or (status is not None and status >= 500)
                if retryable and attempt < retries:
                    wait_for = retry_wait * attempt
                    log(
                        "Historic download: "
                        f"file failed (HTTP {status}); retrying in {wait_for}s "
                        f"({attempt}/{retries}) -> {file_path}"
                    )
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    time.sleep(wait_for)
                    continue
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise
            except requests.RequestException as exc:
                if attempt < retries:
                    wait_for = retry_wait * attempt
                    log(
                        "Historic download: "
                        f"file request error; retrying in {wait_for}s "
                        f"({attempt}/{retries}) -> {file_path} ({exc})"
                    )
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    time.sleep(wait_for)
                    continue
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise
            except Exception:
                if os.path.exists(output_path):
                    os.remove(output_path)
                raise


def build_filter(
    sport: str,
    plan: str,
    from_day: int,
    from_month: int,
    from_year: int,
    to_day: int,
    to_month: int,
    to_year: int,
    market_types: Iterable[str],
    countries: Iterable[str],
    file_types: Iterable[str],
    event_id: Optional[int] = None,
    event_name: Optional[str] = None,
) -> HistoricFilter:
    """Build a historic data filter payload from user inputs."""
    return HistoricFilter(
        sport=sport,
        plan=plan,
        from_day=from_day,
        from_month=from_month,
        from_year=from_year,
        to_day=to_day,
        to_month=to_month,
        to_year=to_year,
        event_id=event_id,
        event_name=event_name,
        market_types=[val for val in market_types if val],
        countries=[val for val in countries if val],
        file_types=[val for val in file_types if val],
    )


def dump_json(data: Any) -> str:
    """Return formatted JSON for console output."""
    return json.dumps(data, indent=2, sort_keys=True)
