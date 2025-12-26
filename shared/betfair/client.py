import datetime as dt
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from shared.utils.progress import log

try:
    import betfairlightweight
except ImportError:  # pragma: no cover - optional for environments without dependency
    betfairlightweight = None

logger = logging.getLogger(__name__)
load_dotenv()


class BetfairClient:
    """Minimal Betfair API wrapper with a dry-run fallback so workflows can run without credentials."""

    def __init__(self, app_key: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None) -> None:
        """Establish session when credentials exist; otherwise mark the client as dry-run."""
        self.app_key = app_key or os.getenv("BETFAIR_APP_KEY")
        self.username = username or os.getenv("BETFAIR_USERNAME")
        self.password = password or os.getenv("BETFAIR_PASSWORD")
        self.cert_file = os.getenv("BETFAIR_CERT_FILE")
        self.key_file = os.getenv("BETFAIR_KEY_FILE")
        self._dry_run = not all([self.app_key, self.username, self.password, betfairlightweight])
        self._client = None
        if not self._dry_run and betfairlightweight:
            try:
                trading = betfairlightweight.APIClient(
                    username=self.username,
                    password=self.password,
                    app_key=self.app_key,
                    cert_files=(self.cert_file, self.key_file) if self.cert_file and self.key_file else None,
                )
                trading.login()
                self._client = trading
                logger.info("BetfairClient logged in as %s", self.username)
                log(f"BetfairClient: logged in as {self.username}")
            except Exception as exc:  # pragma: no cover - network/runtime specific
                self._dry_run = True
                logger.warning("Login failed (%s); switching to dry-run.", exc)
                log(f"BetfairClient: login failed; switching to dry-run ({exc})")
        else:
            logger.warning("BetfairClient running in dry-run mode (missing credentials or dependency).")
            log("BetfairClient: dry-run mode (missing credentials or dependency).")

    @property
    def dry_run(self) -> bool:
        """Expose dry-run status to callers that may choose alternate code paths."""
        return self._dry_run

    def list_markets_for_date(
        self, date: dt.date, country: str = "AU", event_type: str = "horse_racing"
    ) -> List[Dict[str, Any]]:
        """Retrieve market catalogues for a given date; returns mocks when dry-run."""
        if self._dry_run or not self._client:
            return self._mock_catalogues(date)

        market_filter = betfairlightweight.filters.market_filter(
            event_type_ids=[self._lookup_event_type_id(event_type)],
            market_countries=[country],
            market_start_time={
                "from": dt.datetime.combine(date, dt.time.min).isoformat(),
                "to": dt.datetime.combine(date, dt.time.max).isoformat(),
            },
        )
        catalogue = self._client.betting.list_market_catalogue(
            filter=market_filter,
            market_projection=["MARKET_START_TIME", "EVENT", "RUNNER_METADATA"],
            max_results=1000,
        )
        results = []
        for item in catalogue:
            results.append(
                {
                    "market_id": item.market_id,
                    "venue": item.event.venue,
                    "race_start_time": item.market_start_time,
                    "race_name": item.market_name,
                    "country_code": country,
                    "event_type": event_type,
                    "runners": [
                        {
                            "market_id": item.market_id,
                            "selection_id": runner.selection_id,
                            "runner_name": runner.runner_name,
                            "stall_draw": runner.metadata.get("STALL_DRAW") if runner.metadata else None,
                        }
                        for runner in item.runners
                    ],
                }
            )
        return results

    def fetch_market_books(self, market_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetch live market books for a list of markets; returns mocks when dry-run."""
        market_ids = list(market_ids)
        if self._dry_run or not self._client:
            return self._mock_market_books(market_ids)
        books = self._client.betting.list_market_book(
            market_ids=market_ids,
            price_projection=betfairlightweight.filters.price_projection(price_data=["EX_BEST_OFFERS", "EX_TRADED"]),
        )
        payload: List[Dict[str, Any]] = []
        for book in books:
            for runner in book.runners:
                best_back = runner.ex.available_to_back[0] if runner.ex.available_to_back else None
                best_lay = runner.ex.available_to_lay[0] if runner.ex.available_to_lay else None
                payload.append(
                    {
                        "market_id": book.market_id,
                        "selection_id": runner.selection_id,
                        "snapshot_time": dt.datetime.utcnow(),
                        "seconds_to_start": (book.market_definition.market_time - dt.datetime.utcnow()).total_seconds()
                        if book.market_definition and book.market_definition.market_time
                        else None,
                        "best_back_price": best_back.price if best_back else None,
                        "best_back_size": best_back.size if best_back else None,
                        "best_lay_price": best_lay.price if best_lay else None,
                        "best_lay_size": best_lay.size if best_lay else None,
                        "last_traded_price": runner.last_price_traded,
                        "total_matched": runner.total_matched,
                        "runner_status": runner.status,
                        "venue": book.market_definition.venue if book.market_definition else None,
                        "race_start_time": book.market_definition.market_time if book.market_definition else None,
                        "race_name": book.market_definition.market_type if book.market_definition else None,
                    }
                )
        return payload

    def fetch_market_results(self, market_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Fetch results; dry-run derives winners from mock data while live mode uses cleared orders proxy."""
        market_ids = list(market_ids)
        if self._dry_run or not self._client:
            return self._mock_results(market_ids)
        results: List[Dict[str, Any]] = []
        # Betfair API-NG has limited settled market endpoints; cleared orders is a pragmatic proxy for now.
        for market_id in market_ids:
            cleared = self._client.betting.list_cleared_orders(
                bet_status="SETTLED", market_ids=[market_id], group_by="RUNNER", record_count=1000
            )
            winners = {item.selection_id for item in cleared.current_orders if item.win_lose}
            for item in cleared.current_orders:
                results.append(
                    {
                        "market_id": market_id,
                        "selection_id": item.selection_id,
                        "win_flag": item.selection_id in winners,
                        "bsp": None,
                        "place_position": None,
                    }
                )
        return results

    def _lookup_event_type_id(self, event_type: str) -> str:
        """Map human labels to Betfair eventTypeIds; keeps surface API readable."""
        mapping = {"horse_racing": "7"}
        return mapping.get(event_type, "7")

    def _mock_catalogues(self, date: dt.date) -> List[Dict[str, Any]]:
        """Provide deterministic catalogues for offline workflows."""
        base_time = dt.datetime.combine(date, dt.time(hour=4, minute=0))
        markets = []
        for i in range(3):
            market_id = f"1.{date.strftime('%Y%m%d')}{i:02d}"
            markets.append(
                {
                    "market_id": market_id,
                    "venue": f"Mockville {i}",
                    "race_start_time": base_time + dt.timedelta(minutes=30 * i),
                    "race_name": f"Dry Run Stakes {i}",
                    "country_code": "AU",
                    "event_type": "horse_racing",
                    "runners": [
                        {"market_id": market_id, "selection_id": 100 + j, "runner_name": f"Runner {j}", "stall_draw": j}
                        for j in range(8)
                    ],
                }
            )
        return markets

    def _mock_market_books(self, market_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Generate synthetic market book snapshots for deterministic tests."""
        now = dt.datetime.utcnow()
        books: List[Dict[str, Any]] = []
        for idx, market_id in enumerate(market_ids):
            race_time = now + dt.timedelta(minutes=60 + idx * 5)
            for runner_idx in range(8):
                base_price = 2.0 + runner_idx * 0.5
                books.append(
                    {
                        "market_id": market_id,
                        "selection_id": 100 + runner_idx,
                        "snapshot_time": now,
                        "seconds_to_start": int((race_time - now).total_seconds()),
                        "best_back_price": base_price,
                        "best_back_size": 100.0,
                        "best_lay_price": base_price + 0.1,
                        "best_lay_size": 120.0,
                        "last_traded_price": base_price + 0.05,
                        "total_matched": 1000 + 50 * runner_idx,
                        "runner_status": "ACTIVE",
                        "venue": f"Mockville {idx}",
                        "race_start_time": race_time,
                        "race_name": f"Dry Run Stakes {idx}",
                    }
                )
        return books

    def _mock_results(self, market_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Produce simple winners for mock markets so targets exist."""
        results: List[Dict[str, Any]] = []
        for market_id in market_ids:
            winner = 100  # deterministic
            for runner_idx in range(8):
                selection_id = 100 + runner_idx
                results.append(
                    {
                        "market_id": market_id,
                        "selection_id": selection_id,
                        "win_flag": selection_id == winner,
                        "bsp": 2.0 + runner_idx * 0.5,
                        "place_position": None,
                    }
                )
        return results
