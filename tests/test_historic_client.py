from types import SimpleNamespace

import pytest
import requests

from shared.betfair.historic import HistoricDataClient


class _DummyResponse:
    """Minimal response stand-in to exercise historic client retry behavior without network calls."""

    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise an HTTPError for non-success statuses to mirror requests.Response behavior."""
        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self):
        """Return the preloaded JSON payload."""
        return self._payload


def _build_client(fake_session) -> HistoricDataClient:
    """Construct a HistoricDataClient instance without running real auth so tests stay offline."""
    client = HistoricDataClient.__new__(HistoricDataClient)
    client.session = fake_session
    client.ssoid = "test-token"
    client.historic_timeout = 1.0
    return client


def test_get_my_data_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, headers, timeout):  # noqa: ANN001 - signature mirrors requests.Session.get
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return _DummyResponse([{"sport": "Horse Racing"}])

    client = _build_client(SimpleNamespace(get=fake_get))
    monkeypatch.setattr("shared.betfair.historic.time.sleep", lambda _: None)

    payload = client.get_my_data(retries=2, retry_wait=0.0)

    assert calls["count"] == 2
    assert payload == [{"sport": "Horse Racing"}]


def test_get_my_data_raises_after_retries(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, headers, timeout):  # noqa: ANN001 - signature mirrors requests.Session.get
        calls["count"] += 1
        raise requests.ReadTimeout("timed out")

    client = _build_client(SimpleNamespace(get=fake_get))
    monkeypatch.setattr("shared.betfair.historic.time.sleep", lambda _: None)

    with pytest.raises(requests.ReadTimeout):
        client.get_my_data(retries=3, retry_wait=0.0)
    assert calls["count"] == 3
