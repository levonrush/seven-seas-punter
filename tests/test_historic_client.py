import bz2
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


class _DummyDownloadResponse:
    """Minimal streaming response for download_file tests without network calls."""

    def __init__(self, chunks, status_code: int = 200, headers=None) -> None:
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        """Raise HTTPError when status is non-success to mirror requests.Response behavior."""
        if self.status_code >= 400:
            error = requests.HTTPError(f"status={self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def iter_content(self, chunk_size=1024 * 1024):  # noqa: ARG002 - matches requests.Response API
        """Yield preloaded chunks so download code can stream bytes exactly as in production."""
        for chunk in self._chunks:
            yield chunk


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


def test_download_file_rejects_non_bzip2_payload(tmp_path):
    html = b"<!DOCTYPE html><html><head><title>package_not_purchased</title></head></html>"

    def fake_request(method, url, **kwargs):  # noqa: ANN001 - mirrors requests.Session.request
        return _DummyDownloadResponse([html], status_code=200, headers={"Content-Type": "text/html"})

    client = _build_client(SimpleNamespace(request=fake_request))
    output_path = tmp_path / "bad.bz2"

    with pytest.raises(RuntimeError, match="unexpected non-bzip2 payload"):
        client.download_file("/xds/path/1.2.bz2", str(output_path), retries=1, retry_wait=0.0)
    assert not output_path.exists()


def test_download_file_accepts_valid_bzip2_payload(tmp_path):
    payload = bz2.compress(b'{"pt":1,"mc":[]}\n')
    chunks = [payload[:8], payload[8:]]

    def fake_request(method, url, **kwargs):  # noqa: ANN001 - mirrors requests.Session.request
        return _DummyDownloadResponse(chunks, status_code=200, headers={"Content-Type": "application/octet-stream"})

    client = _build_client(SimpleNamespace(request=fake_request))
    output_path = tmp_path / "good.bz2"

    client.download_file("/xds/path/1.2.bz2", str(output_path), retries=1, retry_wait=0.0)

    assert output_path.exists()
    assert bz2.decompress(output_path.read_bytes()) == b'{"pt":1,"mc":[]}\n'
