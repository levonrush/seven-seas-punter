# Betfair integration

This module wraps Betfair API-NG for live markets and the Historic Data API for archive downloads.

## Environment variables
Required for historic downloads:
- `BETFAIR_APP_KEY`
- `BETFAIR_USERNAME`
- `BETFAIR_PASSWORD`

Optional (AU accounts can run without certs):
- `BETFAIR_CERT_PATH` (directory containing `client-2048.crt` + `client-2048.key`)
- `BETFAIR_CERT_FILE`
- `BETFAIR_KEY_FILE`

Optional SSO overrides and retries:
- `BETFAIR_SSO_URL` (e.g., `https://identitysso.betfair.com.au/api/login`)
- `BETFAIR_SSO_RETRIES`
- `BETFAIR_SSO_RETRY_WAIT`
- `BETFAIR_HISTORIC_TIMEOUT` (seconds for historic API reads, default `60`)
- `BETFAIR_HISTORIC_MAX_REQUESTS` (historic API requests per window, default `90`)
- `BETFAIR_HISTORIC_REQUEST_WINDOW` (seconds for the request window, default `10`)

## Files
- `client.py`: lightweight wrapper used by workflow scoring.
- `historic.py`: historic API client used by the downloader.
