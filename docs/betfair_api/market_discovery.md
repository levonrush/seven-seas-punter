# Market Discovery (`listMarketCatalogue`)

Source links:
- https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687465/listMarketCatalogue
- https://developer.betfair.com/exchange-api/

## Endpoint URL
- JSON-RPC: `POST https://api.betfair.com/exchange/betting/json-rpc/v1`
- Rescript alternative: `POST https://api.betfair.com/exchange/betting/rest/v1.0/listMarketCatalogue/`

## HTTP Method
- `POST`

## Required Headers
- `X-Application: <BETFAIR_APP_KEY>`
- `X-Authentication: <session_token>`
- `Content-Type: application/json`
- `Accept: application/json`

## Example Request JSON
```json
{
  "jsonrpc": "2.0",
  "method": "SportsAPING/v1.0/listMarketCatalogue",
  "params": {
    "filter": {
      "eventTypeIds": ["7"],
      "marketCountries": ["AU"],
      "marketTypeCodes": ["WIN"],
      "marketStartTime": {
        "from": "2026-02-12T00:00:00Z",
        "to": "2026-02-12T23:59:59Z"
      }
    },
    "marketProjection": ["EVENT", "MARKET_START_TIME", "RUNNER_METADATA"],
    "sort": "FIRST_TO_START",
    "maxResults": "200"
  },
  "id": 1
}
```

## Example Response JSON
```json
{
  "jsonrpc": "2.0",
  "result": [
    {
      "marketId": "1.23456789",
      "marketName": "R6 1200m Grp1",
      "totalMatched": 153244.56,
      "marketStartTime": "2026-02-12T03:10:00.000Z",
      "event": {
        "id": "321654",
        "name": "Randwick",
        "countryCode": "AU",
        "timezone": "Australia/Sydney",
        "venue": "Randwick"
      },
      "runners": [
        {
          "selectionId": 12345,
          "runnerName": "Example Runner"
        }
      ]
    }
  ],
  "id": 1
}
```

## Important Constraints
- `maxResults` is mandatory and must be a positive value.
- Keep filters tight (`eventTypeIds`, `country`, `marketType`, and time window) to avoid oversized payloads.
- `marketProjection` controls payload size and should only include fields required downstream.
- Typical operational ceiling is `maxResults <= 1000` for practical paging behavior.

## Common Failure Cases
- `INVALID_SESSION_INFORMATION` (expired or missing session token).
- `TOO_MUCH_DATA` (filter too broad).
- `INPUT_VALIDATION_ERROR` (invalid filter shape or missing mandatory fields).
- `SERVICE_BUSY` / transient platform errors.
