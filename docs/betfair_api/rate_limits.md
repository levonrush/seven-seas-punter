# Rate Limits Safeguards

Source links:
- https://support.developer.betfair.com/hc/en-us/articles/360000402291-What-are-the-API-Request-Limits-
- https://developer.betfair.com/exchange-api/

## Endpoint URL
- Applies primarily to `listMarketBook` market data requests on:
  - `POST https://api.betfair.com/exchange/betting/json-rpc/v1`
  - `POST https://api.betfair.com/exchange/betting/rest/v1.0/listMarketBook/`

## HTTP Method
- `POST`

## Required Headers
- `X-Application: <BETFAIR_APP_KEY>`
- `X-Authentication: <session_token>`
- `Content-Type: application/json`

## Example Request JSON
```json
{
  "jsonrpc": "2.0",
  "method": "SportsAPING/v1.0/listMarketBook",
  "params": {
    "marketIds": ["1.23456789", "1.23456790"],
    "priceProjection": {
      "priceData": ["EX_BEST_OFFERS"]
    }
  },
  "id": 1
}
```

## Example Response JSON
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32099,
    "message": "ANGX-0004",
    "data": {
      "APINGException": {
        "errorCode": "TOO_MUCH_DATA",
        "errorDetails": "Exceeded request limits"
      }
    }
  },
  "id": 1
}
```

## Important Constraints
- Market data budget formula: `(sum of request weights) * (number of marketIds) <= 200`.
- Betfair guidance caps `listMarketBook` at roughly 5 requests/second per market ID.
- `priceProjection` overrides increase weight. Keep projections minimal for polling.
- For live execution loops, enforce a fixed poll interval and jittered backoff on transient failures.

Examples from Betfair's weight table for `EX_BEST_OFFERS` with overrides:
- `rollupModel=STAKE`, `rollupLimit=100`: weight `17`
- `rollupModel=STAKE`, `rollupLimit=50`: weight `18`
- `rollupModel=MANAGED_LIABILITY`, `rollupLiabilityFactor=2`: weight `17`

## Common Failure Cases
- `TOO_MUCH_DATA` from overweight requests.
- `SERVICE_BUSY` under bursty traffic.
- `INVALID_SESSION_INFORMATION` when tokens expire during polling loops.
- Self-inflicted throttling from retry storms without backoff.
