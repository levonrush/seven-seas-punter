# Bet Execution (`placeOrders`)

Source links:
- https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687496/placeOrders
- https://developer.betfair.com/exchange-api/

## Endpoint URL
- JSON-RPC: `POST https://api.betfair.com/exchange/betting/json-rpc/v1`
- Rescript alternative: `POST https://api.betfair.com/exchange/betting/rest/v1.0/placeOrders/`

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
  "method": "SportsAPING/v1.0/placeOrders",
  "params": {
    "marketId": "1.23456789",
    "customerRef": "live-20260212-0001",
    "instructions": [
      {
        "selectionId": 12345,
        "side": "BACK",
        "orderType": "LIMIT",
        "limitOrder": {
          "size": 5.0,
          "price": 4.2,
          "persistenceType": "LAPSE"
        }
      }
    ]
  },
  "id": 1
}
```

## Example Response JSON
```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "SUCCESS",
    "marketId": "1.23456789",
    "instructionReports": [
      {
        "status": "SUCCESS",
        "instruction": {
          "selectionId": 12345,
          "side": "BACK",
          "orderType": "LIMIT"
        },
        "betId": "312345678901",
        "placedDate": "2026-02-12T02:59:59.000Z",
        "averagePriceMatched": 0.0,
        "sizeMatched": 0.0
      }
    ]
  },
  "id": 1
}
```

## Important Constraints
- `placeOrders` supports up to 200 instructions per request.
- Throttle trading operations (`placeOrders`, `cancelOrders`, `replaceOrders`, `updateOrders`) conservatively; avoid burst retries in the same market.
- Minimum stake/liability rules are currency-dependent (for GBP, docs show min bet size `1` and minimum BSP liability `10`).
- Submitted price must be on Betfair's valid odds ladder or the instruction is rejected.
- Use `customerRef` for idempotency and auditability.

## Common Failure Cases
- `INVALID_ODDS` (price not on the ladder).
- `INSUFFICIENT_FUNDS`.
- `MARKET_SUSPENDED` / `MARKET_NOT_OPEN_FOR_BETTING`.
- `INVALID_MARKET_ID` or `INVALID_SELECTION_ID`.
- Duplicate `customerRef` handling issues in retry flows.
