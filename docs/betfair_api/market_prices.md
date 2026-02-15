# Market Prices (`listMarketBook`)

Source links:
- https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687510/listMarketBook
- https://support.developer.betfair.com/hc/en-us/articles/360000402291-What-are-the-API-Request-Limits-

## Endpoint URL
- JSON-RPC: `POST https://api.betfair.com/exchange/betting/json-rpc/v1`
- Rescript alternative: `POST https://api.betfair.com/exchange/betting/rest/v1.0/listMarketBook/`

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
  "method": "SportsAPING/v1.0/listMarketBook",
  "params": {
    "marketIds": ["1.23456789"],
    "priceProjection": {
      "priceData": ["EX_BEST_OFFERS", "EX_TRADED"]
    }
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
      "status": "OPEN",
      "isMarketDataDelayed": false,
      "totalMatched": 154210.12,
      "runners": [
        {
          "selectionId": 12345,
          "status": "ACTIVE",
          "lastPriceTraded": 4.2,
          "totalMatched": 21999.5,
          "ex": {
            "availableToBack": [{"price": 4.2, "size": 101.6}],
            "availableToLay": [{"price": 4.3, "size": 98.1}],
            "tradedVolume": [{"price": 4.2, "size": 4120.7}]
          }
        }
      ]
    }
  ],
  "id": 1
}
```

## Important Constraints
- Request budget rule: `(sum of market data weights) * (number of marketIds) <= 200`, otherwise `TOO_MUCH_DATA`.
- Betfair guidance: do not call `listMarketBook` faster than about 5 times/second for the same market.
- `priceProjection` choices materially change request weight; avoid expensive projections unless needed.
- Odds used for execution must respect Betfair tick increments (for example: `2.00-3.00` uses `0.02`, `3.00-4.00` uses `0.05`).

## Common Failure Cases
- `TOO_MUCH_DATA` from overweight price projections or too many market IDs.
- `INVALID_SESSION_INFORMATION` due to expired session.
- `SERVICE_BUSY` on bursty polling.
- Empty runner prices shortly before suspension or in illiquid markets.
