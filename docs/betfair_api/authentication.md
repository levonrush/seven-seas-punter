# Authentication

Source links:
- https://support.developer.betfair.com/hc/en-us/articles/115003899492-How-do-I-login-to-the-API-
- https://developer.betfair.com/exchange-api/

## Endpoint URL
- `POST https://identitysso-cert.betfair.com/api/certlogin` (non-interactive, automation/cert login)
- `POST https://identitysso.betfair.com/api/login` (interactive login, no client cert)
- `POST https://identitysso.betfair.com/api/keepAlive` (refresh session)
- `POST https://identitysso.betfair.com/api/logout` (terminate session)

## HTTP Method
- `POST`

## Required Headers
- `X-Application: <BETFAIR_APP_KEY>`
- `Content-Type: application/x-www-form-urlencoded` for login
- `Accept: application/json`
- `X-Authentication: <session_token>` for `keepAlive` and `logout`

## Example Request JSON
```json
{
  "username": "BETFAIR_USERNAME",
  "password": "BETFAIR_PASSWORD"
}
```

Note: Betfair SSO expects form encoding (`username=...&password=...`) rather than a JSON body.

## Example Response JSON
```json
{
  "sessionToken": "abc123-session-token",
  "loginStatus": "SUCCESS"
}
```

Failure shape:
```json
{
  "loginStatus": "FAIL",
  "error": "INVALID_USERNAME_OR_PASSWORD"
}
```

## Important Constraints
- Non-interactive automation should use certificate login (`certlogin`).
- Session token must be sent as `X-Authentication` on Exchange API calls.
- Keep session alive explicitly with `keepAlive` in long-running jobs.
- Never hardcode credentials, app keys, or certificate paths.

## Common Failure Cases
- Missing/invalid app key.
- Invalid username/password.
- Certificate missing, unreadable, or not paired correctly for cert login.
- Expired session token (`INVALID_SESSION_INFORMATION` in subsequent API calls).
