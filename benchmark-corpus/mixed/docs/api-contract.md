# API Contract

Base path: `/auth`  
Implementation: `src/api/routes.py`  
Business logic: `src/auth/auth_service.py`

All request bodies are JSON. All responses are JSON. All tokens are opaque strings.

---

## POST /auth/login

**Purpose:** Exchange credentials for an access + refresh token pair.

**Request:**
```json
{"username": "alice", "password": "s3cr3t"}
```

**Success 200:**
```json
{
  "access_token":  "<opaque>",
  "refresh_token": "<opaque>",
  "session_id":    "<opaque>",
  "expires_in":    900
}
```

**Guarantees:**
- `expires_in` is always `ACCESS_TTL = 900` seconds (15 minutes). Not runtime-configurable.
- `refresh_token` is valid for `REFRESH_TTL = 604800` seconds (7 days).
- If the user already has `MAX_SESSIONS = 5` concurrent sessions, the oldest is evicted (FIFO)
  before the new session is created. The evicted device receives no notification.
- `session_id` is independent of the tokens and is used for the logout call.

**Errors:**
- `401 {"error": "invalid_credentials"}` — wrong username or password.
- `403 {"error": "account_locked"}` — `user.is_active == False`.

---

## POST /auth/logout

**Purpose:** Invalidate the current session and access token.

**Headers:** `Authorization: Bearer <access_token>`

**Request:**
```json
{"session_id": "<session_id from login response>"}
```

**Success 200:**
```json
{"ok": true}
```

**Important limitation:** The refresh token is NOT invalidated on logout.
A client that retained the refresh token may still exchange it for up to
`REFRESH_TTL = 604800` seconds after logout. See [known-limitations.md](known-limitations.md)
§Logout does not invalidate refresh tokens.

---

## POST /auth/refresh

**Purpose:** Exchange a refresh token for a new access + refresh token pair.

**Request:**
```json
{"refresh_token": "<token from login or previous refresh response>"}
```

**Success 200:** Same shape as `/auth/login` response.

**Guarantees:**
- **Single-use:** The submitted `refresh_token` is consumed by `TokenStore.consume_refresh()`
  atomically on arrival, before any new tokens are issued.
- The new `refresh_token` in the response has a fresh `REFRESH_TTL = 604800`-second window.
- `expires_in` in the response is always `900`.
- A new `session_id` is issued on each successful refresh.

**Critical edge case:** If the network drops after `consume_refresh()` executes but before
the client receives the response, the old refresh token is permanently invalid. The client
cannot retry with it — they must re-authenticate with credentials.
See [known-limitations.md](known-limitations.md) §consume_refresh is not idempotent.

**Errors:**
- `401 {"error": "invalid_or_expired_refresh_token"}` — already consumed, expired, or never issued.

---

## GET /auth/sessions

**Purpose:** List active sessions for the authenticated user.

**Headers:** `Authorization: Bearer <access_token>`

**Permission required:** `api.sessions.list` at `READ` level.
Checked via `User.has_permission("api.sessions.list")` in the route handler.

**Success 200:**
```json
{
  "sessions": [
    {"session_id": "...", "created_at": 1712345678, "last_seen": 1712349999}
  ],
  "count":       1,
  "max_allowed": 5
}
```

**Guarantees:**
- `max_allowed` is always `MAX_SESSIONS = 5`.
- `count` is at most 5 at any point in time.
- Sessions are listed oldest-first (FIFO order matches eviction order).

**Errors:**
- `401 {"error": "invalid_token"}` — access token invalid or expired.
- `403 {"error": "forbidden"}` — user lacks `api.sessions.list` permission.

---

## Token lifetime summary

| Token   | Lifetime       | Constant                          |
|---------|----------------|-----------------------------------|
| access  | 900 s (15 min) | `ACCESS_TTL`  in `token_store.py` |
| refresh | 604800 s (7 d) | `REFRESH_TTL` in `token_store.py` |

Session cap: `MAX_SESSIONS = 5`, defined in `session_manager.py`.
