# Auth System Architecture

This document describes the three core authentication components and how they interact.
See `docs/api-contract.md` for the HTTP-level contract and `docs/adr-001-single-use-tokens.md`
for the rationale behind single-use refresh tokens.

---

## Components

### AuthService (`src/auth/auth_service.py`)

`AuthService` is the top-level orchestrator. All client-facing operations
(login, logout, refresh, validate) go through `AuthService`. It delegates
token storage to `TokenStore` and session tracking to `SessionManager`.

**Login flow:**
1. `AuthService.login(user)` checks `user.is_active`; raises `ValueError` if locked.
2. Calls `TokenStore.create_access_token()` — issues a 900-second access token.
3. Calls `TokenStore.create_refresh_token()` — issues a 604,800-second refresh token.
4. Calls `SessionManager.create_session()` — registers the session; FIFO eviction
   kicks in automatically if the user already holds `MAX_SESSIONS = 5`.
5. Returns `AuthResult` (access_token, refresh_token, session_id, expires_in=900).

**Refresh flow:**
1. `AuthService.refresh(refresh_token)` calls `TokenStore.consume_refresh()`.
2. `consume_refresh()` atomically deletes the token — this is the single-use guarantee.
3. A fresh access + refresh token pair is issued; a new session is created.
4. Returns `AuthResult` or `None` if the token was already consumed.

**Validate flow:**
1. `AuthService.validate(access_token)` calls `TokenStore.validate_access_token()`.
2. Returns the full `User` object (including `permissions`) so the caller can invoke
   `User.has_permission()` without a second lookup.

---

### TokenStore (`src/auth/token_store.py`)

`TokenStore` owns all token lifecycle constants:

| Constant       | Value   | Meaning                              |
|----------------|---------|--------------------------------------|
| `ACCESS_TTL`   | 900     | Access token lifetime in seconds     |
| `REFRESH_TTL`  | 604800  | Refresh token lifetime in seconds    |

Tokens are stored as SHA-256 hashes of the raw token string.

The critical method is `consume_refresh()`. It uses an atomic `dict.pop()` on the
backing store so that even two concurrent calls with the same token cannot both succeed.
The first call gets the user_id; the second gets `None`. See
[ADR-001](adr-001-single-use-tokens.md) for full motivation.

---

### SessionManager (`src/auth/session_manager.py`)

`SessionManager` enforces `MAX_SESSIONS = 5` per user using an `OrderedDict`-backed
FIFO queue. When a sixth session is created, the oldest entry (first in the
`OrderedDict`) is silently evicted via `popitem(last=False)`.

The `/auth/sessions` endpoint exposes the current session list and reports
`"max_allowed": 5` in the response body (see [API Contract](api-contract.md)).

---

## Dependency Graph

```
AuthService
├── TokenStore       (token issuance / validation / consumption)
│     ACCESS_TTL = 900, REFRESH_TTL = 604800
├── SessionManager   (session tracking / FIFO eviction)
│     MAX_SESSIONS = 5
└── User + Permission models
      User.has_permission(name, level)
      Permission.allows(required_level)
      PermissionLevel: READ=10, WRITE=20, ADMIN=30, SUPER=40
```

---

## Cross-cutting concerns

- **Permission checks** happen after `AuthService.validate()` returns a `User`.
  Flask route decorators in `src/api/routes.py` call `g.user.has_permission()`.
- `GET /auth/sessions` requires `has_permission("api.sessions.list")`.
- `Permission` and `PermissionLevel` are defined in `src/models/permission.py`.
- `User` is defined in `src/models/user.py` and carries a `FrozenSet[Permission]`.
