# Known Limitations

Tracks design trade-offs, gaps, and planned improvements in the auth system.

---

## 1. Logout does not invalidate refresh tokens

**Severity:** Medium  
**Component:** `AuthService.logout()` in `src/auth/auth_service.py`

`AuthService.logout()` removes the session via `SessionManager.remove_session()` and
expires the access token by deleting it from `TokenStore`. However, the associated
refresh token (if the client retained it) remains valid in `TokenStore` for up to
`REFRESH_TTL = 604800` seconds (7 days) after logout.

**Impact:** A user who clicks "Log out" on a compromised device leaves a live refresh
token behind. An attacker can exchange it for a fresh access token.

**Workaround:** Call `TokenStore.revoke_all_for_user(user_id)` to delete all tokens
for the user. `AuthService` has the `user_id` available during logout but does not
call `revoke_all_for_user()` automatically to avoid breaking multi-device users whose
other sessions share the same `user_id`.

**Planned fix:** Add a `revoke_refresh: bool = False` parameter to `AuthService.logout()`.
When `True`, call `TokenStore.revoke_all_for_user(user_id)` to nuke all tokens.

---

## 2. FIFO session eviction is silent

**Severity:** Low  
**Component:** `SessionManager.create_session()` in `src/auth/session_manager.py`

When a user reaches `MAX_SESSIONS = 5` concurrent sessions and logs in again,
`SessionManager` evicts the oldest session via `OrderedDict.popitem(last=False)`.
The evicted device receives no notification — its next API call returns
`401 {"error": "invalid_token"}` without explanation.

**Impact:** Confusing user experience on long-lived background sessions (e.g.
a desktop app left open for days).

**Planned fix:** Emit a notification event to a queue (e.g. Redis pub/sub) when a
session is evicted, so the client can display a "you were signed out on another device"
message.

---

## 3. consume_refresh is not idempotent

**Severity:** Low  
**Component:** `TokenStore.consume_refresh()` in `src/auth/token_store.py`

By design (see [ADR-001](adr-001-single-use-tokens.md)), `consume_refresh()` is an
atomic pop. If the network delivers the `POST /auth/refresh` request to the server but
drops the response before the client receives it, the refresh token has been consumed
but the client never got the new tokens. A retry with the same refresh token returns
`401 invalid_or_expired_refresh_token`.

**Impact:** The user must re-authenticate with credentials on unreliable mobile networks.

**Planned fix:** Introduce a short-lived "pending rotation" grace window: the old token
is tombstoned (not fully deleted) for 30 seconds, allowing exactly one retry within that
window. After 30 seconds the tombstone expires and the token is permanently gone.

---

## 4. In-memory backends are not production-safe

**Severity:** High (deployment concern)  
**Components:** `TokenStore.__init__()`, `SessionManager.__init__()`

Both `TokenStore` and `SessionManager` default to in-memory storage
(`dict` and `defaultdict(OrderedDict)` respectively). All sessions and tokens are lost
on process restart and cannot be shared across multiple `AuthService` instances.

**Planned fix:** Inject a Redis-backed backend via `TokenStore(backend=redis_client)`.
The Redis key schema (`token:access:{sha256}`, `token:refresh:{sha256}`) is documented
in the `TokenStore` class docstring. `SessionManager` requires a similar Redis-backed
`OrderedDict` equivalent using sorted sets.

---

## 5. has_permission does not support wildcards

**Severity:** Low  
**Component:** `User.has_permission()` in `src/models/user.py`

`has_permission(name, level)` does an exact string match on `Permission.name`. A
permission named `"api.*"` is NOT treated as a wildcard for `"api.sessions.list"`.

**Impact:** Admin users need an explicit grant for every permission name. Assigning
broad access to a new admin requires updating every permission individually.

**Planned fix:** Add `fnmatch`-based glob matching in `User.has_permission()`:
```python
import fnmatch
if fnmatch.fnmatch(name, perm.name) and perm.allows(level):
    return True
```
