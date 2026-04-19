# ADR-001: Single-Use Refresh Token Rotation

**Status:** Accepted  
**Date:** 2024-01-15  
**Deciders:** Platform Security Team  
**Affects:** `TokenStore.consume_refresh()`, `AuthService.refresh()`

---

## Context

Prior to this decision refresh tokens were long-lived but reusable. An attacker who obtained
a refresh token — via a leaked log file, a compromised device backup, or a shoulder-surf —
could silently issue new access tokens indefinitely without any alarm being raised.

The system needed a mechanism to surface token theft proactively.

---

## Decision

Refresh tokens are **single-use**. The enforcement point is `TokenStore.consume_refresh()`
in `src/auth/token_store.py`.

`consume_refresh()` performs an atomic `dict.pop()` on the backing store:

```python
entry = self._store.pop(key, None)   # atomic delete
```

This guarantees:
1. The token is deleted **before** new tokens are issued.
2. Two concurrent requests with the same token: only one gets a `user_id` back; the other
   gets `None` and is rejected with `401 invalid_or_expired_refresh_token`.
3. If new token issuance fails after `consume_refresh()` succeeds, the old token is already
   gone — the client must re-authenticate.

`AuthService.refresh()` calls `consume_refresh()` as its **first** step. New tokens are
issued by `TokenStore.create_access_token()` and `TokenStore.create_refresh_token()` only
after `consume_refresh()` returns a valid `user_id`.

---

## Consequences

**Positive:**
- Any replay of a consumed refresh token returns `401` immediately.
- **Theft detection:** If a stolen token is replayed, the legitimate user's next refresh
  attempt fails (token already consumed). This surfaces the compromise instead of silently
  allowing a parallel shadow session.
- The `/auth/refresh` endpoint can safely emit a security alert on every `401` from a
  previously-valid (but now consumed) token.
- Rotating refresh tokens provide a sliding 7-day (`REFRESH_TTL = 604800`) active-user
  window: active users stay logged in indefinitely; inactive sessions expire naturally.

**Negative (accepted trade-off):**
- If the network drops after `consume_refresh()` but before the client receives the new
  tokens, the client is locked out and must re-authenticate with credentials.
- `AuthService.refresh()` is intentionally **not idempotent** — retrying with the same
  refresh token returns `401`.
- See [known-limitations.md](known-limitations.md) §consume_refresh is not idempotent.

---

## Alternatives considered

1. **Reusable refresh tokens with a revocation list** — rejected. The revocation list
   becomes a central single point of failure and requires synchronisation across replicas.
2. **Sliding-window refresh tokens** — rejected. They do not detect replay within the
   active window.
3. **Short-lived access tokens only, no refresh tokens** — rejected. Re-authentication
   friction is too high for mobile clients.
4. **Refresh token families** — group related tokens; reuse of any token in the family
   revokes all. Rejected for this iteration — adds stateful complexity with marginal
   benefit over simple single-use rotation.

---

## See also

- `TokenStore.consume_refresh()` in `src/auth/token_store.py`
- `AuthService.refresh()` in `src/auth/auth_service.py`
- API contract for `POST /auth/refresh` in `docs/api-contract.md`
- Network edge-case limitation in `docs/known-limitations.md`
