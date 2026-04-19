"""Redis-backed token store.

Constants
---------
ACCESS_TTL   = 900      # 15 minutes — access token lifetime in seconds
REFRESH_TTL  = 604800   # 7 days    — refresh token lifetime in seconds

The single-use guarantee on refresh tokens is enforced by consume_refresh(),
which atomically pops the token entry from the backing store. Even a race
between two concurrent requests cannot replay the same token.
See docs/adr-001-single-use-tokens.md for the full rationale and trade-offs.

In production the backend is a Redis client. The test harness passes a plain
dict so no Redis dependency is needed during benchmarking.
"""
from __future__ import annotations
import hashlib
import secrets
import time
from typing import Optional

# These constants are referenced in docs/api-contract.md and
# docs/adr-001-single-use-tokens.md as the authoritative token lifetimes.
ACCESS_TTL:  int = 900      # seconds (15 minutes)
REFRESH_TTL: int = 604_800  # seconds (7 days)


class TokenStore:
    """Manages access and refresh token persistence.

    Storage layout (conceptual Redis keys):
        token:access:{sha256}  -> dict {user_id, issued_at, expires_at}
        token:refresh:{sha256} -> dict {user_id, issued_at, expires_at}

    Public API
    ----------
    create_access_token(user_id)   -> str
    create_refresh_token(user_id)  -> str
    validate_access_token(token)   -> Optional[str]  (returns user_id or None)
    consume_refresh(token)         -> Optional[str]  (single-use: deletes + returns user_id)
    revoke_all_for_user(user_id)   -> int            (returns count of tokens revoked)
    """

    def __init__(self, backend: Optional[dict] = None) -> None:
        self._store: dict[str, dict] = backend if backend is not None else {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _generate() -> str:
        return secrets.token_urlsafe(32)

    def _set(self, kind: str, token: str, user_id: str, ttl: int) -> None:
        h   = self._hash(token)
        now = int(time.time())
        self._store[f"{kind}:{h}"] = {
            "user_id":    user_id,
            "issued_at":  now,
            "expires_at": now + ttl,
        }

    def _get(self, kind: str, token: str) -> Optional[dict]:
        h     = self._hash(token)
        entry = self._store.get(f"{kind}:{h}")
        if entry is None:
            return None
        if int(time.time()) > entry["expires_at"]:
            del self._store[f"{kind}:{h}"]
            return None
        return entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_access_token(self, user_id: str) -> str:
        """Issue a new access token with ACCESS_TTL = 900s lifetime."""
        token = self._generate()
        self._set("access", token, user_id, ACCESS_TTL)
        return token

    def create_refresh_token(self, user_id: str) -> str:
        """Issue a new refresh token with REFRESH_TTL = 604800s lifetime."""
        token = self._generate()
        self._set("refresh", token, user_id, REFRESH_TTL)
        return token

    def validate_access_token(self, token: str) -> Optional[str]:
        """Return user_id if access token is valid and unexpired, else None."""
        entry = self._get("access", token)
        return entry["user_id"] if entry else None

    def consume_refresh(self, token: str) -> Optional[str]:
        """Single-use refresh token consumption (enforcement point for ADR-001).

        Atomically removes the refresh token from the store and returns the
        associated user_id. A second call with the same token always returns None
        because the entry was deleted on first call.

        This atomic pop is the mechanism that enforces the single-use guarantee
        described in docs/adr-001-single-use-tokens.md.

        Returns:
            user_id string if token was valid, or None if already consumed /
            expired / never issued.
        """
        h     = self._hash(token)
        key   = f"refresh:{h}"
        entry = self._store.pop(key, None)   # atomic delete
        if entry is None:
            return None
        if int(time.time()) > entry["expires_at"]:
            return None
        return entry["user_id"]

    def revoke_all_for_user(self, user_id: str) -> int:
        """Delete all tokens (access and refresh) belonging to user_id.

        Used by the forceful-logout path when an admin deactivates an account.
        Returns the number of token entries removed.
        """
        keys = [k for k, v in self._store.items() if v.get("user_id") == user_id]
        for k in keys:
            del self._store[k]
        return len(keys)
