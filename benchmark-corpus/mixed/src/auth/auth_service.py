"""AuthService: high-level authentication orchestration.

Brings together TokenStore, SessionManager, and the User/Permission models.
The login/refresh/validate flow is documented in docs/auth-architecture.md.
The API contract (request/response shapes, error codes) is in docs/api-contract.md.

Dependencies:
    TokenStore     (token_store.py)     — token issuance, validation, consumption
    SessionManager (session_manager.py) — concurrent session tracking and eviction
    User           (models/user.py)     — carries the permission payload
"""
from __future__ import annotations
import secrets
from dataclasses import dataclass
from typing import Dict, Optional

from ..models.user import User
from .token_store import TokenStore, ACCESS_TTL
from .session_manager import SessionManager


@dataclass
class AuthResult:
    """Value object returned by AuthService.login() and AuthService.refresh().

    Matches the JSON shape documented in docs/api-contract.md:
        {"access_token": ..., "refresh_token": ..., "session_id": ..., "expires_in": 900}
    """
    access_token:  str
    refresh_token: str
    session_id:    str
    user_id:       str
    expires_in:    int   # always ACCESS_TTL = 900 seconds


class AuthService:
    """Orchestrates login, logout, token refresh, and token validation.

    All token lifetimes are owned by TokenStore constants:
        ACCESS_TTL  = 900s   (access token — 15 minutes)
        REFRESH_TTL = 604800s (refresh token — 7 days)

    All session caps are owned by SessionManager:
        MAX_SESSIONS = 5 concurrent sessions per user

    Public API
    ----------
    login(user)                        -> AuthResult
    logout(session_id, access_token)   -> bool
    refresh(refresh_token)             -> Optional[AuthResult]
    validate(access_token)             -> Optional[User]
    """

    def __init__(
        self,
        token_store:     TokenStore,
        session_manager: SessionManager,
        user_db:         Optional[Dict[str, User]] = None,
    ) -> None:
        self._tokens   = token_store
        self._sessions = session_manager
        self._user_db: Dict[str, User] = user_db or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self, user: User) -> AuthResult:
        """Authenticate *user* and issue a new access + refresh token pair.

        Steps:
        1. Verify user.is_active; raise ValueError if the account is locked.
        2. Call TokenStore.create_access_token() — 900s lifetime (ACCESS_TTL).
        3. Call TokenStore.create_refresh_token() — 604800s lifetime (REFRESH_TTL).
        4. Call SessionManager.create_session() — FIFO eviction if MAX_SESSIONS=5
           concurrent sessions already exist for this user.
        5. Cache the User object in _user_db for later validate() calls.
        6. Return AuthResult.

        The session_id is an independent random token; it is not derived from
        either the access or refresh token, allowing session revocation to be
        decoupled from token revocation.
        """
        if not user.is_active:
            raise ValueError(f"User {user.user_id!r} account is deactivated")

        access    = self._tokens.create_access_token(user.user_id)
        refresh   = self._tokens.create_refresh_token(user.user_id)
        session_id = secrets.token_urlsafe(16)
        self._sessions.create_session(user.user_id, session_id)
        self._user_db[user.user_id] = user

        return AuthResult(
            access_token=access,
            refresh_token=refresh,
            session_id=session_id,
            user_id=user.user_id,
            expires_in=ACCESS_TTL,
        )

    def logout(self, session_id: str, access_token: str) -> bool:
        """Invalidate the given session and expire the access token.

        The refresh token is NOT invalidated here intentionally — a client
        that retained the refresh token may exchange it once more. This is a
        known limitation; see docs/known-limitations.md
        §Logout does not invalidate refresh tokens.

        To fully revoke all tokens for a user call
        TokenStore.revoke_all_for_user(user_id) separately.
        """
        self._sessions.remove_session(session_id)
        # Expire the access token by removing it from the store
        user_id = self._tokens.validate_access_token(access_token)
        if user_id:
            h = self._tokens._hash(access_token)
            self._tokens._store.pop(f"access:{h}", None)
        return True

    def refresh(self, refresh_token: str) -> Optional[AuthResult]:
        """Rotate the refresh token and issue a fresh access + refresh pair.

        Calls TokenStore.consume_refresh() which enforces the single-use
        guarantee described in docs/adr-001-single-use-tokens.md:
        the old refresh token is atomically deleted before new tokens are
        issued. If the token was already consumed, returns None immediately.

        If the network drops after consume_refresh() succeeds but before the
        client receives the response, the user must re-authenticate with
        credentials. This is the known limitation documented in
        docs/known-limitations.md §consume_refresh is not idempotent.
        """
        user_id = self._tokens.consume_refresh(refresh_token)
        if user_id is None:
            return None  # already consumed, expired, or never issued

        user = self._user_db.get(user_id)
        if user is None or not user.is_active:
            return None

        access     = self._tokens.create_access_token(user_id)
        new_refresh = self._tokens.create_refresh_token(user_id)
        session_id  = secrets.token_urlsafe(16)
        self._sessions.create_session(user_id, session_id)

        return AuthResult(
            access_token=access,
            refresh_token=new_refresh,
            session_id=session_id,
            user_id=user_id,
            expires_in=ACCESS_TTL,
        )

    def validate(self, access_token: str) -> Optional[User]:
        """Return the full User object if access_token is valid, else None.

        Delegates to TokenStore.validate_access_token() to resolve the token
        to a user_id, then returns the cached User (including permissions) from
        _user_db. This two-step lookup keeps TokenStore free of permission logic.

        The returned User is used by Flask route decorators in src/api/routes.py
        to call User.has_permission() without a second DB round-trip.
        """
        user_id = self._tokens.validate_access_token(access_token)
        if user_id is None:
            return None
        return self._user_db.get(user_id)
