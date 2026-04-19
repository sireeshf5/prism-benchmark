"""Flask blueprint: /auth/login /auth/logout /auth/refresh /auth/sessions.

All routes delegate to AuthService. Token lifetimes and session limits are
owned by TokenStore (ACCESS_TTL=900, REFRESH_TTL=604800) and SessionManager
(MAX_SESSIONS=5) — they are not hardcoded here.

Full endpoint contracts (request/response schemas, guarantees, error codes):
    docs/api-contract.md

Known limitations affecting these routes:
    docs/known-limitations.md
"""
from __future__ import annotations
from functools import wraps
from typing import Callable

from flask import Blueprint, g, jsonify, request, current_app

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def require_auth(f: Callable) -> Callable:
    """Decorator that validates the Bearer token and injects g.user.

    Calls AuthService.validate() to resolve the token to a User object.
    Sets g.user so that route handlers can call g.user.has_permission()
    for fine-grained access control without a second token lookup.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "missing_token"}), 401
        token        = header[7:]
        auth_service = current_app.extensions["auth_service"]
        user         = auth_service.validate(token)
        if user is None:
            return jsonify({"error": "invalid_token"}), 401
        g.user = user
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["POST"])
def login():
    """Exchange credentials for an access + refresh token pair.

    Request (JSON):
        {"username": "...", "password": "..."}

    Success 200:
        {"access_token": "...", "refresh_token": "...", "session_id": "...", "expires_in": 900}

    expires_in is always ACCESS_TTL = 900 (15 minutes).
    Concurrent session cap is MAX_SESSIONS = 5; oldest is evicted if reached.

    Errors:
        401 {"error": "invalid_credentials"}
        403 {"error": "account_locked"}

    See docs/api-contract.md §POST /auth/login for full contract.
    """
    body     = request.get_json(force=True) or {}
    username = body.get("username", "")
    password = body.get("password", "")

    auth_service = current_app.extensions["auth_service"]
    user_db      = current_app.extensions["user_db"]

    user = user_db.find_by_username(username)
    if user is None or not user_db.check_password(username, password):
        return jsonify({"error": "invalid_credentials"}), 401
    if not user.is_active:
        return jsonify({"error": "account_locked"}), 403

    result = auth_service.login(user)
    return jsonify({
        "access_token":  result.access_token,
        "refresh_token": result.refresh_token,
        "session_id":    result.session_id,
        "expires_in":    result.expires_in,   # 900
    }), 200


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------

@auth_bp.route("/logout", methods=["POST"])
@require_auth
def logout():
    """Invalidate current session and access token.

    Requires: Authorization: Bearer <access_token>

    Request (JSON):
        {"session_id": "..."}

    Success 200:
        {"ok": true}

    Warning: The refresh token is NOT invalidated — see docs/known-limitations.md
    §Logout does not invalidate refresh tokens.
    """
    body       = request.get_json(force=True) or {}
    session_id = body.get("session_id", "")
    token      = request.headers["Authorization"][7:]

    auth_service = current_app.extensions["auth_service"]
    auth_service.logout(session_id, token)
    return jsonify({"ok": True}), 200


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------

@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """Exchange a refresh token for a new access + refresh pair.

    Single-use: TokenStore.consume_refresh() atomically deletes the submitted
    refresh token on first call. Any retry with the same token returns 401.
    See docs/adr-001-single-use-tokens.md for the full rationale.

    Request (JSON):
        {"refresh_token": "..."}

    Success 200 (same shape as /login response):
        {"access_token": "...", "refresh_token": "...", "session_id": "...", "expires_in": 900}

    Errors:
        401 {"error": "invalid_or_expired_refresh_token"}

    See docs/api-contract.md §POST /auth/refresh for guarantees.
    """
    body          = request.get_json(force=True) or {}
    refresh_token = body.get("refresh_token", "")

    auth_service = current_app.extensions["auth_service"]
    result       = auth_service.refresh(refresh_token)
    if result is None:
        return jsonify({"error": "invalid_or_expired_refresh_token"}), 401
    return jsonify({
        "access_token":  result.access_token,
        "refresh_token": result.refresh_token,
        "session_id":    result.session_id,
        "expires_in":    result.expires_in,
    }), 200


# ---------------------------------------------------------------------------
# GET /auth/sessions
# ---------------------------------------------------------------------------

@auth_bp.route("/sessions", methods=["GET"])
@require_auth
def list_sessions():
    """List active sessions for the authenticated user.

    Requires: Authorization: Bearer <access_token>
    Permission: g.user.has_permission("api.sessions.list") must be True.

    Success 200:
        {
          "sessions": [{"session_id": "...", "created_at": 1234, "last_seen": 1234}],
          "count": <int>,
          "max_allowed": 5
        }

    max_allowed is always MAX_SESSIONS = 5.

    See docs/api-contract.md §GET /auth/sessions.
    """
    from ..auth.session_manager import MAX_SESSIONS

    if not g.user.has_permission("api.sessions.list"):
        return jsonify({"error": "forbidden"}), 403

    session_manager = current_app.extensions["session_manager"]
    sessions        = session_manager.list_sessions(g.user.user_id)

    return jsonify({
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": int(s.created_at),
                "last_seen":  int(s.last_seen),
            }
            for s in sessions
        ],
        "count":       len(sessions),
        "max_allowed": MAX_SESSIONS,
    }), 200
