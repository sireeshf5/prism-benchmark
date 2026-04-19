"""Session manager with per-user FIFO eviction.

Constants
---------
MAX_SESSIONS = 5   # Maximum concurrent sessions per user

When a sixth session is created the oldest session (by creation time) is
silently evicted. This is FIFO eviction — the policy is documented in
docs/auth-architecture.md and its implications are tracked in
docs/known-limitations.md §FIFO session eviction is silent.
"""
from __future__ import annotations
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

# Authoritative cap — referenced in docs/auth-architecture.md and
# exposed to API clients via GET /auth/sessions as max_allowed.
MAX_SESSIONS: int = 5


@dataclass
class Session:
    """A single active authentication session.

    Attributes:
        session_id:  Opaque random string issued by AuthService.login().
        user_id:     Owner of the session.
        created_at:  Unix timestamp of session creation.
        last_seen:   Unix timestamp of most recent activity (updated externally).
        metadata:    Optional dict for device info, IP, user-agent, etc.
    """
    session_id: str
    user_id:    str
    created_at: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    metadata:   dict  = field(default_factory=dict)


class SessionManager:
    """Tracks active sessions and enforces MAX_SESSIONS = 5 per user.

    Internally uses an OrderedDict per user_id so FIFO eviction is O(1):
    popitem(last=False) removes the oldest inserted entry in constant time.

    Public API
    ----------
    create_session(user_id, session_id, metadata) -> Session
    get_session(session_id)                       -> Optional[Session]
    list_sessions(user_id)                        -> List[Session]
    remove_session(session_id)                    -> bool
    remove_all_for_user(user_id)                  -> int
    """

    def __init__(self) -> None:
        # _sessions[user_id] is an OrderedDict[session_id -> Session]
        self._sessions: Dict[str, OrderedDict[str, Session]] = defaultdict(OrderedDict)
        # flat index for O(1) lookup by session_id without scanning all users
        self._index: Dict[str, str] = {}  # session_id -> user_id

    def create_session(
        self,
        user_id:    str,
        session_id: str,
        metadata:   Optional[dict] = None,
    ) -> Session:
        """Create a new session, evicting the oldest if MAX_SESSIONS = 5 is reached.

        FIFO eviction: the session with the earliest created_at (the first
        entry in the OrderedDict) is removed silently. The evicted device will
        receive a 401 on its next API call.

        See docs/known-limitations.md §FIFO session eviction is silent for
        the planned improvement to notify evicted clients.
        """
        user_sessions = self._sessions[user_id]

        # Enforce cap — evict oldest (first inserted) entries until under limit
        while len(user_sessions) >= MAX_SESSIONS:
            evicted_id, _ = user_sessions.popitem(last=False)
            self._index.pop(evicted_id, None)

        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        user_sessions[session_id] = session
        self._index[session_id]   = user_id
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Look up a session by ID. Returns None if not found."""
        user_id = self._index.get(session_id)
        if user_id is None:
            return None
        return self._sessions[user_id].get(session_id)

    def list_sessions(self, user_id: str) -> List[Session]:
        """Return all active sessions for user_id, oldest first."""
        return list(self._sessions[user_id].values())

    def remove_session(self, session_id: str) -> bool:
        """Remove a session by ID. Returns True if found and removed."""
        user_id = self._index.pop(session_id, None)
        if user_id is None:
            return False
        self._sessions[user_id].pop(session_id, None)
        return True

    def remove_all_for_user(self, user_id: str) -> int:
        """Remove all sessions for user_id. Returns count removed."""
        user_sessions = self._sessions.pop(user_id, OrderedDict())
        for sid in user_sessions:
            self._index.pop(sid, None)
        return len(user_sessions)
