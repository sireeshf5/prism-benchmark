"""User dataclass with permission helper.

See docs/data-model.md for field semantics and relationship to AuthService.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet
from .permission import Permission, PermissionLevel


@dataclass
class User:
    """Authenticated user record returned by AuthService.validate().

    Attributes:
        user_id:     Unique identifier (UUID string). Used as the key in TokenStore.
        username:    Human-readable login name.
        email:       Verified email address.
        permissions: FrozenSet of Permission objects assigned at login time.
        is_active:   False means account is locked; AuthService.login() raises
                     ValueError and AuthService.refresh() returns None.

    See docs/data-model.md for the full field reference.
    """
    user_id:     str
    username:    str
    email:       str
    permissions: FrozenSet[Permission] = field(default_factory=frozenset)
    is_active:   bool = True

    def has_permission(self, name: str, level: PermissionLevel = PermissionLevel.READ) -> bool:
        """Return True if the user holds *name* at or above *level*.

        Iterates self.permissions looking for an exact name match whose
        Permission.allows(level) returns True.

        Called in two places:
          1. Flask route decorators in src/api/routes.py after AuthService.validate().
          2. The GET /auth/sessions handler checks
             g.user.has_permission("api.sessions.list").

        Note: No wildcard matching — "api.*" does NOT match "api.sessions.list".
        See docs/known-limitations.md §has_permission does not support wildcards.

        Args:
            name:  Exact permission name, e.g. "api.sessions.list".
            level: Minimum PermissionLevel required (default READ = 10).

        Returns:
            True if a matching Permission exists and allows *level*.
        """
        for perm in self.permissions:
            if perm.name == name and perm.allows(level):
                return True
        return False
