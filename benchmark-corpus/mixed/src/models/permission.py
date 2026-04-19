"""Permission model: PermissionLevel enum and Permission dataclass."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum


class PermissionLevel(IntEnum):
    """Numeric permission levels. Higher value = more access."""
    READ  = 10
    WRITE = 20
    ADMIN = 30
    SUPER = 40


@dataclass(frozen=True)
class Permission:
    """A single named permission granted to a user.

    Attributes:
        name:  Dot-namespaced permission string, e.g. "api.tokens.revoke".
        level: Minimum PermissionLevel required.

    See docs/data-model.md for full field semantics and examples.
    """
    name:  str
    level: PermissionLevel

    def allows(self, required_level: PermissionLevel) -> bool:
        """Return True if this permission satisfies *required_level*.

        Uses integer comparison because PermissionLevel is an IntEnum:
        ADMIN (30) >= WRITE (20) >= READ (10).
        """
        return self.level >= required_level
