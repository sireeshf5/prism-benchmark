# Data Model Reference

Describes the `User` and `Permission` types that flow through the auth system.

---

## User (`src/models/user.py`)

`User` is the core identity object returned by `AuthService.validate()`.

```python
@dataclass
class User:
    user_id:     str
    username:    str
    email:       str
    permissions: FrozenSet[Permission]
    is_active:   bool
```

### Field reference

| Field         | Type                | Notes                                                          |
|---------------|---------------------|----------------------------------------------------------------|
| `user_id`     | `str` (UUID)        | Immutable primary key. Used as the key stored in `TokenStore`. |
| `is_active`   | `bool`              | `False` blocks `AuthService.login()` and `AuthService.refresh()`. |
| `permissions` | `FrozenSet[Permission]` | Immutable at construction. Assign a new `User` to change. |

### `User.has_permission(name, level)`

```python
def has_permission(self, name: str, level: PermissionLevel = PermissionLevel.READ) -> bool:
```

Iterates `self.permissions` for an exact `name` match whose `level >= required_level`.
Returns `True` on first match.

Called in two places:
1. After `AuthService.validate()` returns a `User`, Flask route decorators in
   `src/api/routes.py` call `g.user.has_permission()`.
2. The `GET /auth/sessions` route handler explicitly checks
   `g.user.has_permission("api.sessions.list")` before listing sessions.

**Known limitation:** No wildcard matching. A permission named `"api.*"` does NOT match
`"api.sessions.list"`. See [known-limitations.md](known-limitations.md)
§has_permission does not support wildcards.

---

## Permission (`src/models/permission.py`)

```python
@dataclass(frozen=True)
class Permission:
    name:  str              # e.g. "api.sessions.list"
    level: PermissionLevel
```

Frozen dataclass — instances are hashable and safe to store in `FrozenSet`.

### PermissionLevel enum

```python
class PermissionLevel(IntEnum):
    READ  = 10
    WRITE = 20
    ADMIN = 30
    SUPER = 40
```

Uses `IntEnum` so levels are comparable with `>=`. `Permission.allows(required_level)`
returns `self.level >= required_level`.

Example: a permission with `level=ADMIN (30)` satisfies a `WRITE (20)` check because
`30 >= 20` is `True`.

---

## Relationship map

```
TokenStore
  stores  -> user_id (str only — no User object)

AuthService
  _user_db -> Dict[user_id, User]     (populated on login / refresh)
  validate() -> resolves token -> user_id (via TokenStore) -> User (via _user_db)

User
  .has_permission(name, level) -> consults FrozenSet[Permission]

Permission
  .allows(required_level) -> self.level >= required_level
```

`TokenStore` is intentionally free of any User or Permission knowledge.
The two-step lookup (`token → user_id → User`) keeps the token layer
stateless with respect to application-level authorisation.
