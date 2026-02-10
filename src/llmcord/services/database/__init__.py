"""Database service package."""

from __future__ import annotations

from llmcord.services.database.bad_keys import BadKeysMixin, KeyRotator
from llmcord.services.database.core import DatabaseCore
from llmcord.services.database.messages import MessageDataMixin
from llmcord.services.database.users import UserPreferencesMixin


class LibsqlUnavailableError(RuntimeError):
    """libsql is not available."""


try:
    import libsql as _libsql
except ImportError:

    class _LibsqlStub:
        """Fallback stub for libsql when the dependency is unavailable."""

        def connect(self, *_args: object, **_kwargs: object) -> None:
            raise LibsqlUnavailableError

    _libsql = _LibsqlStub()

libsql = _libsql


class BadKeysDB(
    DatabaseCore,
    BadKeysMixin,
    UserPreferencesMixin,
    MessageDataMixin,
):
    """Turso/libSQL-based tracking of bad API keys and persistent storage.

    Combines functionality from:
    - DatabaseCore: Connection management
    - BadKeysMixin: API key tracking
    - UserPreferencesMixin: User settings
    - MessageDataMixin: Message search data and responses
    """

    def __init__(
        self,
        db_url: str | None = None,
        auth_token: str | None = None,
        local_db_path: str = "bad_keys.db",
    ) -> None:
        """Initialize the Turso database connection and tables."""
        super().__init__(db_url, auth_token, local_db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize all database tables."""
        self._init_bad_keys_tables()
        self._init_user_tables()
        self._init_message_tables()
        self._sync()


# Global instance initialized once to share the DB connection across services.
_bad_keys_state: dict[str, BadKeysDB | None] = {"instance": None}


def init_bad_keys_db(
    db_url: str | None = None,
    auth_token: str | None = None,
) -> BadKeysDB:
    """Initialize the global bad keys database instance."""
    instance = BadKeysDB(
        db_url=db_url,
        auth_token=auth_token,
    )
    _bad_keys_state["instance"] = instance
    return instance


def get_bad_keys_db() -> BadKeysDB:
    """Get the global bad keys database instance, initializing if needed."""
    instance = _bad_keys_state["instance"]
    if instance is None:
        instance = BadKeysDB()
        _bad_keys_state["instance"] = instance
    return instance


__all__ = [
    "BadKeysDB",
    "KeyRotator",
    "get_bad_keys_db",
    "init_bad_keys_db",
    "libsql",
]
