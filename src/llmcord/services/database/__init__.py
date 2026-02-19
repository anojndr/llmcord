"""Database service package."""

from __future__ import annotations

import importlib
from typing import Any

from llmcord.services.database.core import DatabaseCore
from llmcord.services.database.messages import MessageDataMixin
from llmcord.services.database.users import UserPreferencesMixin


class LibsqlUnavailableError(RuntimeError):
    """libsql is not available."""


libsql_module: Any
try:
    libsql_module = importlib.import_module("libsql")
except ImportError:

    class _LibsqlStub:
        """Fallback stub for libsql when the dependency is unavailable."""

        def connect(self, *_args: object, **_kwargs: object) -> None:
            raise LibsqlUnavailableError

    libsql_module = _LibsqlStub()

libsql: Any = libsql_module


class AppDB(
    DatabaseCore,
    UserPreferencesMixin,
    MessageDataMixin,
):
    """Turso/libSQL-based persistent application data storage.

    Combines functionality from:
    - DatabaseCore: Connection management
    - UserPreferencesMixin: User settings
    - MessageDataMixin: Message search data and responses
    """

    def __init__(
        self,
        db_url: str | None = None,
        auth_token: str | None = None,
        local_db_path: str = "llmcord.db",
    ) -> None:
        """Initialize the Turso database connection and tables."""
        super().__init__(db_url, auth_token, local_db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize all database tables."""
        self._init_user_tables()
        self._init_message_tables()
        self._sync()


# Global instance initialized once to share the DB connection across services.
_db_state: dict[str, AppDB | None] = {"instance": None}


def init_db(
    db_url: str | None = None,
    auth_token: str | None = None,
) -> AppDB:
    """Initialize the global database instance."""
    instance = AppDB(
        db_url=db_url,
        auth_token=auth_token,
    )
    _db_state["instance"] = instance
    return instance


def get_db() -> AppDB:
    """Get the global database instance, initializing if needed."""
    instance = _db_state["instance"]
    if instance is None:
        instance = AppDB()
        _db_state["instance"] = instance
    return instance


__all__ = [
    "AppDB",
    "get_db",
    "init_db",
    "libsql",
]
