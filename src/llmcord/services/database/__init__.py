"""Database service package."""

from __future__ import annotations

from llmcord.services.database.core import DatabaseCore
from llmcord.services.database.messages import MessageDataMixin
from llmcord.services.database.users import UserPreferencesMixin


class AppDB(
    DatabaseCore,
    UserPreferencesMixin,
    MessageDataMixin,
):
    """SQLite-backed persistent application data storage.

    Combines functionality from:
    - DatabaseCore: Connection management
    - UserPreferencesMixin: User settings
    - MessageDataMixin: Message search data and responses
    """

    def __init__(
        self,
        local_db_path: str = "llmcord.db",
    ) -> None:
        """Initialize the database service.

        Note: actual connection setup and table initialization are async and
        are performed lazily on first use (or eagerly via `init_db`).
        """
        super().__init__(local_db_path)

    async def _init_db(self) -> None:
        """Initialize all database tables."""
        await self._init_user_tables()
        await self._init_message_tables()

    async def init(self) -> None:
        """Eagerly initialize the database connection and tables."""
        await self._ensure_initialized()


# Global instance initialized once to share the DB connection across services.
_db_state: dict[str, AppDB | None] = {"instance": None}


async def init_db(
    local_db_path: str = "llmcord.db",
) -> AppDB:
    """Initialize the global database instance."""
    instance = AppDB(
        local_db_path=local_db_path,
    )
    await instance.init()
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
]
