"""Database service package."""
from __future__ import annotations

from .bad_keys import BadKeysMixin, KeyRotator
from .core import DatabaseCore
from .messages import MessageDataMixin
from .users import UserPreferencesMixin
import libsql


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


# Global instance will be initialized with config values
_bad_keys_state: dict[str, BadKeysDB | None] = {"instance": None}


def init_bad_keys_db(
    db_url: str | None = None,
    auth_token: str | None = None,
) -> BadKeysDB:
    """Initialize the global bad keys database instance."""
    _bad_keys_state["instance"] = BadKeysDB(db_url=db_url, auth_token=auth_token)
    return _bad_keys_state["instance"]


def get_bad_keys_db() -> BadKeysDB:
    """Get the global bad keys database instance, initializing if needed."""
    if _bad_keys_state["instance"] is None:
        _bad_keys_state["instance"] = BadKeysDB()
    return _bad_keys_state["instance"]

__all__ = ["BadKeysDB", "KeyRotator", "init_bad_keys_db", "get_bad_keys_db"]
