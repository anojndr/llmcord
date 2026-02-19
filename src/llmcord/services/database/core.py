"""Core database functionality for local SQLite persistence."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseProtocol(Protocol):
    """Protocol for database connection management."""

    def _get_connection(self) -> sqlite3.Connection: ...
    def _sync(self) -> None: ...


class DatabaseCore:
    """Core database connection management."""

    def __init__(
        self,
        local_db_path: str | Path = "llmcord.db",
    ) -> None:
        """Initialize local SQLite database connection settings."""
        self.local_db_path = str(local_db_path)
        self._conn: sqlite3.Connection | None = None

    def _reconnect(self) -> None:
        """Force reconnection to the database."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                logger.debug("Failed to close SQLite connection cleanly: %s", exc)
            finally:
                self._conn = None
        self._get_connection()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.local_db_path)
            logger.info("Using local SQLite database at %s", self.local_db_path)
        return self._conn

    def _sync(self) -> None:
        """No-op sync for local SQLite mode."""
