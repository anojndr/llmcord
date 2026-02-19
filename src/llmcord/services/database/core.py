"""Core database functionality for local SQLite persistence."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class DatabaseProtocol(Protocol):
    """Protocol for database connection management."""

    def _get_connection(self) -> sqlite3.Connection: ...

    def _run_db_call(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T: ...

    async def _run_db_call_async(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T: ...

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
        self._db_lock = threading.RLock()

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
            self._conn = sqlite3.connect(
                self.local_db_path,
                check_same_thread=False,
            )
            logger.info("Using local SQLite database at %s", self.local_db_path)
        return self._conn

    def _run_db_call(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Run a DB call while holding the database lock."""
        with self._db_lock:
            return func(*args, **kwargs)

    async def _run_db_call_async(
        self,
        func: Callable[_P, _T],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Run a DB call in a worker thread without blocking the event loop."""
        result = await asyncio.to_thread(self._run_db_call, func, *args, **kwargs)
        return cast("_T", result)

    def _sync(self) -> None:
        """No-op sync for local SQLite mode."""
