"""Core database functionality for local SQLite persistence."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, cast

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine
    from pathlib import Path

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class DatabaseProtocol(Protocol):
    """Protocol for database connection management."""

    async def _get_connection(self) -> aiosqlite.Connection: ...

    async def _ensure_initialized(self) -> None: ...

    def _run_db_call(
        self,
        func: Callable[_P, Awaitable[_T]],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T: ...

    async def _run_db_call_async(
        self,
        func: Callable[_P, Awaitable[_T]],
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
        self._connections: dict[int, aiosqlite.Connection] = {}
        self._connect_locks: dict[int, asyncio.Lock] = {}
        self._init_locks: dict[int, asyncio.Lock] = {}
        self._initialized_loops: set[int] = set()

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create the aiosqlite connection for the current event loop."""
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        if loop_id in self._connections:
            return self._connections[loop_id]

        connect_lock = self._connect_locks.get(loop_id)
        if connect_lock is None:
            connect_lock = asyncio.Lock()
            self._connect_locks[loop_id] = connect_lock

        async with connect_lock:
            if loop_id in self._connections:
                return self._connections[loop_id]

            conn = await aiosqlite.connect(self.local_db_path)
            await conn.execute("PRAGMA foreign_keys = ON")
            self._connections[loop_id] = conn
            logger.info("Using local SQLite database at %s", self.local_db_path)
            return conn

    async def _ensure_initialized(self) -> None:
        """Ensure database tables/migrations have run for this event loop."""
        loop_id = id(asyncio.get_running_loop())
        if loop_id in self._initialized_loops:
            return

        init_lock = self._init_locks.get(loop_id)
        if init_lock is None:
            init_lock = asyncio.Lock()
            self._init_locks[loop_id] = init_lock

        async with init_lock:
            if loop_id in self._initialized_loops:
                return
            init_db = getattr(self, "_init_db", None)
            if callable(init_db):
                await init_db()
            self._initialized_loops.add(loop_id)

    def _run_db_call(
        self,
        func: Callable[_P, Awaitable[_T]],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Run an async DB call from sync code.

        Prefer async methods. This is primarily for compatibility in test fakes
        and any sync call sites.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                cast(
                    "Coroutine[object, object, _T]",
                    func(*args, **kwargs),
                ),
            )
        msg = "Synchronous DB methods cannot be called from a running event loop"
        raise RuntimeError(msg)

    async def _run_db_call_async(
        self,
        func: Callable[_P, Awaitable[_T]],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        """Run a DB call without blocking the event loop."""
        await self._ensure_initialized()
        return await func(*args, **kwargs)

    def _sync(self) -> None:
        """No-op sync for local SQLite mode."""

    async def aclose(self) -> None:
        """Close any open aiosqlite connections."""
        for conn in list(self._connections.values()):
            try:
                await conn.close()
            except sqlite3.Error as exc:
                logger.debug("Failed to close SQLite connection cleanly: %s", exc)
        self._connections.clear()
