"""Core database functionality for Turso/libSQL."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Any, Concatenate, Protocol, cast

import libsql as libsql_module

from llmcord.core.error_handling import log_exception

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
libsql: Any = libsql_module
LibsqlConnection = Any

LIBSQL_ERROR = cast(
    "type[BaseException]",
    getattr(libsql, "LibsqlError", getattr(libsql, "Error", Exception)),
)


class DatabaseProtocol(Protocol):
    """Protocol for database connection management."""

    def _get_connection(self) -> LibsqlConnection: ...
    def _sync(self) -> None: ...


def _with_reconnect[T_Database: DatabaseProtocol, **P, T](
    method: Callable[Concatenate[T_Database, P], T],
) -> Callable[Concatenate[T_Database, P], T]:
    """Handle stale Turso connections by reconnecting and retrying."""

    @functools.wraps(method)
    def wrapper(self: T_Database, *args: P.args, **kwargs: P.kwargs) -> T:
        max_retries = 2
        try:
            return method(self, *args, **kwargs)
        except (ValueError, LIBSQL_ERROR) as exc:
            error_str = str(exc)
            # Check for Hrana stream errors (stale connection)
            if "stream not found" not in error_str and "Hrana" not in error_str:
                raise
            logger.warning(
                "Turso connection error, reconnecting (attempt %d): %s",
                1,
                exc,
            )
            # Assuming self has a _reconnect method
            if hasattr(self, "_reconnect"):
                self._reconnect()
            try:
                return method(self, *args, **kwargs)
            except (ValueError, LIBSQL_ERROR) as exc:
                log_exception(
                    logger=logger,
                    message="Failed to reconnect to Turso",
                    error=exc,
                    context={"max_retries": max_retries},
                )
                raise

    return wrapper


class DatabaseCore:
    """Core database connection management."""

    def __init__(
        self,
        db_url: str | None = None,
        auth_token: str | None = None,
        local_db_path: str = "bad_keys.db",
    ) -> None:
        """Initialize the Turso database connection.

        Args:
            db_url: Turso database URL (e.g., libsql://your-db.turso.io)
            auth_token: Turso authentication token
            local_db_path: Local path for embedded replica (for offline reads)

        """
        self.db_url = db_url or os.getenv("TURSO_DATABASE_URL")
        self.auth_token = auth_token or os.getenv("TURSO_AUTH_TOKEN")
        self.local_db_path = local_db_path
        self._conn = None
        # Subclasses should call _init_db() or specific init methods

    def _reconnect(self) -> None:
        """Force reconnection to the database."""
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        self._get_connection()

    def _fallback_to_local(self, reason: BaseException) -> None:
        """Switch to local-only SQLite mode after Turso failures."""
        logger.warning(
            "Falling back to local SQLite database at %s due to Turso error: %s",
            self.local_db_path,
            reason,
        )
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
        self.db_url = None
        self.auth_token = None
        self._conn = libsql.connect(self.local_db_path)

    def _get_connection(self) -> LibsqlConnection:
        """Get or create the database connection."""
        if self._conn is None:
            if self.db_url and self.auth_token:
                # Connect to Turso cloud with local embedded replica
                try:
                    self._conn = libsql.connect(
                        self.local_db_path,
                        sync_url=self.db_url,
                        auth_token=self.auth_token,
                    )
                    # Sync with remote on initial connection
                    if self._conn is not None:
                        self._conn.sync()
                    logger.info("Connected to Turso database: %s", self.db_url)
                except (ValueError, LIBSQL_ERROR) as exc:
                    self._fallback_to_local(exc)
            else:
                # Fallback to local-only SQLite if no Turso credentials
                self._conn = libsql.connect(self.local_db_path)
                logger.warning(
                    "No Turso credentials found, using local SQLite database",
                )
        return self._conn

    def _sync(self) -> None:
        """Sync changes with Turso cloud if connected to remote."""
        if self._conn is not None and self.db_url and self.auth_token:
            try:
                self._conn.sync()
            except (ValueError, LIBSQL_ERROR) as exc:
                self._fallback_to_local(exc)
