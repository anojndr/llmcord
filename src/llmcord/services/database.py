"""Turso (libSQL) based tracking of bad API keys to avoid wasting retries.

Uses Turso cloud database for persistent storage across deployments.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

import libsql

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")
LIBSQL_ERROR = getattr(libsql, "LibsqlError", getattr(libsql, "Error", Exception))


def _with_reconnect(
    method: Callable[Concatenate[BadKeysDB, P], T],
) -> Callable[Concatenate[BadKeysDB, P], T]:
    """Handle stale Turso connections by reconnecting and retrying."""
    @functools.wraps(method)
    def wrapper(self: BadKeysDB, *args: P.args, **kwargs: P.kwargs) -> T:
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
            self._reconnect()
            try:
                return method(self, *args, **kwargs)
            except (ValueError, LIBSQL_ERROR):
                logger.exception(
                    "Failed to reconnect to Turso after %d attempts",
                    max_retries,
                )
                raise
    return wrapper


class BadKeysDB:
    """Turso/libSQL-based tracking of bad API keys to avoid wasting retries."""

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
        self._init_db()

    def _reconnect(self) -> None:
        """Force reconnection to the database."""
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        self._get_connection()

    def _get_connection(self) -> libsql.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            if self.db_url and self.auth_token:
                # Connect to Turso cloud with local embedded replica
                self._conn = libsql.connect(
                    self.local_db_path,
                    sync_url=self.db_url,
                    auth_token=self.auth_token,
                )
                # Sync with remote on initial connection
                self._conn.sync()
                logger.info("Connected to Turso database: %s", self.db_url)
            else:
                # Fallback to local-only SQLite if no Turso credentials
                self._conn = libsql.connect(self.local_db_path)
                logger.warning(
                    "No Turso credentials found, using local SQLite database",
                )
        return self._conn

    def _init_db(self) -> None:
        """Initialize the database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bad_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, key_hash)
            )
        """)
        # User model preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_model_preferences (
                user_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # User search decider model preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_search_decider_preferences (
                user_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Message search data table stores web search results and extracted URL content.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_search_data (
                message_id TEXT PRIMARY KEY,
                search_results TEXT,
                tavily_metadata TEXT,
                lens_results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Response data table stores rendered response payloads for UI actions.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_response_data (
                message_id TEXT PRIMARY KEY,
                request_message_id TEXT,
                request_user_id TEXT,
                full_response TEXT,
                grounding_metadata TEXT,
                tavily_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: ensure message_response_data has the expected schema.
        with contextlib.suppress(LIBSQL_ERROR, ValueError):
            cursor.execute("PRAGMA table_info(message_response_data)")
            columns = {row[1] for row in cursor.fetchall()}

            required_columns = {
                "message_id": "TEXT",
                "request_message_id": "TEXT",
                "request_user_id": "TEXT",
                "full_response": "TEXT",
                "grounding_metadata": "TEXT",
                "tavily_metadata": "TEXT",
                "created_at": "TIMESTAMP",
            }

            if columns and "message_id" not in columns:
                cursor.execute("DROP TABLE message_response_data")
                cursor.execute("""
                    CREATE TABLE message_response_data (
                        message_id TEXT PRIMARY KEY,
                        request_message_id TEXT,
                        request_user_id TEXT,
                        full_response TEXT,
                        grounding_metadata TEXT,
                        tavily_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                columns = set(required_columns)

            for column_name, column_type in required_columns.items():
                if column_name not in columns:
                    cursor.execute(
                        "ALTER TABLE message_response_data ADD COLUMN "
                        f"{column_name} {column_type}",
                    )
            conn.commit()
        # Migration: Add lens_results column if it doesn't exist.
        with contextlib.suppress(LIBSQL_ERROR, ValueError):
            cursor.execute(
                "ALTER TABLE message_search_data ADD COLUMN lens_results TEXT",
            )
            conn.commit()
        conn.commit()
        self._sync()

    def _sync(self) -> None:
        """Sync changes with Turso cloud if connected to remote."""
        if self._conn and self.db_url and self.auth_token:
            try:
                self._conn.sync()
            except LIBSQL_ERROR as exc:
                logger.warning("Failed to sync with Turso: %s", exc)

    def _hash_key(self, api_key: str) -> str:
        """Create a hash of the API key to avoid storing sensitive data."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    @_with_reconnect
    def is_key_bad(self, provider: str, api_key: str) -> bool:
        """Check if an API key is marked as bad."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM bad_keys WHERE provider = ? AND key_hash = ?",
            (provider, self._hash_key(api_key)),
        )
        return cursor.fetchone() is not None

    @_with_reconnect
    def mark_key_bad(
        self,
        provider: str,
        api_key: str,
        error_message: str | None = None,
    ) -> None:
        """Mark an API key as bad."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message)"
            " VALUES (?, ?, ?)",
            (provider, self._hash_key(api_key), error_message),
        )
        conn.commit()
        self._sync()
        logger.info(
            "Marked API key as bad for provider '%s' (hash: %s...)",
            provider,
            self._hash_key(api_key)[:8],
        )

    def get_good_keys(self, provider: str, all_keys: list[str]) -> list[str]:
        """Filter out bad keys from a list of API keys."""
        return [key for key in all_keys if not self.is_key_bad(provider, key)]

    @_with_reconnect
    def is_key_bad_synced(self, provider: str, api_key: str) -> bool:
        """Check if an API key is marked as bad for the main or decider provider.

        This ensures keys marked bad by main model are also recognized by decider
        and vice versa.
        Uses a single optimized query instead of two separate calls.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"
        key_hash = self._hash_key(api_key)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM bad_keys WHERE (provider = ? OR provider = ?) "
            "AND key_hash = ? LIMIT 1",
            (base_provider, decider_provider, key_hash),
        )
        return cursor.fetchone() is not None

    @_with_reconnect
    def mark_key_bad_synced(
        self,
        provider: str,
        api_key: str,
        error_message: str | None = None,
    ) -> None:
        """Mark an API key as bad for both the main and decider providers.

        This ensures a bad key is recognized by both main model and search decider.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"

        conn = self._get_connection()
        cursor = conn.cursor()
        key_hash = self._hash_key(api_key)

        # Mark for both base provider and decider provider
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message)"
            " VALUES (?, ?, ?)",
            (base_provider, key_hash, error_message),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message)"
            " VALUES (?, ?, ?)",
            (decider_provider, key_hash, error_message),
        )
        conn.commit()
        self._sync()
        logger.info(
            "Marked API key as bad for '%s' and '%s' (hash: %s...)",
            base_provider,
            decider_provider,
            key_hash[:8],
        )

    @_with_reconnect
    def get_good_keys_synced(self, provider: str, all_keys: list[str]) -> list[str]:
        """Filter out bad keys, checking both main and decider providers.

        This ensures keys marked bad by either are filtered out for both.
        Uses a single bulk query for efficiency instead of checking each key
        individually.
        """
        if not all_keys:
            return []

        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"

        # Get all bad key hashes for both providers in a single query
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key_hash FROM bad_keys WHERE provider = ? OR provider = ?",
            (base_provider, decider_provider),
        )
        bad_hashes = {row[0] for row in cursor.fetchall()}

        # Filter keys locally using the pre-fetched bad hashes
        return [key for key in all_keys if self._hash_key(key) not in bad_hashes]

    @_with_reconnect
    def reset_provider_keys_synced(self, provider: str) -> None:
        """Reset all bad keys for both the main and decider providers."""
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM bad_keys WHERE provider = ? OR provider = ?",
            (base_provider, decider_provider),
        )
        conn.commit()
        self._sync()
        logger.info(
            "Reset all bad keys for '%s' and '%s'",
            base_provider,
            decider_provider,
        )

    @_with_reconnect
    def get_bad_key_count(self, provider: str) -> int:
        """Get the count of bad keys for a provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM bad_keys WHERE provider = ?",
            (provider,),
        )
        return cursor.fetchone()[0]

    @_with_reconnect
    def reset_provider_keys(self, provider: str) -> None:
        """Reset all bad keys for a specific provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys WHERE provider = ?", (provider,))
        conn.commit()
        self._sync()
        logger.info("Reset all bad keys for provider '%s'", provider)

    @_with_reconnect
    def reset_all(self) -> None:
        """Reset all bad keys for all providers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys")
        conn.commit()
        self._sync()
        logger.info("Reset all bad keys database")

    # User model preferences methods
    @_with_reconnect
    def get_user_model(self, user_id: str) -> str | None:
        """Get the preferred model for a user.

        Returns None if not set.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model FROM user_model_preferences WHERE user_id = ?",
            (str(user_id),),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    @_with_reconnect
    def set_user_model(self, user_id: str, model: str) -> None:
        """Set the preferred model for a user."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO user_model_preferences (user_id, model, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   model = ?,
                   updated_at = CURRENT_TIMESTAMP""",
            (str(user_id), model, model),
        )
        conn.commit()
        self._sync()
        logger.info("Set model preference for user %s: %s", user_id, model)

    # User search decider model preferences methods
    @_with_reconnect
    def get_user_search_decider_model(self, user_id: str) -> str | None:
        """Get the preferred search decider model for a user.

        Returns None if not set.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model FROM user_search_decider_preferences WHERE user_id = ?",
            (str(user_id),),
        )
        result = cursor.fetchone()
        return result[0] if result else None

    @_with_reconnect
    def set_user_search_decider_model(self, user_id: str, model: str) -> None:
        """Set the preferred search decider model for a user."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO user_search_decider_preferences (user_id, model, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   model = ?,
                   updated_at = CURRENT_TIMESTAMP""",
            (str(user_id), model, model),
        )
        conn.commit()
        self._sync()
        logger.info(
            "Set search decider model preference for user %s: %s",
            user_id,
            model,
        )

    @_with_reconnect
    def reset_all_user_model_preferences(self) -> int:
        """Reset all user model preferences.

        Returns the number of preferences deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_model_preferences")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM user_model_preferences")
        conn.commit()
        self._sync()
        logger.info("Reset all user model preferences (%d users affected)", count)
        return count

    @_with_reconnect
    def reset_all_user_search_decider_preferences(self) -> int:
        """Reset all user search decider model preferences.

        Returns the number of preferences deleted.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_search_decider_preferences")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM user_search_decider_preferences")
        conn.commit()
        self._sync()
        logger.info(
            "Reset all user search decider model preferences (%d users affected)",
            count,
        )
        return count

    # Message search data methods for persisting web search results in chat history
    @_with_reconnect
    def save_message_search_data(
        self,
        message_id: str,
        search_results: str | None = None,
        tavily_metadata: dict[str, Any] | None = None,
        lens_results: str | None = None,
    ) -> None:
        """Save web search results, lens results, and metadata for a Discord message.

        This allows search results to persist in chat history when conversations
        are rebuilt.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO message_search_data (
                   message_id,
                   search_results,
                   tavily_metadata,
                   lens_results
               )
               VALUES (?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET
                   search_results = COALESCE(?, search_results),
                   tavily_metadata = COALESCE(?, tavily_metadata),
                   lens_results = COALESCE(?, lens_results)""",
            (
                str(message_id),
                search_results,
                json.dumps(tavily_metadata) if tavily_metadata else None,
                lens_results,
                search_results,
                json.dumps(tavily_metadata) if tavily_metadata else None,
                lens_results,
            ),
        )
        conn.commit()
        # Sync in background to avoid blocking
        try:
            self._sync()
        except libsql.LibsqlError as exc:
            logger.debug("Background sync after save failed: %s", exc)
        logger.info("Saved search data for message %s", message_id)

    @_with_reconnect
    def get_message_search_data(
        self,
        message_id: str,
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """Get web search results, metadata, and lens results for a Discord message.

        Returns a tuple of (search_results, tavily_metadata, lens_results) or
        (None, None, None) if not found.
        Uses local replica for fast reads - sync happens periodically in background.
        """
        # Use local replica for fast reads (synced periodically)
        # Only sync if we haven't found data and need fresh data
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT search_results, tavily_metadata, lens_results "
            "FROM message_search_data WHERE message_id = ?",
            (str(message_id),),
        )
        result = cursor.fetchone()
        if result:
            search_results = result[0]
            try:
                tavily_metadata = json.loads(result[1]) if result[1] else None
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode tavily_metadata for message %s, returning None",
                    message_id,
                )
                tavily_metadata = None
            lens_results = result[2]
            logger.info(
                "Retrieved search data for message %s: search_results=%s, "
                "tavily_metadata=%s, lens_results=%s",
                message_id,
                bool(search_results),
                bool(tavily_metadata),
                bool(lens_results),
            )
            return search_results, tavily_metadata, lens_results
        return None, None, None

    @_with_reconnect
    def save_message_response_data(
        self,
        message_id: str,
        request_message_id: str,
        request_user_id: str,
        full_response: str | None = None,
        grounding_metadata: dict[str, Any] | list[Any] | None = None,
        tavily_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save response payloads for a Discord message.

        Stores the full response plus any source metadata so buttons can work
        after a bot restart.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO message_response_data (
                   message_id,
                   request_message_id,
                   request_user_id,
                   full_response,
                   grounding_metadata,
                   tavily_metadata
               )
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET
                   request_message_id = COALESCE(?, request_message_id),
                   request_user_id = COALESCE(?, request_user_id),
                   full_response = COALESCE(?, full_response),
                   grounding_metadata = COALESCE(?, grounding_metadata),
                   tavily_metadata = COALESCE(?, tavily_metadata)""",
            (
                str(message_id),
                str(request_message_id),
                str(request_user_id),
                full_response,
                json.dumps(grounding_metadata) if grounding_metadata else None,
                json.dumps(tavily_metadata) if tavily_metadata else None,
                str(request_message_id),
                str(request_user_id),
                full_response,
                json.dumps(grounding_metadata) if grounding_metadata else None,
                json.dumps(tavily_metadata) if tavily_metadata else None,
            ),
        )
        conn.commit()
        try:
            self._sync()
        except libsql.LibsqlError as exc:
            logger.debug("Background sync after save failed: %s", exc)
        logger.info("Saved response data for message %s", message_id)

    @_with_reconnect
    def get_message_response_data(
        self,
        message_id: str,
    ) -> tuple[
        str | None,
        dict[str, Any] | list[Any] | None,
        dict[str, Any] | None,
        str | None,
        str | None,
    ]:
        """Get response data for a Discord message.

        Returns (full_response, grounding_metadata, tavily_metadata,
        request_message_id, request_user_id) or (None, None, None, None, None).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT full_response, grounding_metadata, tavily_metadata, "
            "request_message_id, request_user_id "
            "FROM message_response_data WHERE message_id = ?",
            (str(message_id),),
        )
        result = cursor.fetchone()
        if not result:
            return None, None, None, None, None

        full_response = result[0]
        grounding_metadata_raw = result[1]
        tavily_metadata_raw = result[2]
        request_message_id = result[3]
        request_user_id = result[4]

        grounding_metadata = None
        tavily_metadata = None

        if grounding_metadata_raw:
            try:
                grounding_metadata = json.loads(grounding_metadata_raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode grounding_metadata for message %s",
                    message_id,
                )

        if tavily_metadata_raw:
            try:
                tavily_metadata = json.loads(tavily_metadata_raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode tavily_metadata for message %s",
                    message_id,
                )

        return (
            full_response,
            grounding_metadata,
            tavily_metadata,
            request_message_id,
            request_user_id,
        )


class KeyRotator:
    """Reusable key rotation mechanism with synced bad key tracking.

    This provides a consistent pattern for all services (main model, search decider,
    Tavily, and any future services) to handle API key rotation and retry logic.

    Usage:
        rotator = KeyRotator("my_provider", api_keys)

        # For async operations:
        async for key in rotator.get_keys_async():
            try:
                result = await my_api_call(key)
                break  # Success, exit the loop
            except Exception as e:
                rotator.mark_current_bad(str(e))

        # Or for sync operations:
        for key in rotator.get_keys():
            try:
                result = my_api_call(key)
                break  # Success
            except Exception as e:
                rotator.mark_current_bad(str(e))
    """

    def __init__(
        self,
        provider: str,
        all_keys: list[str],
        max_retries_multiplier: int = 2,
    ) -> None:
        """Initialize the key rotator.

        Args:
            provider: The provider name (e.g., "gemini", "openai", "tavily")
            all_keys: List of all API keys for this provider
            max_retries_multiplier: How many times to cycle through keys (default: 2)

        """
        self.provider = provider
        self.all_keys = all_keys.copy()
        self.max_retries_multiplier = max_retries_multiplier
        self._current_key = None
        self._attempt_count = 0
        self._good_keys = None
        self._db = get_bad_keys_db()

    def _init_good_keys(self) -> None:
        """Initialize the good keys list from the database."""
        if self._good_keys is None:
            # Get good keys (filter out known bad ones - synced across all providers)
            self._good_keys = self._db.get_good_keys_synced(
                self.provider,
                self.all_keys,
            )

            # If all keys are bad, reset and try again with all keys
            if not self._good_keys:
                logger.warning(
                    "All API keys for '%s' are marked as bad. Resetting...",
                    self.provider,
                )
                self._db.reset_provider_keys_synced(self.provider)
                self._good_keys = self.all_keys.copy()

    def get_keys(self) -> Iterator[str]:
        """Yield API keys to try, handling rotation and retries.

        Yields keys one at a time. If a key fails, call mark_current_bad() before
        the next iteration to mark it as bad and remove it from the rotation.
        """
        self._init_good_keys()
        max_attempts = len(self.all_keys) * self.max_retries_multiplier

        while self._good_keys and self._attempt_count < max_attempts:
            self._attempt_count += 1

            # Get the next good key to try
            key_index = (self._attempt_count - 1) % len(self._good_keys)
            self._current_key = self._good_keys[key_index]

            yield self._current_key

            # If we get here without mark_current_bad being called, the key worked
            # Reset for potential reuse
            self._current_key = None

    async def get_keys_async(self) -> AsyncIterator[str]:
        """Yield API keys asynchronously, handling rotation and retries.

        Same as get_keys() but for async contexts.
        """
        for key in self.get_keys():
            yield key

    def mark_current_bad(self, error_message: str | None = None) -> None:
        """Mark the current key as bad and remove it from the good keys list.

        Args:
            error_message: Optional error message describing why the key failed

        """
        if self._current_key is None:
            return

        error_msg = (error_message or "Unknown error")[:200]
        logger.warning(
            "API key failed for '%s' (attempt %d): %s",
            self.provider,
            self._attempt_count,
            error_msg,
        )

        # Mark the current key as bad (synced across providers)
        self._db.mark_key_bad_synced(self.provider, self._current_key, error_msg)

        # Remove the bad key from good_keys list for this session
        if self._current_key in self._good_keys:
            self._good_keys.remove(self._current_key)

        # If all keys are exhausted, try resetting once
        if not self._good_keys and self._attempt_count >= len(self.all_keys):
            logger.warning(
                "All keys exhausted for '%s'. Resetting synced keys for retry...",
                self.provider,
            )
            self._db.reset_provider_keys_synced(self.provider)
            self._good_keys = self.all_keys.copy()

        self._current_key = None

    def get_good_key_count(self) -> int:
        """Get the current count of good keys."""
        self._init_good_keys()
        return len(self._good_keys)

    def get_total_key_count(self) -> int:
        """Get the total count of all keys."""
        return len(self.all_keys)

    @property
    def has_keys(self) -> bool:
        """Check if there are any good keys available."""
        self._init_good_keys()
        return bool(self._good_keys)


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
