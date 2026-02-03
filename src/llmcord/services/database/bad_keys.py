"""Bad API keys tracking and rotation logic."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator, Iterator

from llmcord.services.database.core import _with_reconnect

logger = logging.getLogger(__name__)


class BadKeysMixin:
    """Mixin for bad keys tracking."""

    def _init_bad_keys_tables(self) -> None:
        """Initialize bad keys tables."""
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
        conn.commit()

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
            (
                "INSERT OR REPLACE INTO bad_keys "
                "(provider, key_hash, error_message) VALUES (?, ?, ?)"
            ),
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
        """Check if an API key is bad for the main or decider provider.

        This ensures keys marked bad by the main model are also
        recognized by the decider model, and vice versa.
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

        This ensures a bad key is recognized by both the main model and the
        search decider.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"

        conn = self._get_connection()
        cursor = conn.cursor()
        key_hash = self._hash_key(api_key)

        # Mark for both base provider and decider provider
        cursor.execute(
            (
                "INSERT OR REPLACE INTO bad_keys "
                "(provider, key_hash, error_message) VALUES (?, ?, ?)"
            ),
            (base_provider, key_hash, error_message),
        )
        cursor.execute(
            (
                "INSERT OR REPLACE INTO bad_keys "
                "(provider, key_hash, error_message) VALUES (?, ?, ?)"
            ),
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
    def get_good_keys_synced(
        self,
        provider: str,
        all_keys: list[str],
    ) -> list[str]:
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
        return [
            key for key in all_keys if self._hash_key(key) not in bad_hashes
        ]

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


class KeyRotator:
    """Reusable key rotation mechanism with synced bad key tracking.

    This provides a consistent pattern for all services (main model, search
    decider, Tavily, and any future services) to handle API key rotation and
    retry logic.

    Usage:
        rotator = KeyRotator("my_provider", api_keys, db=get_bad_keys_db())

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
        db: BadKeysMixin,
        max_retries_multiplier: int = 2,
    ) -> None:
        """Initialize the key rotator.

        Args:
            provider: The provider name (e.g., "gemini", "openai", "tavily")
            all_keys: List of all API keys for this provider
            db: Database instance used for bad key tracking
            max_retries_multiplier: How many times to cycle through keys
                (default: 2)

        """
        self.provider = provider
        self.all_keys = all_keys.copy()
        self.max_retries_multiplier = max_retries_multiplier
        self._current_key = None
        self._attempt_count = 0
        self._good_keys = None
        self._db = db

    def _init_good_keys(self) -> None:
        """Initialize the good keys list from the database."""
        if self._good_keys is None:
            # Get good keys (filter out known bad ones - synced across
            # providers).
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

        Yields keys one at a time. If a key fails, call mark_current_bad()
        before the next iteration to mark it as bad and remove it from the
        rotation.
        """
        self._init_good_keys()
        max_attempts = len(self.all_keys) * self.max_retries_multiplier

        while self._good_keys and self._attempt_count < max_attempts:
            self._attempt_count += 1

            # Get the next good key to try
            key_index = (self._attempt_count - 1) % len(self._good_keys)
            self._current_key = self._good_keys[key_index]

            yield self._current_key

            # If we get here without mark_current_bad being called, the
            # key worked.
            # Reset for potential reuse.
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
        self._db.mark_key_bad_synced(
            self.provider,
            self._current_key,
            error_msg,
        )

        # Remove the bad key from good_keys list for this session
        if self._current_key in self._good_keys:
            self._good_keys.remove(self._current_key)

        # If all keys are exhausted, try resetting once
        if not self._good_keys and self._attempt_count >= len(self.all_keys):
            logger.warning(
                (
                    "All keys exhausted for '%s'. "
                    "Resetting synced keys for retry..."
                ),
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
