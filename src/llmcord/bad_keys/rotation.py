"""Key rotation and global database access for bad key tracking."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .db import BadKeysDB

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


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
