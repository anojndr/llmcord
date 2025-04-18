# llm-discordbot/api_key_manager.py
import random
import logging
from typing import List, Optional, Dict, Set
import asyncio

from rate_limit_manager import RateLimitManager

class ApiKeyManager:
    """Manages API keys for a specific provider, including rotation and rate limit awareness."""

    def __init__(self, provider_name: str, api_keys: List[str]):
        if not isinstance(api_keys, list): # Allow empty list for local models
             if api_keys is None:
                 api_keys = []
             else:
                 # If it's not a list and not None, it's likely a single key string
                 api_keys = [str(api_keys)] # Convert single key to list

        self.provider_name = provider_name
        self.all_keys: Set[str] = set(api_keys) # Use a set for efficient lookups
        self.rate_limit_manager = RateLimitManager(provider_name)
        self._lock = asyncio.Lock() # Lock for managing key selection/updates

        if not self.all_keys:
             logging.warning(f"ApiKeyManager initialized for '{provider_name}' with an empty key list.")

        logging.info(f"ApiKeyManager initialized for '{provider_name}' with {len(self.all_keys)} key(s).")

    async def get_valid_key(self) -> Optional[str]:
        """
        Gets a random, valid (not currently rate-limited) API key.
        Resets the rate limit database for this provider if all keys are limited.
        Returns None if no valid keys are available after checking/resetting,
        or if no keys were configured initially for a non-local provider.
        """
        async with self._lock:
            # Handle case where no keys were ever configured
            if not self.all_keys:
                # Check if it's a known local provider (can operate without keys)
                # Add more local provider names if needed
                if self.provider_name in ["ollama", "lmstudio", "vllm", "oobabooga", "jan"]:
                    logging.debug(f"No keys configured for local provider '{self.provider_name}', returning None (expected).")
                    return None # Local models often don't need keys
                else:
                    logging.error(f"No API keys configured for provider '{self.provider_name}'. Cannot get a key.")
                    return None

            valid_keys = await self.rate_limit_manager.get_valid_keys(list(self.all_keys))

            if not valid_keys:
                logging.warning(f"All {len(self.all_keys)} API keys for provider '{self.provider_name}' are currently rate-limited.")

                # Reset the database since all configured keys are blocked
                logging.info(f"Resetting rate limit database for provider '{self.provider_name}' as all keys were blocked.")
                await self.rate_limit_manager.reset_database()
                # After reset, all keys are considered valid again for the next attempt
                valid_keys = list(self.all_keys)
                if not valid_keys: # Should not happen if self.all_keys is not empty, but safety check
                     logging.error(f"Something went wrong. No valid keys found even after reset for '{self.provider_name}'.")
                     return None

            # Select a random key from the valid ones
            selected_key = random.choice(valid_keys)
            logging.debug(f"Selected key for {self.provider_name}: ...{selected_key[-4:]}")
            return selected_key

    async def mark_rate_limited(self, api_key: str):
        """Marks a specific API key as rate-limited."""
        if api_key in self.all_keys:
             async with self._lock:
                 await self.rate_limit_manager.add_rate_limited_key(api_key)
                 logging.warning(f"Marked key ...{api_key[-4:]} for provider '{self.provider_name}' as rate-limited.")
        else:
             logging.warning(f"Attempted to mark unknown key ...{api_key[-4:]} as rate-limited for provider '{self.provider_name}'. Key not in configured list.")

    def get_total_keys(self) -> int:
        """Returns the total number of keys configured for this provider."""
        return len(self.all_keys)
