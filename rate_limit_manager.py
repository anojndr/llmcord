# llm-discordbot/rate_limit_manager.py
import sqlite3
import time
import os
import logging
from typing import List, Set
import asyncio
from datetime import datetime, timedelta

DATABASE_DIR = "ratelimit_db"
COOLDOWN_PERIOD_HOURS = 24

class RateLimitManager:
    """Manages rate-limited API keys using a persistent SQLite database."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Sanitize provider name for filesystem compatibility
        safe_provider_name = "".join(c if c.isalnum() else "_" for c in provider_name)
        self.db_path = os.path.join(DATABASE_DIR, f"{safe_provider_name}_ratelimit.db")
        self._lock = asyncio.Lock() # Lock for database operations

        # Ensure the database directory exists
        os.makedirs(DATABASE_DIR, exist_ok=True)

        # Initialize database and cleanup expired keys on startup
        # Use asyncio.ensure_future for compatibility if needed, but create_task is generally preferred
        asyncio.create_task(self._initialize_db())


    async def _get_db_connection(self):
        """Establishes and returns a database connection."""
        try:
            # Use check_same_thread=False for async usage, manage concurrency with asyncio.Lock
            # Set timeout to handle potential database locking issues
            conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row # Return rows as dict-like objects
            # Improve performance and reliability with WAL mode
            conn.execute("PRAGMA journal_mode=WAL;")
            return conn
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database {self.db_path}: {e}", exc_info=True)
            raise # Re-raise the exception

    async def _initialize_db(self):
        """Initializes the database table if it doesn't exist and cleans up expired keys."""
        async with self._lock:
            conn = None # Initialize conn to None
            try:
                conn = await self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rate_limited_keys (
                        api_key TEXT PRIMARY KEY,
                        rate_limited_at REAL NOT NULL
                    )
                ''')
                conn.commit()
                logging.info(f"Database initialized for provider '{self.provider_name}' at {self.db_path}")
                # Clean up expired keys on startup
                await self._cleanup_expired_keys(conn)
            except sqlite3.Error as e:
                logging.error(f"Failed to initialize or cleanup database for {self.provider_name}: {e}", exc_info=True)
            finally:
                if conn:
                    conn.close()

    async def _cleanup_expired_keys(self, conn):
        """Removes keys whose cooldown period has expired. Assumes lock is held."""
        try:
            cooldown_seconds = COOLDOWN_PERIOD_HOURS * 3600
            expiration_time = time.time() - cooldown_seconds
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rate_limited_keys WHERE rate_limited_at < ?", (expiration_time,))
            deleted_count = cursor.rowcount
            conn.commit()
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} expired rate-limited keys for provider '{self.provider_name}'.")
        except sqlite3.Error as e:
            logging.error(f"Error cleaning up expired keys for {self.provider_name}: {e}", exc_info=True)
            # Don't rollback here, as other operations might be pending

    async def add_rate_limited_key(self, api_key: str):
        """Adds or updates a key in the rate limit database."""
        async with self._lock:
            conn = None
            try:
                conn = await self._get_db_connection()
                cursor = conn.cursor()
                current_time = time.time()
                # Use INSERT OR REPLACE to handle existing keys
                cursor.execute('''
                    INSERT OR REPLACE INTO rate_limited_keys (api_key, rate_limited_at)
                    VALUES (?, ?)
                ''', (api_key, current_time))
                conn.commit()
                logging.debug(f"Added/Updated rate-limited key ...{api_key[-4:]} for {self.provider_name}.")
            except sqlite3.Error as e:
                logging.error(f"Failed to add rate-limited key for {self.provider_name}: {e}", exc_info=True)
            finally:
                if conn:
                    conn.close()

    async def is_rate_limited(self, api_key: str) -> bool:
        """Checks if a specific key is currently rate-limited."""
        async with self._lock:
            conn = None
            try:
                conn = await self._get_db_connection()
                # Cleanup before checking
                await self._cleanup_expired_keys(conn)

                cursor = conn.cursor()
                cursor.execute("SELECT rate_limited_at FROM rate_limited_keys WHERE api_key = ?", (api_key,))
                row = cursor.fetchone()

                if row:
                    # Key exists, check timestamp (already cleaned up expired ones)
                    logging.debug(f"Key ...{api_key[-4:]} for {self.provider_name} is currently rate-limited.")
                    return True
                return False
            except sqlite3.Error as e:
                logging.error(f"Failed to check rate limit status for {self.provider_name}: {e}", exc_info=True)
                return False # Assume not rate-limited if DB check fails
            finally:
                if conn:
                    conn.close()

    async def get_valid_keys(self, all_keys: List[str]) -> List[str]:
        """Returns a list of keys from 'all_keys' that are not currently rate-limited."""
        async with self._lock:
            conn = None
            try:
                conn = await self._get_db_connection()
                await self._cleanup_expired_keys(conn) # Ensure expired keys are removed first

                cursor = conn.cursor()
                cursor.execute("SELECT api_key FROM rate_limited_keys")
                limited_keys_rows = cursor.fetchall()
                limited_keys_set = {row['api_key'] for row in limited_keys_rows}

                valid_keys = [key for key in all_keys if key not in limited_keys_set]
                logging.debug(f"Provider '{self.provider_name}': Total keys={len(all_keys)}, Limited keys={len(limited_keys_set)}, Valid keys={len(valid_keys)}")
                return valid_keys
            except sqlite3.Error as e:
                logging.error(f"Failed to get valid keys for {self.provider_name}: {e}", exc_info=True)
                return all_keys # Return all keys as a fallback if DB fails
            finally:
                if conn:
                    conn.close()

    async def reset_database(self):
        """Deletes all entries from the rate limit table for this provider."""
        async with self._lock:
            conn = None
            try:
                conn = await self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM rate_limited_keys")
                conn.commit()
                logging.info(f"Rate limit database for provider '{self.provider_name}' has been reset.")
            except sqlite3.Error as e:
                logging.error(f"Failed to reset database for {self.provider_name}: {e}", exc_info=True)
            finally:
                if conn:
                    conn.close()
