"""
Turso (libSQL) based tracking of bad API keys to avoid wasting retries.
Uses Turso cloud database for persistent storage across deployments.
"""
import hashlib
import logging
import os
import functools

import libsql


def _with_reconnect(method):
    """Decorator to handle stale Turso connections by reconnecting and retrying."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                return method(self, *args, **kwargs)
            except (ValueError, Exception) as e:
                error_str = str(e)
                # Check for Hrana stream errors (stale connection)
                if "stream not found" in error_str or "Hrana" in error_str:
                    if attempt < max_retries - 1:
                        logging.warning(f"Turso connection error, reconnecting (attempt {attempt + 1}): {e}")
                        self._reconnect()
                    else:
                        logging.error(f"Failed to reconnect to Turso after {max_retries} attempts: {e}")
                        raise
                else:
                    raise
    return wrapper


class BadKeysDB:
    """Turso/libSQL-based tracking of bad API keys to avoid wasting retries."""
    
    def __init__(
        self, 
        db_url: str = None,
        auth_token: str = None,
        local_db_path: str = "bad_keys.db"
    ):
        """
        Initialize the Turso database connection.
        
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
    
    def _reconnect(self):
        """Force reconnection to the database."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._get_connection()
    
    def _get_connection(self):
        """Get or create the database connection."""
        if self._conn is None:
            if self.db_url and self.auth_token:
                # Connect to Turso cloud with local embedded replica
                self._conn = libsql.connect(
                    self.local_db_path,
                    sync_url=self.db_url,
                    auth_token=self.auth_token
                )
                # Sync with remote on initial connection
                self._conn.sync()
                logging.info(f"Connected to Turso database: {self.db_url}")
            else:
                # Fallback to local-only SQLite if no Turso credentials
                self._conn = libsql.connect(self.local_db_path)
                logging.warning("No Turso credentials found, using local SQLite database")
        return self._conn
    
    def _init_db(self):
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
        # Message search data table - stores web search results and extracted URL content
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message_search_data (
                message_id TEXT PRIMARY KEY,
                search_results TEXT,
                tavily_metadata TEXT,
                lens_results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: Add lens_results column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE message_search_data ADD COLUMN lens_results TEXT")
            conn.commit()
        except Exception:
            pass  # Column already exists
        conn.commit()
        self._sync()
    
    def _sync(self):
        """Sync changes with Turso cloud if connected to remote."""
        if self._conn and self.db_url and self.auth_token:
            try:
                self._conn.sync()
            except Exception as e:
                logging.warning(f"Failed to sync with Turso: {e}")
    
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
            (provider, self._hash_key(api_key))
        )
        return cursor.fetchone() is not None
    
    @_with_reconnect
    def mark_key_bad(self, provider: str, api_key: str, error_message: str = None):
        """Mark an API key as bad."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
            (provider, self._hash_key(api_key), error_message)
        )
        conn.commit()
        self._sync()
        logging.info(f"Marked API key as bad for provider '{provider}' (hash: {self._hash_key(api_key)[:8]}...)")
    
    def get_good_keys(self, provider: str, all_keys: list[str]) -> list[str]:
        """Filter out bad keys from a list of API keys."""
        return [key for key in all_keys if not self.is_key_bad(provider, key)]
    
    @_with_reconnect
    def is_key_bad_synced(self, provider: str, api_key: str) -> bool:
        """
        Check if an API key is marked as bad for EITHER the main provider or its decider variant.
        This ensures keys marked bad by main model are also recognized by decider and vice versa.
        Uses a single optimized query instead of two separate calls.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"
        key_hash = self._hash_key(api_key)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM bad_keys WHERE (provider = ? OR provider = ?) AND key_hash = ? LIMIT 1",
            (base_provider, decider_provider, key_hash)
        )
        return cursor.fetchone() is not None
    
    @_with_reconnect
    def mark_key_bad_synced(self, provider: str, api_key: str, error_message: str = None):
        """
        Mark an API key as bad for BOTH the main provider and its decider variant.
        This ensures a bad key is recognized by both main model and search decider.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        key_hash = self._hash_key(api_key)
        
        # Mark for both base provider and decider provider
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
            (base_provider, key_hash, error_message)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
            (decider_provider, key_hash, error_message)
        )
        conn.commit()
        self._sync()
        logging.info(f"Marked API key as bad for '{base_provider}' and '{decider_provider}' (hash: {key_hash[:8]}...)")
    
    @_with_reconnect
    def get_good_keys_synced(self, provider: str, all_keys: list[str]) -> list[str]:
        """
        Filter out bad keys from a list of API keys, checking both main and decider providers.
        This ensures keys marked bad by either are filtered out for both.
        Uses a single bulk query for efficiency instead of checking each key individually.
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
            (base_provider, decider_provider)
        )
        bad_hashes = {row[0] for row in cursor.fetchall()}
        
        # Filter keys locally using the pre-fetched bad hashes
        return [key for key in all_keys if self._hash_key(key) not in bad_hashes]
    
    @_with_reconnect
    def reset_provider_keys_synced(self, provider: str):
        """
        Reset all bad keys for BOTH the main provider and its decider variant.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys WHERE provider = ? OR provider = ?", (base_provider, decider_provider))
        conn.commit()
        self._sync()
        logging.info(f"Reset all bad keys for '{base_provider}' and '{decider_provider}'")
    
    @_with_reconnect
    def get_bad_key_count(self, provider: str) -> int:
        """Get the count of bad keys for a provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM bad_keys WHERE provider = ?",
            (provider,)
        )
        return cursor.fetchone()[0]
    
    @_with_reconnect
    def reset_provider_keys(self, provider: str):
        """Reset all bad keys for a specific provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys WHERE provider = ?", (provider,))
        conn.commit()
        self._sync()
        logging.info(f"Reset all bad keys for provider '{provider}'")
    
    @_with_reconnect
    def reset_all(self):
        """Reset all bad keys for all providers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys")
        conn.commit()
        self._sync()
        logging.info("Reset all bad keys database")
    
    # User model preferences methods
    @_with_reconnect
    def get_user_model(self, user_id: str) -> str | None:
        """Get the preferred model for a user. Returns None if not set."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model FROM user_model_preferences WHERE user_id = ?",
            (str(user_id),)
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
               ON CONFLICT(user_id) DO UPDATE SET model = ?, updated_at = CURRENT_TIMESTAMP""",
            (str(user_id), model, model)
        )
        conn.commit()
        self._sync()
        logging.info(f"Set model preference for user {user_id}: {model}")
    
    # User search decider model preferences methods
    @_with_reconnect
    def get_user_search_decider_model(self, user_id: str) -> str | None:
        """Get the preferred search decider model for a user. Returns None if not set."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model FROM user_search_decider_preferences WHERE user_id = ?",
            (str(user_id),)
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
               ON CONFLICT(user_id) DO UPDATE SET model = ?, updated_at = CURRENT_TIMESTAMP""",
            (str(user_id), model, model)
        )
        conn.commit()
        self._sync()
        logging.info(f"Set search decider model preference for user {user_id}: {model}")
    
    # Message search data methods for persisting web search results in chat history
    @_with_reconnect
    def save_message_search_data(self, message_id: str, search_results: str = None, tavily_metadata: dict | None = None, lens_results: str = None) -> None:
        """
        Save web search results, lens results, and metadata associated with a Discord message.
        This allows search results to persist in chat history when conversations are rebuilt.
        """
        import json
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO message_search_data (message_id, search_results, tavily_metadata, lens_results) 
               VALUES (?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET 
                   search_results = COALESCE(?, search_results), 
                   tavily_metadata = COALESCE(?, tavily_metadata),
                   lens_results = COALESCE(?, lens_results)""",
            (str(message_id), search_results, json.dumps(tavily_metadata) if tavily_metadata else None, lens_results,
             search_results, json.dumps(tavily_metadata) if tavily_metadata else None, lens_results)
        )
        conn.commit()
        self._sync()
        logging.info(f"Saved search data for message {message_id}")
    
    @_with_reconnect
    def get_message_search_data(self, message_id: str) -> tuple[str | None, dict | None, str | None]:
        """
        Get web search results, metadata, and lens results associated with a Discord message.
        Returns a tuple of (search_results, tavily_metadata, lens_results) or (None, None, None) if not found.
        """
        import json
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT search_results, tavily_metadata, lens_results FROM message_search_data WHERE message_id = ?",
            (str(message_id),)
        )
        result = cursor.fetchone()
        if result:
            search_results = result[0]
            try:
                tavily_metadata = json.loads(result[1]) if result[1] else None
            except json.JSONDecodeError:
                logging.warning(f"Failed to decode tavily_metadata for message {message_id}, returning None")
                tavily_metadata = None
            lens_results = result[2]
            return search_results, tavily_metadata, lens_results
        return None, None, None


# Global instance will be initialized with config values
bad_keys_db = None


def init_bad_keys_db(db_url: str = None, auth_token: str = None):
    """Initialize the global bad keys database instance."""
    global bad_keys_db
    bad_keys_db = BadKeysDB(db_url=db_url, auth_token=auth_token)
    return bad_keys_db


def get_bad_keys_db() -> BadKeysDB:
    """Get the global bad keys database instance, initializing if needed."""
    global bad_keys_db
    if bad_keys_db is None:
        bad_keys_db = BadKeysDB()
    return bad_keys_db
