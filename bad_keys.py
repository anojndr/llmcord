"""
Turso (libSQL) based tracking of bad API keys to avoid wasting retries.
Uses Turso cloud database for persistent storage across deployments.
"""
import hashlib
import logging
import os

import libsql


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
        """Initialize the database table if it doesn't exist."""
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
    
    def is_key_bad(self, provider: str, api_key: str) -> bool:
        """Check if an API key is marked as bad."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM bad_keys WHERE provider = ? AND key_hash = ?",
            (provider, self._hash_key(api_key))
        )
        return cursor.fetchone() is not None
    
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
    
    def is_key_bad_synced(self, provider: str, api_key: str) -> bool:
        """
        Check if an API key is marked as bad for EITHER the main provider or its decider variant.
        This ensures keys marked bad by main model are also recognized by decider and vice versa.
        """
        base_provider = provider.removeprefix("decider_")
        decider_provider = f"decider_{base_provider}"
        return self.is_key_bad(base_provider, api_key) or self.is_key_bad(decider_provider, api_key)
    
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
    
    def get_good_keys_synced(self, provider: str, all_keys: list[str]) -> list[str]:
        """
        Filter out bad keys from a list of API keys, checking both main and decider providers.
        This ensures keys marked bad by either are filtered out for both.
        """
        return [key for key in all_keys if not self.is_key_bad_synced(provider, key)]
    
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
    
    def get_bad_key_count(self, provider: str) -> int:
        """Get the count of bad keys for a provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM bad_keys WHERE provider = ?",
            (provider,)
        )
        return cursor.fetchone()[0]
    
    def reset_provider_keys(self, provider: str):
        """Reset all bad keys for a specific provider."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys WHERE provider = ?", (provider,))
        conn.commit()
        self._sync()
        logging.info(f"Reset all bad keys for provider '{provider}'")
    
    def reset_all(self):
        """Reset all bad keys for all providers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM bad_keys")
        conn.commit()
        self._sync()
        logging.info("Reset all bad keys database")


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
