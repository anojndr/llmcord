"""
SQLite-based tracking of bad API keys to avoid wasting retries.
"""
import hashlib
import logging
import sqlite3


class BadKeysDB:
    """SQLite-based tracking of bad API keys to avoid wasting retries."""
    
    def __init__(self, db_path: str = "bad_keys.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
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
        finally:
            conn.close()
    
    def _hash_key(self, api_key: str) -> str:
        """Create a hash of the API key to avoid storing sensitive data."""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    def is_key_bad(self, provider: str, api_key: str) -> bool:
        """Check if an API key is marked as bad."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM bad_keys WHERE provider = ? AND key_hash = ?",
                (provider, self._hash_key(api_key))
            )
            return cursor.fetchone() is not None
        finally:
            conn.close()
    
    def mark_key_bad(self, provider: str, api_key: str, error_message: str = None):
        """Mark an API key as bad."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
                (provider, self._hash_key(api_key), error_message)
            )
            conn.commit()
            logging.info(f"Marked API key as bad for provider '{provider}' (hash: {self._hash_key(api_key)[:8]}...)")
        finally:
            conn.close()
    
    def get_good_keys(self, provider: str, all_keys: list[str]) -> list[str]:
        """Filter out bad keys from a list of API keys."""
        return [key for key in all_keys if not self.is_key_bad(provider, key)]
    
    def get_bad_key_count(self, provider: str) -> int:
        """Get the count of bad keys for a provider."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM bad_keys WHERE provider = ?",
                (provider,)
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def reset_provider_keys(self, provider: str):
        """Reset all bad keys for a specific provider."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bad_keys WHERE provider = ?", (provider,))
            conn.commit()
            logging.info(f"Reset all bad keys for provider '{provider}'")
        finally:
            conn.close()
    
    def reset_all(self):
        """Reset all bad keys for all providers."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bad_keys")
            conn.commit()
            logging.info("Reset all bad keys database")
        finally:
            conn.close()


# Global instance of the bad keys database
bad_keys_db = BadKeysDB()
