"""User preferences management."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class UserPreferencesMixin:
    """Mixin for user preference storage."""

    def _init_user_tables(self) -> None:
        """Initialize user preference tables."""
        # We assume self is mixed in with DatabaseCore
        conn = self._get_connection()
        cursor = conn.cursor()

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
        conn.commit()

    # User model preferences methods
    def get_user_model(self, user_id: str) -> str | None:
        """Get the preferred model for a user.

        Returns None if not set.
        """
        # _with_reconnect should be applied by the consumer or we need to import it.
        # Ideally, we apply it here if we can import it, or rely on the main class.
        # For simplicity in this mixin structure, we assume the methods utilizing I/O
        # will be decorated in the final class OR we import the decorator here.
        # Let's import the decorator to keep it self-contained.
        return self._get_user_model_impl(user_id)

    def _get_user_model_impl(self, user_id: str) -> str | None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT model FROM user_model_preferences WHERE user_id = ?",
            (str(user_id),),
        )
        result = cursor.fetchone()
        return result[0] if result else None

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
