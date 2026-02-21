"""User preferences management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import DatabaseProtocol as _Base
else:
    _Base = object

logger = logging.getLogger(__name__)


class UserPreferencesMixin(_Base):
    """Mixin for user preference storage."""

    async def _init_user_tables(self) -> None:
        """Initialize user preference tables."""
        conn = await self._get_connection()

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_model_preferences (
                user_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_search_decider_preferences (
                user_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()

    # User model preferences methods
    async def aget_user_model(self, user_id: str) -> str | None:
        """Get the preferred model for a user without blocking the event loop."""
        return await self._run_db_call_async(self._get_user_model_impl, user_id)

    async def _get_user_model_impl(self, user_id: str) -> str | None:
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT model FROM user_model_preferences WHERE user_id = ?",
            (str(user_id),),
        ) as cursor:
            result = await cursor.fetchone()
        return result[0] if result else None

    async def aset_user_model(self, user_id: str, model: str) -> None:
        """Set the preferred model for a user without blocking the event loop."""
        await self._run_db_call_async(self._set_user_model_impl, user_id, model)

    async def _set_user_model_impl(self, user_id: str, model: str) -> None:
        conn = await self._get_connection()
        await conn.execute(
            """INSERT INTO user_model_preferences (user_id, model, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   model = ?,
                   updated_at = CURRENT_TIMESTAMP""",
            (str(user_id), model, model),
        )
        await conn.commit()
        logger.info("Set model preference for user %s: %s", user_id, model)

    # User search decider model preferences methods
    async def aget_user_search_decider_model(self, user_id: str) -> str | None:
        """Get decider model for a user without blocking the event loop."""
        return await self._run_db_call_async(
            self._get_user_search_decider_model_impl,
            user_id,
        )

    async def _get_user_search_decider_model_impl(self, user_id: str) -> str | None:
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT model FROM user_search_decider_preferences WHERE user_id = ?",
            (str(user_id),),
        ) as cursor:
            result = await cursor.fetchone()
        return result[0] if result else None

    async def aset_user_search_decider_model(self, user_id: str, model: str) -> None:
        """Set decider model for a user without blocking the event loop."""
        await self._run_db_call_async(
            self._set_user_search_decider_model_impl,
            user_id,
            model,
        )

    async def _set_user_search_decider_model_impl(
        self,
        user_id: str,
        model: str,
    ) -> None:
        conn = await self._get_connection()
        await conn.execute(
            (
                "INSERT INTO user_search_decider_preferences "
                "(user_id, model, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id) DO UPDATE SET model = ?, "
                "updated_at = CURRENT_TIMESTAMP"
            ),
            (str(user_id), model, model),
        )
        await conn.commit()
        logger.info(
            "Set search decider model preference for user %s: %s",
            user_id,
            model,
        )

    async def areset_all_user_model_preferences(self) -> int:
        """Reset all model preferences without blocking the event loop."""
        return await self._run_db_call_async(
            self._reset_all_user_model_preferences_impl,
        )

    async def _reset_all_user_model_preferences_impl(self) -> int:
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM user_model_preferences",
        ) as cursor:
            row = await cursor.fetchone()
        count = int(row[0]) if row else 0
        await conn.execute("DELETE FROM user_model_preferences")
        await conn.commit()
        logger.info(
            "Reset all user model preferences (%d users affected)",
            count,
        )
        return count

    async def areset_all_user_search_decider_preferences(self) -> int:
        """Reset all decider model preferences without blocking the event loop."""
        return await self._run_db_call_async(
            self._reset_all_user_search_decider_preferences_impl,
        )

    async def _reset_all_user_search_decider_preferences_impl(self) -> int:
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM user_search_decider_preferences",
        ) as cursor:
            row = await cursor.fetchone()
        count = int(row[0]) if row else 0
        await conn.execute("DELETE FROM user_search_decider_preferences")
        await conn.commit()
        logger.info(
            "Reset all user search decider model preferences (%d users affected)",
            count,
        )
        return count
