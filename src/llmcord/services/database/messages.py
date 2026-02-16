"""Message data storage for search results and responses."""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import libsql as libsql_module

libsql: Any = libsql_module
LibsqlCursor = Any
LibsqlConnection = Any

if TYPE_CHECKING:
    from .core import DatabaseProtocol as _Base
else:
    _Base = object

logger = logging.getLogger(__name__)
LIBSQL_ERROR = getattr(
    libsql,
    "LibsqlError",
    getattr(libsql, "Error", Exception),
)


@dataclass(slots=True)
class MessageResponsePayload:
    """Payload for storing message response data."""

    request_message_id: str
    request_user_id: str
    full_response: str | None = None
    thought_process: str | None = None
    grounding_metadata: dict[str, Any] | list[Any] | None = None
    tavily_metadata: dict[str, Any] | None = None
    failed_extractions: list[str] | None = None


class MessageDataMixin(_Base):
    """Mixin for message data storage."""

    def _init_message_tables(self) -> None:
        """Initialize message data tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Message search data table stores web search results and extracted URL
        # content.
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
                thought_process TEXT,
                grounding_metadata TEXT,
                tavily_metadata TEXT,
                failed_extractions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrations
        self._run_message_migrations(cursor, conn)

        conn.commit()

    def _run_message_migrations(
        self,
        cursor: LibsqlCursor,
        conn: LibsqlConnection,
    ) -> None:
        """Run migrations for message tables."""
        # Migration: ensure message_response_data has the expected schema.
        with contextlib.suppress(LIBSQL_ERROR, ValueError):
            cursor.execute("PRAGMA table_info(message_response_data)")
            columns = {row[1] for row in cursor.fetchall()}

            required_columns = {
                "message_id": "TEXT",
                "request_message_id": "TEXT",
                "request_user_id": "TEXT",
                "full_response": "TEXT",
                "thought_process": "TEXT",
                "grounding_metadata": "TEXT",
                "tavily_metadata": "TEXT",
                "failed_extractions": "TEXT",
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
                        thought_process TEXT,
                        grounding_metadata TEXT,
                        tavily_metadata TEXT,
                        failed_extractions TEXT,
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

    # Message search data methods
    def save_message_search_data(
        self,
        message_id: str,
        search_results: str | None = None,
        tavily_metadata: dict[str, Any] | None = None,
        lens_results: str | None = None,
    ) -> None:
        """Save web search results, lens results, and metadata for a message."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Ensure all parameters are primitive types that libsql can handle
        # (str, int, float, None) - convert any non-primitive types to strings
        # Use int() first to strip any subclass metadata (e.g., Discord Snowflake)
        try:
            msg_id_str = str(int(message_id))
        except (ValueError, TypeError):
            msg_id_str = str(message_id)

        # Ensure search_results is a plain string or None
        search_results_str: str | None = None
        if search_results:
            search_results_str = str(search_results)

        # Serialize metadata with fallback for non-JSON types
        tavily_json: str | None = None
        if tavily_metadata:
            try:
                tavily_json = json.dumps(tavily_metadata, default=str)
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to serialize tavily_metadata: %s", exc)
                tavily_json = None

        # Ensure lens_results is a plain string or None
        lens_results_str: str | None = None
        if lens_results:
            lens_results_str = str(lens_results)

        # Build params tuple with verified types
        params = (
            msg_id_str,
            search_results_str,
            tavily_json,
            lens_results_str,
            search_results_str,
            tavily_json,
            lens_results_str,
        )

        # Debug log parameter types if there's an issue
        try:
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
                params,
            )
        except ValueError:
            # Log parameter types to help debug "Unsupported parameter type" errors
            param_types = [
                (i, type(p).__name__, repr(p)[:100]) for i, p in enumerate(params)
            ]
            logger.exception(
                "libsql parameter type error. Param types: %s",
                param_types,
            )
            raise

        conn.commit()
        # Sync in background to avoid blocking
        try:
            self._sync()
        except LIBSQL_ERROR as exc:
            logger.debug("Background sync after save failed: %s", exc)
        logger.info("Saved search data for message %s", message_id)

    def get_message_search_data(
        self,
        message_id: str,
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        """Get web search results, metadata, and lens results for a message."""
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
                    ("Failed to decode tavily_metadata for message %s, returning None"),
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

    def save_message_response_data(
        self,
        message_id: str,
        payload: MessageResponsePayload,
    ) -> None:
        """Save response payloads for a Discord message."""
        conn = self._get_connection()
        cursor = conn.cursor()
        request_message_id = payload.request_message_id
        request_user_id = payload.request_user_id
        full_response = payload.full_response
        thought_process = payload.thought_process
        grounding_metadata = payload.grounding_metadata
        tavily_metadata = payload.tavily_metadata
        failed_extractions = payload.failed_extractions
        cursor.execute(
            """INSERT INTO message_response_data (
                   message_id,
                   request_message_id,
                   request_user_id,
                   full_response,
                   thought_process,
                   grounding_metadata,
                   tavily_metadata,
                   failed_extractions
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET
                   request_message_id = COALESCE(?, request_message_id),
                   request_user_id = COALESCE(?, request_user_id),
                   full_response = COALESCE(?, full_response),
                   thought_process = COALESCE(?, thought_process),
                   grounding_metadata = COALESCE(?, grounding_metadata),
                   tavily_metadata = COALESCE(?, tavily_metadata),
                   failed_extractions = COALESCE(?, failed_extractions)""",
            (
                str(message_id),
                str(request_message_id),
                str(request_user_id),
                full_response,
                thought_process,
                json.dumps(grounding_metadata) if grounding_metadata else None,
                json.dumps(tavily_metadata) if tavily_metadata else None,
                json.dumps(failed_extractions) if failed_extractions else None,
                str(request_message_id),
                str(request_user_id),
                full_response,
                thought_process,
                json.dumps(grounding_metadata) if grounding_metadata else None,
                json.dumps(tavily_metadata) if tavily_metadata else None,
                json.dumps(failed_extractions) if failed_extractions else None,
            ),
        )
        conn.commit()
        try:
            self._sync()
        except LIBSQL_ERROR as exc:
            logger.debug("Background sync after save failed: %s", exc)
        logger.info("Saved response data for message %s", message_id)

    def get_message_response_data(
        self,
        message_id: str,
    ) -> tuple[
        str | None,
        str | None,
        dict[str, Any] | list[Any] | None,
        dict[str, Any] | None,
        str | None,
        str | None,
        list[str] | None,
    ]:
        """Get response data for a Discord message."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT full_response, thought_process, grounding_metadata, "
            "tavily_metadata, "
            "request_message_id, request_user_id, failed_extractions "
            "FROM message_response_data WHERE message_id = ?",
            (str(message_id),),
        )
        result = cursor.fetchone()
        if not result:
            return None, None, None, None, None, None, None

        full_response = result[0]
        thought_process = result[1]
        grounding_metadata_raw = result[2]
        tavily_metadata_raw = result[3]
        request_message_id = result[4]
        request_user_id = result[5]
        failed_extractions_raw = result[6]

        grounding_metadata = None
        tavily_metadata = None
        failed_extractions = None

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

        if failed_extractions_raw:
            try:
                parsed_failed_extractions = json.loads(failed_extractions_raw)
                if isinstance(parsed_failed_extractions, list):
                    failed_extractions = [
                        str(entry) for entry in parsed_failed_extractions
                    ]
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode failed_extractions for message %s",
                    message_id,
                )

        return (
            full_response,
            thought_process,
            grounding_metadata,
            tavily_metadata,
            request_message_id,
            request_user_id,
            failed_extractions,
        )
