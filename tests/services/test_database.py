"""Tests for database services and key rotation."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from llmcord.services.database import BadKeysDB, KeyRotator

MIN_CREATE_TABLE_CALLS = 5
EXPECTED_ROTATED_KEYS = 4


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


@pytest.fixture
def mock_libsql() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Provide mocked libsql connection and cursor objects."""
    with patch("llmcord.services.database.libsql.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cursor


def test_init_db(mock_libsql: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Initialize DB and ensure schema setup is attempted."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    BadKeysDB(local_db_path=":memory:")

    # Check if tables were created
    assert_true(
        condition=mock_cursor.execute.call_count >= MIN_CREATE_TABLE_CALLS,
        message="Expected multiple schema creation calls",
    )
    # We can inspect the calls if needed, but checking count is a good start


def test_mark_key_bad(mock_libsql: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Mark key as bad should write entry and commit."""
    _mock_connect, mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()

    db.mark_key_bad("provider", "api_key", "error")

    # Check insert
    expected_hash = hashlib.sha256(b"api_key").hexdigest()[:16]
    query, params = mock_cursor.execute.call_args[0]
    assert_true(
        condition="INSERT OR REPLACE INTO bad_keys" in query,
        message="Expected bad_keys insert statement",
    )
    assert_true(
        condition=params == ("provider", expected_hash, "error"),
        message="Expected provider, hash, and error params",
    )
    mock_conn.commit.assert_called()


def test_is_key_bad_true(mock_libsql: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Return True when cursor finds a matching row."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchone.return_value = (1,)
    db = BadKeysDB()

    assert_true(
        condition=db.is_key_bad("provider", "api_key") is True,
        message="Expected key to be flagged as bad",
    )


def test_is_key_bad_false(mock_libsql: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """Return False when cursor finds no matching row."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchone.return_value = None
    db = BadKeysDB()

    assert_true(
        condition=db.is_key_bad("provider", "api_key") is False,
        message="Expected key to be considered good",
    )


def test_user_model_preferences(
    mock_libsql: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Set and get user model preferences via database."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()

    # Set
    db.set_user_model("user1", "model1")
    assert_true(
        condition="INSERT INTO user_model_preferences"
        in mock_cursor.execute.call_args[0][0],
        message="Expected insert into user_model_preferences",
    )

    # Get
    mock_cursor.fetchone.return_value = ("model1",)
    assert_true(
        condition=db.get_user_model("user1") == "model1",
        message="Expected stored model preference",
    )


def test_message_search_data(
    mock_libsql: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Save and load message search data from database."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()

    # Save
    db.save_message_search_data("msg1", "results")
    insert_query = mock_cursor.execute.call_args[0][0]
    assert_true(
        condition="INSERT INTO message_search_data" in insert_query,
        message="Expected insert into message_search_data",
    )

    # Get
    mock_cursor.fetchone.return_value = ("results", None, None)
    results, _meta, _lens = db.get_message_search_data("msg1")
    assert_true(
        condition=results == "results",
        message="Expected saved search results",
    )


def test_key_rotator(mock_libsql: tuple[MagicMock, MagicMock, MagicMock]) -> None:
    """KeyRotator should yield all keys with retries when none are bad."""
    _mock_connect, _mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchall.return_value = []  # No bad keys

    # Create a local db instance that uses the mocked libsql
    local_db = BadKeysDB()

    # Patch get_bad_keys_db to return our local_db so KeyRotator uses it
    with patch("llmcord.services.database.get_bad_keys_db", return_value=local_db):
        rotator = KeyRotator("provider", ["key1", "key2"], db=local_db)
        keys = list(rotator.get_keys())
        assert_true(
            condition=len(keys) == EXPECTED_ROTATED_KEYS,
            message="Expected keys repeated for retries",
        )
