import pytest
from unittest.mock import MagicMock, patch
from llmcord.services.database import BadKeysDB, KeyRotator

@pytest.fixture
def mock_libsql():
    with patch("llmcord.services.database.libsql.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_connect, mock_conn, mock_cursor

def test_init_db(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB(local_db_path=":memory:")
    
    # Check if tables were created
    assert mock_cursor.execute.call_count >= 5 # At least 5 CREATE TABLE calls and some migrations
    # We can inspect the calls if needed, but checking count is a good start

def test_mark_key_bad(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()
    
    db.mark_key_bad("provider", "api_key", "error")
    
    # Check insert
    mock_cursor.execute.assert_called_with(
        "INSERT OR REPLACE INTO bad_keys (provider, key_hash, error_message) VALUES (?, ?, ?)",
        ("provider", db._hash_key("api_key"), "error")
    )
    mock_conn.commit.assert_called()

def test_is_key_bad_true(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchone.return_value = (1,)
    db = BadKeysDB()
    
    assert db.is_key_bad("provider", "api_key") is True

def test_is_key_bad_false(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchone.return_value = None
    db = BadKeysDB()
    
    assert db.is_key_bad("provider", "api_key") is False

def test_user_model_preferences(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()
    
    # Set
    db.set_user_model("user1", "model1")
    assert "INSERT INTO user_model_preferences" in mock_cursor.execute.call_args[0][0]
    
    # Get
    mock_cursor.fetchone.return_value = ("model1",)
    assert db.get_user_model("user1") == "model1"

def test_message_search_data(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    db = BadKeysDB()
    
    # Save
    db.save_message_search_data("msg1", "results")
    assert "INSERT INTO message_search_data" in mock_cursor.execute.call_args[0][0]
    
    # Get
    mock_cursor.fetchone.return_value = ("results", None, None)
    results, meta, lens = db.get_message_search_data("msg1")
    assert results == "results"

def test_key_rotator(mock_libsql):
    mock_connect, mock_conn, mock_cursor = mock_libsql
    mock_cursor.fetchall.return_value = [] # No bad keys
    
    # Create a local db instance that uses the mocked libsql
    local_db = BadKeysDB()
    
    # Patch get_bad_keys_db to return our local_db so KeyRotator uses it
    with patch("llmcord.services.database.get_bad_keys_db", return_value=local_db):
        rotator = KeyRotator("provider", ["key1", "key2"])
        keys = list(rotator.get_keys())
        assert len(keys) == 4 # 2 keys * 2 retries
