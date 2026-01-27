from __future__ import annotations

import pathlib

import pytest

import bad_keys


@pytest.fixture()
def temp_db(tmp_path: pathlib.Path) -> bad_keys.BadKeysDB:
    db_path = tmp_path / "bad_keys.db"
    return bad_keys.BadKeysDB(local_db_path=str(db_path))


def test_bad_keys_mark_and_check(temp_db: bad_keys.BadKeysDB) -> None:
    temp_db.mark_key_bad("openai", "secret-key", "oops")

    assert temp_db.is_key_bad("openai", "secret-key") is True
    assert temp_db.is_key_bad("openai", "other-key") is False
    assert temp_db.get_good_keys("openai", ["secret-key", "other-key"]) == [
        "other-key",
    ]


def test_bad_keys_synced(temp_db: bad_keys.BadKeysDB) -> None:
    temp_db.mark_key_bad_synced("openai", "key1")
    assert temp_db.is_key_bad_synced("openai", "key1") is True
    assert temp_db.is_key_bad_synced("decider_openai", "key1") is True


def test_user_preferences(temp_db: bad_keys.BadKeysDB) -> None:
    assert temp_db.get_user_model("123") is None
    temp_db.set_user_model("123", "gpt-4.1")
    assert temp_db.get_user_model("123") == "gpt-4.1"

    assert temp_db.get_user_search_decider_model("123") is None
    temp_db.set_user_search_decider_model("123", "gemini-3-flash-preview")
    assert temp_db.get_user_search_decider_model("123") == "gemini-3-flash-preview"

    assert temp_db.reset_all_user_model_preferences() == 1
    assert temp_db.reset_all_user_search_decider_preferences() == 1


def test_message_search_data_roundtrip(temp_db: bad_keys.BadKeysDB) -> None:
    temp_db.save_message_search_data(
        message_id="1",
        search_results="result",
        tavily_metadata={"queries": ["q1"]},
        lens_results="lens",
    )

    search_results, tavily_metadata, lens_results = temp_db.get_message_search_data("1")
    assert search_results == "result"
    assert tavily_metadata == {"queries": ["q1"]}
    assert lens_results == "lens"


def test_message_search_data_handles_bad_json(temp_db: bad_keys.BadKeysDB) -> None:
    conn = temp_db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO message_search_data (message_id, tavily_metadata) "
        "VALUES (?, ?)",
        ("2", "{bad json}"),
    )
    conn.commit()

    _, tavily_metadata, _ = temp_db.get_message_search_data("2")
    assert tavily_metadata is None


def test_key_rotator_marks_bad(monkeypatch: pytest.MonkeyPatch, temp_db: bad_keys.BadKeysDB) -> None:
    monkeypatch.setattr(bad_keys, "get_bad_keys_db", lambda: temp_db)

    rotator = bad_keys.KeyRotator("openai", ["key1", "key2"], max_retries_multiplier=1)

    keys = list(rotator.get_keys())
    assert keys == ["key1", "key2"]

    rotator = bad_keys.KeyRotator("openai", ["key1", "key2"], max_retries_multiplier=1)
    iterator = rotator.get_keys()
    current = next(iterator)
    assert current == "key1"
    rotator.mark_current_bad("error")

    assert temp_db.is_key_bad_synced("openai", "key1") is True
    assert rotator.get_good_key_count() == 1
    assert rotator.has_keys is True


def test_get_good_keys_synced(temp_db: bad_keys.BadKeysDB) -> None:
    temp_db.mark_key_bad_synced("tavily", "bad")
    keys = temp_db.get_good_keys_synced("tavily", ["bad", "good"])
    assert keys == ["good"]


def test_reset_provider_keys_synced(temp_db: bad_keys.BadKeysDB) -> None:
    temp_db.mark_key_bad_synced("tavily", "bad")
    temp_db.reset_provider_keys_synced("tavily")
    assert temp_db.get_bad_key_count("tavily") == 0
