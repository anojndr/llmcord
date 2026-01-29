from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmcord.services import database


@pytest.fixture()
def db(tmp_path: Path) -> database.BadKeysDB:
    instance = database.BadKeysDB(local_db_path=str(tmp_path / "test.db"))
    database._bad_keys_state["instance"] = instance
    return instance


def test_with_reconnect_wrapper() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.calls = 0
            self.reconnected = 0

        def _reconnect(self) -> None:
            self.reconnected += 1

        @database._with_reconnect
        def flaky(self) -> str:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Hrana stream not found")
            return "ok"

        @database._with_reconnect
        def always_bad(self) -> str:
            raise ValueError("nope")

        @database._with_reconnect
        def stale_twice(self) -> str:
            raise ValueError("Hrana stream not found")

    dummy = Dummy()
    assert dummy.flaky() == "ok"
    assert dummy.reconnected == 1

    with pytest.raises(ValueError):
        dummy.always_bad()

    with pytest.raises(ValueError):
        dummy.stale_twice()


def test_mark_and_check_bad_keys(db: database.BadKeysDB) -> None:
    assert db.is_key_bad("provider", "key") is False

    db.mark_key_bad("provider", "key", "error")

    assert db.is_key_bad("provider", "key") is True
    assert db.get_bad_key_count("provider") == 1
    assert db.get_good_keys("provider", ["key", "good"]) == ["good"]

    db.reset_provider_keys("provider")
    assert db.is_key_bad("provider", "key") is False

    db._reconnect()


def test_synced_key_functions(db: database.BadKeysDB) -> None:
    db.mark_key_bad_synced("openai", "key", "error")
    assert db.is_key_bad_synced("openai", "key") is True
    assert db.is_key_bad_synced("decider_openai", "key") is True

    keys = db.get_good_keys_synced("openai", ["key", "good"])
    assert keys == ["good"]

    db.reset_provider_keys_synced("openai")
    assert db.is_key_bad_synced("openai", "key") is False

    assert db.get_good_keys_synced("openai", []) == []


def test_user_model_preferences(db: database.BadKeysDB) -> None:
    assert db.get_user_model("1") is None

    db.set_user_model("1", "model-a")
    assert db.get_user_model("1") == "model-a"

    assert db.get_user_search_decider_model("1") is None
    db.set_user_search_decider_model("1", "model-b")
    assert db.get_user_search_decider_model("1") == "model-b"

    assert db.reset_all_user_model_preferences() == 1
    assert db.reset_all_user_search_decider_preferences() == 1


def test_message_search_data(db: database.BadKeysDB) -> None:
    db.save_message_search_data(
        "123",
        search_results="results",
        tavily_metadata={"a": 1},
        lens_results="lens",
    )

    search_results, metadata, lens = db.get_message_search_data("123")
    assert search_results == "results"
    assert metadata == {"a": 1}
    assert lens == "lens"

    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE message_search_data SET tavily_metadata = ? WHERE message_id = ?",
        ("not-json", "123"),
    )
    conn.commit()

    search_results, metadata, lens = db.get_message_search_data("123")
    assert search_results == "results"
    assert metadata is None
    assert lens == "lens"

    missing = db.get_message_search_data("missing")
    assert missing == (None, None, None)


def test_message_search_data_sync_failure(
    db: database.BadKeysDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLibsqlError(Exception):
        pass

    monkeypatch.setattr(database.libsql, "LibsqlError", FakeLibsqlError, raising=False)

    def raise_sync() -> None:
        raise FakeLibsqlError("sync failed")

    monkeypatch.setattr(db, "_sync", raise_sync)

    db.save_message_search_data("sync", search_results="x")


def test_message_response_data(db: database.BadKeysDB) -> None:
    db.save_message_response_data(
        "555",
        "111",
        "222",
        full_response="full",
        grounding_metadata=[{"ground": True}],
        tavily_metadata={"t": 1},
    )

    full, grounding, tavily, req_msg, req_user = db.get_message_response_data("555")
    assert full == "full"
    assert grounding == [{"ground": True}]
    assert tavily == {"t": 1}
    assert req_msg == "111"
    assert req_user == "222"

    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE message_response_data SET grounding_metadata = ? WHERE message_id = ?",
        ("no-json", "555"),
    )
    cursor.execute(
        "UPDATE message_response_data SET tavily_metadata = ? WHERE message_id = ?",
        ("no-json", "555"),
    )
    conn.commit()

    full, grounding, tavily, req_msg, req_user = db.get_message_response_data("555")
    assert full == "full"
    assert grounding is None
    assert tavily is None
    assert req_msg == "111"
    assert req_user == "222"

    missing = db.get_message_response_data("missing")
    assert missing == (None, None, None, None, None)


def test_message_response_data_sync_failure(
    db: database.BadKeysDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLibsqlError(Exception):
        pass

    monkeypatch.setattr(database.libsql, "LibsqlError", FakeLibsqlError, raising=False)

    def raise_sync() -> None:
        raise FakeLibsqlError("sync failed")

    monkeypatch.setattr(db, "_sync", raise_sync)

    db.save_message_response_data("sync", "1", "2", full_response="x")


def test_key_rotator(db: database.BadKeysDB) -> None:
    rotator = database.KeyRotator("provider", ["k1", "k2"], max_retries_multiplier=1)

    keys = []
    for key in rotator.get_keys():
        keys.append(key)
        rotator.mark_current_bad("bad")

    assert keys == ["k1", "k2"]
    assert rotator.get_good_key_count() == 2  # reset after exhaustion
    assert rotator.get_total_key_count() == 2
    assert rotator.has_keys is True


def test_key_rotator_all_keys_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDB:
        def __init__(self) -> None:
            self.reset_called = 0

        def get_good_keys_synced(self, _provider: str, _keys: list[str]) -> list[str]:
            return []

        def reset_provider_keys_synced(self, _provider: str) -> None:
            self.reset_called += 1

        def mark_key_bad_synced(self, _provider: str, _key: str, _msg: str) -> None:
            return None

    dummy_db = DummyDB()
    monkeypatch.setattr(database, "get_bad_keys_db", lambda: dummy_db)

    rotator = database.KeyRotator("provider", ["k1"])

    assert rotator.get_good_key_count() == 1
    assert dummy_db.reset_called == 1


def test_key_rotator_mark_current_bad_without_key(db: database.BadKeysDB) -> None:
    rotator = database.KeyRotator("provider", ["k1", "k2"])
    rotator.mark_current_bad("oops")


def test_key_rotator_async_iteration(db: database.BadKeysDB) -> None:
    async def _collect() -> list[str]:
        rotator = database.KeyRotator("provider", ["k1", "k2"], max_retries_multiplier=1)
        results: list[str] = []
        async for key in rotator.get_keys_async():
            results.append(key)
            break
        return results

    results = asyncio.run(_collect())
    assert results == ["k1"]


def test_hash_key_length(db: database.BadKeysDB) -> None:
    assert len(db._hash_key("secret")) == 16


def test_reset_all(db: database.BadKeysDB) -> None:
    db.mark_key_bad("provider", "key")
    db.mark_key_bad("provider", "key2")
    db.reset_all()
    assert db.get_bad_key_count("provider") == 0


def test_sync_with_remote_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCursor:
        def execute(self, _sql: str, _params: tuple | None = None) -> None:
            return None

        def fetchall(self) -> list[tuple]:
            return []

        def fetchone(self):
            return None

    class DummyConnection:
        def __init__(self) -> None:
            self.synced = 0

        def cursor(self) -> DummyCursor:
            return DummyCursor()

        def commit(self) -> None:
            return None

        def sync(self) -> None:
            self.synced += 1

        def close(self) -> None:
            return None

    def fake_connect(_path: str, sync_url: str | None = None, auth_token: str | None = None):
        assert sync_url == "libsql://example"
        assert auth_token == "token"
        return DummyConnection()

    monkeypatch.setattr(database.libsql, "connect", fake_connect)

    db = database.BadKeysDB(
        db_url="libsql://example",
        auth_token="token",
        local_db_path=str(tmp_path / "remote.db"),
    )

    assert isinstance(db._conn, DummyConnection)
    assert db._conn.synced >= 1

    db._sync()
    assert db._conn.synced >= 2


def test_migration_recreates_response_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCursor:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, sql: str, _params: tuple | None = None) -> None:
            self.commands.append(sql)

        def fetchall(self) -> list[tuple]:
            return [(0, "old")]

        def fetchone(self):
            return None

    class DummyConnection:
        def __init__(self) -> None:
            self.cursor_obj = DummyCursor()

        def cursor(self) -> DummyCursor:
            return self.cursor_obj

        def commit(self) -> None:
            return None

        def sync(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_connect(_path: str, sync_url: str | None = None, auth_token: str | None = None):
        return DummyConnection()

    monkeypatch.setattr(database.libsql, "connect", fake_connect)

    db = database.BadKeysDB(local_db_path=str(tmp_path / "migrate.db"))

    commands = "\n".join(db._conn.cursor_obj.commands)
    assert "DROP TABLE message_response_data" in commands


def test_sync_with_remote_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyConnection:
        def sync(self) -> None:
            raise database.LIBSQL_ERROR("sync failed")

    db = database.BadKeysDB(local_db_path=":memory:")
    db._conn = DummyConnection()
    db.db_url = "libsql://example"
    db.auth_token = "token"

    db._sync()


def test_init_and_get_bad_keys_db(tmp_path: Path) -> None:
    database._bad_keys_state["instance"] = None

    instance = database.init_bad_keys_db()
    assert database.get_bad_keys_db() is instance

    database._bad_keys_state["instance"] = None
    instance2 = database.get_bad_keys_db()
    assert instance2 is not None
