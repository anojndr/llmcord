from __future__ import annotations

import logging
from types import SimpleNamespace

from llmcord import entrypoint
from llmcord.discord import events
from llmcord.services import database as database_module

STARTUP_STARTED_ATTR = "_llmcord_startup_started_at"
TWITTER_INIT_TASK_ATTR = "_llmcord_twitter_init_task"
TWITTER_INIT_FUNC_NAME = "_init_twitter_accounts"
DB_STATE_NAME = "_db_state"


class _FakeUser:
    def __init__(self, user_id: int, display_name: str) -> None:
        self.id = user_id
        self._display_name = display_name

    def __str__(self) -> str:
        return self._display_name


class _FakeTree:
    def __init__(self, synced_commands: list[object]) -> None:
        self._synced_commands = synced_commands

    async def sync(self) -> list[object]:
        return self._synced_commands


class _FakeDiscordBot:
    def __init__(
        self,
        *,
        user: _FakeUser,
        guild_count: int = 0,
        synced_commands: list[object] | None = None,
        application_id: int | None = None,
    ) -> None:
        self.user = user
        self.guilds = [object() for _ in range(guild_count)]
        self.tree = _FakeTree(synced_commands or [])
        self.application_id = application_id
        self.views: list[object] = []
        self.started_with: str | None = None

    def add_view(self, view: object) -> None:
        self.views.append(view)

    async def start(self, token: str) -> None:
        self.started_with = token


class _FakeTask:
    def done(self) -> bool:
        return False


class _FakeTwitterPool:
    def __init__(self) -> None:
        self.added_usernames: list[str] = []
        self.login_calls = 0

    async def get_account(self, username: str) -> object | None:
        if username == "existing-user":
            return object()
        return None

    async def add_account(
        self,
        username: str,
        _password: str,
        _email: str,
        _email_password: str,
        *,
        cookies: object | None = None,
    ) -> None:
        del cookies
        self.added_usernames.append(username)

    async def login_all(self) -> None:
        self.login_calls += 1


def _perf_counter(values: list[float]):
    iterator = iter(values)
    return lambda: next(iterator)


async def test_entrypoint_main_logs_startup_sequence(
    monkeypatch,
    caplog,
) -> None:
    calls: list[str] = []
    fake_bot = _FakeDiscordBot(user=_FakeUser(123, "TestBot"))

    async def fake_init_db() -> None:
        calls.append("db")

    async def fake_start_server() -> None:
        calls.append("server")

    async def fake_shutdown() -> None:
        calls.append("shutdown")

    monkeypatch.setattr(
        entrypoint,
        "preload_runtime_dependencies",
        lambda: calls.append("preload"),
    )
    monkeypatch.setattr(entrypoint, "init_db", fake_init_db)
    monkeypatch.setattr(entrypoint, "start_server", fake_start_server)
    monkeypatch.setattr(entrypoint, "shutdown", fake_shutdown)
    monkeypatch.setattr(entrypoint, "discord_bot", fake_bot)
    monkeypatch.setattr(entrypoint, "config", {"bot_token": "TOKEN"})
    monkeypatch.setattr(entrypoint, "discord_voice_supported", False)
    monkeypatch.setattr(entrypoint.time, "perf_counter", _perf_counter([1.0, 2.0, 2.6]))

    caplog.set_level(logging.INFO, logger="llmcord.entrypoint")

    await entrypoint.main()

    assert calls == ["preload", "db", "server", "shutdown"]
    assert fake_bot.started_with == "TOKEN"
    assert "Starting llmcord startup sequence" in caplog.text
    assert (
        "Discord voice transport is unavailable because PyNaCl is not installed; "
        "text commands, slash commands, and attachment handling are unaffected"
        in caplog.text
    )
    assert "Runtime dependency preload completed in 0.60s" in caplog.text
    assert (
        "Starting Discord client login with the configured bot token and "
        "waiting for the ready event" in caplog.text
    )


async def test_on_ready_logs_detailed_startup_messages(
    monkeypatch,
    caplog,
) -> None:
    fake_bot = _FakeDiscordBot(
        user=_FakeUser(123, "TestBot"),
        guild_count=2,
        synced_commands=[object(), object(), object()],
    )
    setattr(fake_bot, STARTUP_STARTED_ATTR, 100.0)
    scheduled: dict[str, object] = {}
    retry_handlers: list[object] = []
    view_sentinel = object()

    def fake_create_task(coro, *, name: str) -> _FakeTask:
        coro.close()
        scheduled["name"] = name
        task = _FakeTask()
        scheduled["task"] = task
        return task

    monkeypatch.setattr(events, "discord_bot", fake_bot)
    monkeypatch.setattr(
        events,
        "config",
        {
            "twitter_accounts": [
                {
                    "username": "new-user",
                    "password": "pw",
                    "email": "user@example.com",
                    "email_password": "email-pw",
                },
                {
                    "password": "missing-username",
                },
            ],
            "twitter_login_timeout_seconds": 45,
        },
    )
    monkeypatch.setattr(events, "PersistentResponseView", lambda: view_sentinel)
    monkeypatch.setattr(events, "set_retry_handler", retry_handlers.append)
    monkeypatch.setattr(events.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        events.time,
        "perf_counter",
        _perf_counter([101.5, 102.0, 102.4]),
    )

    caplog.set_level(logging.INFO, logger="llmcord.discord.events")

    await events.on_ready()

    assert fake_bot.views == [view_sentinel]
    assert len(retry_handlers) == 1
    assert callable(retry_handlers[0])
    assert scheduled["name"] == "llmcord-twitter-init"
    assert getattr(fake_bot, TWITTER_INIT_TASK_ATTR) is scheduled["task"]
    assert (
        "Discord client ready as TestBot (123); connected to 2 guild(s) after "
        "1.50s of startup. Syncing application commands." in caplog.text
    )
    assert (
        "Discord startup tasks completed in 0.40s: synced 3 application "
        "command(s), registered persistent response controls, and installed the "
        "retry handler" in caplog.text
    )
    assert (
        "Bot invite URL (application_id=123, permissions=412317191168): "
        "https://discord.com/oauth2/authorize?client_id=123&permissions="
        "412317191168&scope=bot" in caplog.text
    )
    assert (
        "Queued background Twitter/X initialization for 2 configured account(s) "
        "(usable=1, login timeout=45.0s)" in caplog.text
    )


async def test_init_twitter_accounts_logs_summary_counts(
    monkeypatch,
    caplog,
) -> None:
    fake_pool = _FakeTwitterPool()

    monkeypatch.setattr(events, "twitter_api", SimpleNamespace(pool=fake_pool))
    monkeypatch.setattr(
        events,
        "config",
        {
            "twitter_accounts": [
                {
                    "username": "existing-user",
                    "password": "pw1",
                    "email": "existing@example.com",
                    "email_password": "email-pw1",
                },
                {
                    "username": "new-user",
                    "password": "pw2",
                    "email": "new@example.com",
                    "email_password": "email-pw2",
                },
                {
                    "password": "missing-username",
                },
            ],
            "twitter_login_timeout_seconds": 60,
        },
    )
    monkeypatch.setattr(events.time, "perf_counter", _perf_counter([10.0, 10.75]))

    caplog.set_level(logging.INFO, logger="llmcord.discord.events")

    await getattr(events, TWITTER_INIT_FUNC_NAME)()

    assert fake_pool.added_usernames == ["new-user"]
    assert fake_pool.login_calls == 1
    assert (
        "Starting Twitter/X account initialization (configured=3, usable=2, "
        "skipped_missing_username=1, timeout=60.0s)" in caplog.text
    )
    assert (
        "Twitter/X account initialization finished in 0.75s (configured=3, "
        "usable=2, skipped_missing_username=1, already_loaded=1, added=1, "
        "skipped_incomplete_credentials=0)" in caplog.text
    )


async def test_init_db_logs_absolute_ready_path(
    tmp_path,
    caplog,
) -> None:
    db_path = tmp_path / "llmcord.db"

    caplog.set_level(logging.INFO, logger="llmcord.services.database")

    instance = await database_module.init_db(str(db_path))
    try:
        assert (
            f"SQLite database ready at {db_path.resolve()} "
            "(connection opened, schema initialized)" in caplog.text
        )
    finally:
        await instance.aclose()
        getattr(database_module, DB_STATE_NAME)["instance"] = None
