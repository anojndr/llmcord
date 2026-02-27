from __future__ import annotations

import llmcord.__main__ as main_module

EXPECTED_RUN_CALLS = 2


def test_main_suppresses_secondary_keyboard_interrupt(monkeypatch) -> None:
    run_calls: list[int] = []

    class _FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def get_loop(self):
            return object()

        def run(self, coro):
            run_calls.append(1)
            coro.close()
            raise KeyboardInterrupt

    async def _fake_main() -> None:
        return None

    async def _fake_shutdown() -> None:
        return None

    monkeypatch.setattr(main_module.asyncio, "Runner", _FakeRunner)
    monkeypatch.setattr(main_module, "install_global_exception_hooks", lambda: None)
    monkeypatch.setattr(
        main_module,
        "register_asyncio_exception_handler",
        lambda _loop: None,
    )
    monkeypatch.setattr(main_module.llmcord.entrypoint, "main", _fake_main)
    monkeypatch.setattr(main_module.llmcord.entrypoint, "shutdown", _fake_shutdown)

    main_module.main()

    assert len(run_calls) == EXPECTED_RUN_CALLS
