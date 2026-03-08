from __future__ import annotations

from llmcord import entrypoint


def test_preload_runtime_dependencies_warms_hot_paths(
    monkeypatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "llmcord.processing.preload_runtime_dependencies",
        lambda: calls.append("processing"),
    )
    monkeypatch.setattr(
        "llmcord.services.search.decider.preload_runtime_dependencies",
        lambda: calls.append("search"),
    )

    entrypoint.preload_runtime_dependencies()

    assert calls == ["processing", "search"]
