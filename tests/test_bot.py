from __future__ import annotations

import bot


def test_get_channel_locked_model(monkeypatch) -> None:
    monkeypatch.setattr(
        bot,
        "config",
        {
            "channel_model_overrides": {
                "123": "model-a",
                456: "model-b",
            },
        },
    )

    assert bot.get_channel_locked_model(123) == "model-a"
    assert bot.get_channel_locked_model(456) == "model-b"
    assert bot.get_channel_locked_model(789) is None


def test_build_model_autocomplete() -> None:
    config_data = {
        "models": {
            "model-a": {},
            "model-b": {},
        },
    }

    handlers = bot.ModelAutocompleteHandlers(
        get_current=lambda _: "model-a",
        get_default=lambda: "model-a",
    )

    choices = bot._build_model_autocomplete(
        curr_str="model",
        handlers=handlers,
        user_id="123",
        config_data=config_data,
    )

    assert choices
    assert choices[0].value == "model-a"
    assert any(choice.value == "model-b" for choice in choices)
