from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import discord
import pytest

from llmcord.discord.ui.response_view import ShowThoughtProcessButton
from llmcord.logic.discord_ui import update_response_view
from llmcord.logic.generation_types import GenerationContext, GenerationState


class _FakeMessage:
    def __init__(self) -> None:
        self.last_view: discord.ui.View | None = None

    async def edit(self, **kwargs: object) -> None:
        view = kwargs.get("view")
        if isinstance(view, discord.ui.View):
            self.last_view = view


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("thought_process", "expect_button"),
    [
        ("", False),
        ("hidden chain of thought", True),
    ],
)
async def test_update_response_view_thought_button_visibility(
    thought_process: str,
    *,
    expect_button: bool,
) -> None:
    fake_message = _FakeMessage()
    state = GenerationState(
        response_msgs=cast("list[discord.Message]", [fake_message]),
        response_contents=["Visible answer"],
        input_tokens=0,
        max_message_length=4096,
        embed=discord.Embed(description="Visible answer"),
        grounding_metadata=None,
        last_edit_time=0.0,
        generated_images=[],
        generated_image_hashes=set(),
        display_model="google-gemini-cli/gemini-3-flash-preview",
        thought_process=thought_process,
    )
    context = cast(
        "GenerationContext",
        SimpleNamespace(
            tavily_metadata=None,
            retry_callback=None,
            new_msg=SimpleNamespace(author=SimpleNamespace(id=12345)),
        ),
    )

    await update_response_view(
        context=context,
        state=state,
        full_response="Visible answer",
        grounding_metadata=None,
    )

    assert fake_message.last_view is not None
    has_button = any(
        isinstance(child, ShowThoughtProcessButton)
        for child in fake_message.last_view.children
    )
    assert has_button is expect_button
