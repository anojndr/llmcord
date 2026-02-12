from __future__ import annotations

import llmcord.discord.commands  # noqa: F401
from llmcord.globals import discord_bot


def test_slash_commands_registered() -> None:
    names = {cmd.name for cmd in discord_bot.tree.get_commands()}
    assert {"model", "searchdecidermodel", "resetallpreferences"}.issubset(names)
