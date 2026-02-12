from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import discord

if TYPE_CHECKING:
    from llmcord.services.extractors import TweetProtocol


@dataclass(slots=True)
class FakeUser:
    id: int

    @property
    def mention(self) -> str:
        return f"<@{self.id}>"


@dataclass(slots=True)
class FakeChannel:
    id: int = 123
    type: discord.ChannelType = discord.ChannelType.text


@dataclass(slots=True)
class FakeAttachment:
    url: str
    content_type: str | None
    filename: str = "file"


@dataclass(slots=True)
class FakeMessageReference:
    message_id: int | None = None
    cached_message: Any = None


@dataclass(slots=True)
class FakeMessage:
    id: int
    content: str
    author: FakeUser
    channel: FakeChannel = field(default_factory=FakeChannel)
    attachments: list[FakeAttachment] = field(default_factory=list)
    embeds: list[Any] = field(default_factory=list)
    components: list[Any] = field(default_factory=list)
    reference: FakeMessageReference | None = None
    type: discord.MessageType = discord.MessageType.default


class DummyTwitterApi:
    async def tweet_details(self, tweet_id: int) -> TweetProtocol | None:
        return None

    def tweet_replies(
        self,
        tweet_id: int,
        limit: int,
    ) -> AsyncIterator[TweetProtocol]:
        async def _gen() -> AsyncIterator[TweetProtocol]:
            if False:  # pragma: no cover
                yield cast("TweetProtocol", object())
            return

        return _gen()
