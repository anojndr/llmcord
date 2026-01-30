"""Response related views and buttons."""
from collections.abc import Awaitable, Callable, Mapping

import discord

from llmcord.discord.ui.constants import RETRY_RESPONSE_ID, VIEW_RESPONSE_BETTER_ID
from llmcord.discord.ui.metadata import has_grounding_data
from llmcord.discord.ui.sources_view import SourceButton, TavilySourceButton
from llmcord.discord.ui.utils import get_response_data, get_retry_handler, upload_to_textis


class RetryButton(discord.ui.Button):
    """Button to retry the generation."""

    def __init__(
        self,
        callback_fn: Callable[[], Awaitable[None]] | None = None,
        user_id: int | None = None,
    ) -> None:
        """Initialize a retry button for the given user."""
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Retry with stable model",
            emoji="🔄",
            custom_id=RETRY_RESPONSE_ID,
        )
        self.callback_fn = callback_fn
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        """Retry the generation when allowed for the requesting user."""
        response_data = None
        if self.callback_fn is None or self.user_id is None:
            response_data = get_response_data(interaction.message.id)
            self.user_id = response_data.request_user_id

        if not self.user_id or interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "❌ You cannot retry this message.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        if self.callback_fn:
            await self.callback_fn()
            return

        retry_handler = get_retry_handler()
        if not retry_handler:
            await interaction.followup.send(
                "❌ Retry is unavailable right now. Please try again later.",
                ephemeral=True,
            )
            return

        if not response_data:
            response_data = get_response_data(interaction.message.id)

        if not response_data.request_message_id or not response_data.request_user_id:
            await interaction.followup.send(
                "❌ Missing retry context for this message.",
                ephemeral=True,
            )
            return

        await retry_handler(
            interaction,
            response_data.request_message_id,
            response_data.request_user_id,
        )


class ViewResponseBetterButton(discord.ui.Button):
    """Button to upload and view the full response."""

    def __init__(self, full_response: str | None = None) -> None:
        super().__init__(
            label="View Response Better",
            style=discord.ButtonStyle.secondary,
            emoji="📄",
            custom_id=VIEW_RESPONSE_BETTER_ID,
        )
        self.full_response = full_response

    async def callback(self, interaction: discord.Interaction) -> None:
        """Upload the response to text.is and share the link."""
        await interaction.response.defer(ephemeral=True)

        full_response = self.full_response
        if not full_response:
            response_data = get_response_data(interaction.message.id)
            full_response = response_data.full_response

        if not full_response:
            await interaction.followup.send(
                "❌ Missing response content for this message.",
                ephemeral=True,
            )
            return

        paste_url = await upload_to_textis(full_response)

        if paste_url:
            embed = discord.Embed(
                title="View Response",
                description=(
                    "Your response has been uploaded for better viewing:\n\n"
                    f"**[Click here to view]({paste_url})**"
                ),
                color=discord.Color.green(),
            )
            embed.set_footer(text="Powered by text.is")
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send(
                "❌ Failed to upload response to text.is. Please try again later.",
                ephemeral=True,
            )


class ResponseView(discord.ui.View):
    """View with a button that uploads responses to text.is."""

    def __init__(
        self,
        full_response: str,
        metadata: object | None = None,
        tavily_metadata: Mapping[str, object] | None = None,
        retry_callback: Callable[[], Awaitable[None]] | None = None,
        user_id: int | None = None,
    ) -> None:
        """Initialize the response view and its optional buttons."""
        super().__init__(timeout=None)
        self.full_response = full_response
        self.metadata = metadata
        self.tavily_metadata = tavily_metadata

        # View Response Better button (always present)
        self.add_item(ViewResponseBetterButton(full_response))

        # Add Retry button if callback is provided
        if retry_callback and user_id:
            self.add_item(RetryButton(retry_callback, user_id))

        # Add Gemini grounding sources button if we have grounding metadata
        if metadata and has_grounding_data(metadata):
            self.add_item(SourceButton(metadata))

        # Add Tavily sources button if we have tavily metadata with URLs or queries
        if tavily_metadata and (
            tavily_metadata.get("urls") or tavily_metadata.get("queries")
        ):
            self.add_item(TavilySourceButton(tavily_metadata))


class PersistentResponseView(discord.ui.View):
    """Persistent view to handle response buttons after restarts."""

    def __init__(self) -> None:
        super().__init__(timeout=None)
        self.add_item(ViewResponseBetterButton())
        self.add_item(RetryButton())
        self.add_item(SourceButton())
        self.add_item(TavilySourceButton())


class TextDisplay(discord.ui.Button):
    """A button that displays text content (simulation)."""

    def __init__(self, content: str) -> None:
        """Initialize the text display button."""
        super().__init__(
            label=content[:80] if content else "...",
            style=discord.ButtonStyle.secondary,
            disabled=True,
        )
        self.content = content
        self.type = (
            discord.ComponentType.text_display
            if hasattr(discord.ComponentType, "text_display")
            else discord.ComponentType.button
        )


class LayoutView(discord.ui.View):
    """View to hold text displays."""

    def __init__(self) -> None:
        """Initialize the layout view."""
        super().__init__(timeout=None)
