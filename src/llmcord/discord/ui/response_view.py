"""Response related views and buttons."""

from collections.abc import Awaitable, Callable, Mapping

import discord

from llmcord.discord.ui.constants import (
    RETRY_RESPONSE_ID,
    SHOW_FAILED_URLS_ID,
    SHOW_THOUGHT_PROCESS_ID,
    VIEW_RESPONSE_BETTER_ID,
)
from llmcord.discord.ui.metadata import has_grounding_data
from llmcord.discord.ui.sources_view import SourceButton, TavilySourceButton
from llmcord.discord.ui.utils import (
    build_error_embed,
    get_response_data,
    get_retry_handler,
    upload_to_rentry,
)


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
            label="Retry",
            emoji="🔄",
            custom_id=RETRY_RESPONSE_ID,
        )
        self.callback_fn = callback_fn
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        """Retry the generation when allowed for the requesting user."""
        response_data = None
        if (self.callback_fn is None or self.user_id is None) and interaction.message:
            response_data = get_response_data(interaction.message.id)
            self.user_id = response_data.request_user_id

        if not self.user_id or interaction.user.id != self.user_id:
            await interaction.response.send_message(
                embed=build_error_embed("You can only retry your own message."),
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
                embed=build_error_embed(
                    "Retry is unavailable right now. Please try again later.",
                ),
                ephemeral=True,
            )
            return

        if not response_data and interaction.message:
            response_data = get_response_data(interaction.message.id)

        if (
            not response_data
            or not response_data.request_message_id
            or not response_data.request_user_id
        ):
            await interaction.followup.send(
                embed=build_error_embed(
                    "Missing retry context for this message.",
                ),
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
        """Initialize the button with optional response content."""
        super().__init__(
            label="View Response Better",
            style=discord.ButtonStyle.secondary,
            emoji="📄",
            custom_id=VIEW_RESPONSE_BETTER_ID,
        )
        self.full_response = full_response

    async def callback(self, interaction: discord.Interaction) -> None:
        """Upload the response to rentry.co and share the link."""
        await interaction.response.defer(ephemeral=True)

        full_response = self.full_response
        if not full_response and interaction.message:
            response_data = get_response_data(interaction.message.id)
            full_response = response_data.full_response

        if not full_response:
            await interaction.followup.send(
                embed=build_error_embed(
                    "Missing response content for this message.",
                ),
                ephemeral=True,
            )
            return

        paste_url = await upload_to_rentry(full_response)

        if paste_url:
            embed = discord.Embed(
                title="View Response",
                description=(
                    "Your response has been uploaded for better viewing:\n\n"
                    f"**[Click here to view]({paste_url})**"
                ),
                color=discord.Color.green(),
            )
            embed.set_footer(text="Powered by rentry.co")
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send(
                embed=build_error_embed(
                    ("Failed to upload response to rentry.co. Please try again later."),
                ),
                ephemeral=True,
            )


class ShowThoughtProcessButton(discord.ui.Button):
    """Button to reveal hidden thought process for providers that expose it."""

    def __init__(self, thought_process: str | None = None) -> None:
        """Initialize button with optional thought process content."""
        super().__init__(
            label="Show Thought Process",
            style=discord.ButtonStyle.secondary,
            emoji="🧠",
            custom_id=SHOW_THOUGHT_PROCESS_ID,
        )
        self.thought_process = thought_process

    async def callback(self, interaction: discord.Interaction) -> None:
        """Reveal thought process to the requesting user."""
        await interaction.response.defer(ephemeral=True)

        thought_process = self.thought_process
        if not thought_process and interaction.message:
            response_data = get_response_data(interaction.message.id)
            thought_process = response_data.thought_process

        if not thought_process:
            await interaction.followup.send(
                embed=build_error_embed(
                    "No thought process is available for this response.",
                ),
                ephemeral=True,
            )
            return

        paste_url = await upload_to_rentry(thought_process)
        if paste_url:
            embed = discord.Embed(
                title="Thought Process",
                description=(
                    "The hidden thought process has been uploaded for viewing:\n\n"
                    f"**[Click here to view]({paste_url})**"
                ),
                color=discord.Color.green(),
            )
            embed.set_footer(text="Powered by rentry.co")
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        await interaction.followup.send(
            embed=build_error_embed(
                "Failed to show thought process. Please try again later.",
            ),
            ephemeral=True,
        )


class FailedUrlsButton(discord.ui.Button):
    """Button to reveal failed URL extractions."""

    def __init__(self, failed_extractions: list[str] | None = None) -> None:
        """Initialize button with optional failed extraction details."""
        super().__init__(
            label="failed urls",
            style=discord.ButtonStyle.secondary,
            emoji="⚠️",
            custom_id=SHOW_FAILED_URLS_ID,
        )
        self.failed_extractions = failed_extractions

    async def callback(self, interaction: discord.Interaction) -> None:
        """Reveal failed URLs to the requesting user."""
        await interaction.response.defer(ephemeral=True)

        failed_extractions = self.failed_extractions
        if failed_extractions is None and interaction.message:
            response_data = get_response_data(interaction.message.id)
            failed_extractions = response_data.failed_extractions

        if not failed_extractions:
            await interaction.followup.send(
                embed=build_error_embed(
                    "No failed URLs are available for this response.",
                ),
                ephemeral=True,
            )
            return

        lines = [f"- {url}" for url in failed_extractions]
        max_description_length = 3800
        description = ""
        for index, line in enumerate(lines):
            next_description = f"{description}\n{line}" if description else line
            if len(next_description) <= max_description_length:
                description = next_description
                continue

            remaining = len(lines) - index
            if remaining > 0:
                suffix = f"\n- ... and {remaining} more"
                if len(description) + len(suffix) <= max_description_length:
                    description += suffix
            break

        embed = discord.Embed(
            title="Failed URLs",
            description=description,
            color=discord.Color.orange(),
        )
        await interaction.followup.send(embed=embed, ephemeral=True)


class ResponseView(discord.ui.View):
    """View with a button that uploads responses to rentry.co."""

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

        # View Response Better button (only if response exists)
        if full_response:
            self.add_item(ViewResponseBetterButton(full_response))

        # Add Retry button if callback is provided
        if retry_callback and user_id:
            self.add_item(RetryButton(retry_callback, user_id))

        # Add Gemini grounding sources button if we have grounding metadata
        if metadata and has_grounding_data(metadata):
            self.add_item(SourceButton(metadata))

        # Add Tavily sources button if we have metadata with URLs or queries.
        if tavily_metadata and (
            tavily_metadata.get("urls") or tavily_metadata.get("queries")
        ):
            self.add_item(TavilySourceButton(tavily_metadata))


class PersistentResponseView(discord.ui.View):
    """Persistent view to handle response buttons after restarts."""

    def __init__(self) -> None:
        """Initialize persistent response buttons."""
        super().__init__(timeout=None)
        self.add_item(ViewResponseBetterButton())
        self.add_item(ShowThoughtProcessButton())
        self.add_item(RetryButton())
        self.add_item(SourceButton())
        self.add_item(TavilySourceButton())
        self.add_item(FailedUrlsButton())


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
        if hasattr(discord.ComponentType, "text_display"):
            self._underlying.type = discord.ComponentType.text_display  # type: ignore[attr-defined, misc, assignment]


class LayoutView(discord.ui.View):
    """View to hold text displays."""

    def __init__(self) -> None:
        """Initialize the layout view."""
        super().__init__(timeout=None)
