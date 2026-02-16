"""Source related views and buttons."""

from collections.abc import Mapping

import discord

from llmcord.discord.ui.constants import (
    GROUNDING_SOURCES_ID,
    TAVILY_SOURCES_ID,
    URL_MAX_LENGTH,
)
from llmcord.discord.ui.embed_limits import call_with_embed_limits, enforce_embed_limits
from llmcord.discord.ui.metadata import (
    add_chunked_embed_field,
    build_grounding_sources_embed,
)
from llmcord.discord.ui.utils import build_error_embed, get_response_data


class SourceButton(discord.ui.Button):
    """Button to show sources from grounding metadata."""

    def __init__(self, metadata: object | None = None) -> None:
        """Initialize the sources button."""
        super().__init__(
            label="Show Sources",
            style=discord.ButtonStyle.secondary,
            custom_id=GROUNDING_SOURCES_ID,
        )
        self.metadata = metadata

    async def callback(self, interaction: discord.Interaction) -> None:
        """Show sources in an ephemeral embed."""
        metadata = self.metadata
        if metadata is None and interaction.message:
            response_data = get_response_data(interaction.message.id)
            metadata = response_data.grounding_metadata

        if not metadata:
            await call_with_embed_limits(
                interaction.response.send_message,
                embed=build_error_embed(
                    "No source information available for this message.",
                ),
                ephemeral=True,
            )
            return

        embed = build_grounding_sources_embed(metadata)
        await call_with_embed_limits(
            interaction.response.send_message,
            embed=embed,
            ephemeral=True,
        )


class TavilySourcesView(discord.ui.View):
    """Paginated view for displaying web search sources.

    Supports Tavily and Exa.
    """

    # Limits for embed content
    MAX_EMBED_SIZE = 5500  # Leave buffer below 6000
    FIELD_LIMIT = 1024
    SOURCES_PER_PAGE = 10  # Reasonable default

    def __init__(self, search_metadata: Mapping[str, object]) -> None:
        """Initialize the paginated sources view."""
        super().__init__(timeout=300)  # 5 minute timeout
        # Handle None or malformed metadata
        self.search_metadata = dict(search_metadata or {})
        self.current_page = 0

        # Detect provider from metadata (defaults to "tavily" for compatibility)
        self.provider = self.search_metadata.get("provider", "tavily")

        # Prepare source entries with defensive defaults
        raw_queries = self.search_metadata.get("queries")
        if isinstance(raw_queries, list):
            self.queries = [str(query) for query in raw_queries if query]
        else:
            self.queries = []

        raw_urls = self.search_metadata.get("urls")
        if isinstance(raw_urls, list):
            self.urls = [
                dict(url_info) for url_info in raw_urls if isinstance(url_info, dict)
            ]
        else:
            self.urls = []
        self.sources = self._prepare_sources()
        self.pages = self._paginate_sources()
        self.total_pages = len(self.pages)

        # Update button states
        self.update_buttons()

    def _prepare_sources(self) -> list[str]:
        """Prepare formatted source strings."""
        sources = []
        for i, url_info in enumerate(self.urls):
            title = str(url_info.get("title", "No title"))[:80]
            url = str(url_info.get("url", ""))
            # Truncate very long URLs
            if len(url) > URL_MAX_LENGTH:
                url = f"{url[:URL_MAX_LENGTH]}..."
            sources.append(f"{i + 1}. [{title}]({url})")
        return sources

    def _paginate_sources(self) -> list[list[str]]:
        """Split sources into pages that fit within embed limits."""
        if not self.sources:
            return [[]]

        pages: list[list[str]] = []
        current_page: list[str] = []
        current_size = 0

        # Account for queries field size (on every page)
        queries_size = 0
        if self.queries:
            queries_text = "\n".join(f"• {q}" for q in self.queries)
            queries_text = queries_text[: self.FIELD_LIMIT]
            queries_size = len("Search Queries") + len(queries_text) + 50

        base_size = len("Web Search Sources") + queries_size + 100

        for source in self.sources:
            source_size = len(source) + 2  # +2 for newline

            # Check if adding this source would exceed limits
            if (
                current_size + source_size > self.MAX_EMBED_SIZE - base_size
                or len(current_page) >= self.SOURCES_PER_PAGE
            ):
                if current_page:
                    pages.append(current_page)
                current_page = [source]
                current_size = source_size
            else:
                current_page.append(source)
                current_size += source_size

        if current_page:
            pages.append(current_page)

        return pages if pages else [[]]

    def update_buttons(self) -> None:
        """Update button disabled states based on current page."""
        self.prev_button.disabled = self.current_page <= 0
        self.next_button.disabled = self.current_page >= self.total_pages - 1

    def build_embed(self) -> discord.Embed:
        """Build the embed for the current page."""
        embed = discord.Embed(
            title="Web Search Sources",
            color=discord.Color.blue(),
        )

        # Display search queries (on every page)
        if self.queries:
            queries_text = "\n".join(f"• {q}" for q in self.queries)
            queries_text = queries_text[: self.FIELD_LIMIT]
            embed.add_field(
                name="Search Queries",
                value=queries_text,
                inline=False,
            )

        # Display sources for current page
        if self.pages and self.pages[self.current_page]:
            page_sources = self.pages[self.current_page]

            # Split into multiple fields if needed
            add_chunked_embed_field(
                embed,
                page_sources,
                "Sources",
                self.FIELD_LIMIT,
            )
        elif not self.urls:
            embed.add_field(
                name="Sources",
                value="No URLs available",
                inline=False,
            )

        # Footer with pagination info - show provider name
        mode = self.search_metadata.get("mode")
        if self.provider == "exa":
            provider_name = "Exa Search"
        elif mode == "research":
            provider_name = "Tavily Research"
        else:
            provider_name = "Tavily Search"
        footer_text = (
            f"Page {self.current_page + 1}/{self.total_pages} • "
            f"{len(self.urls)} total sources • Powered by {provider_name}"
        )
        embed.set_footer(text=footer_text)

        return enforce_embed_limits(embed)

    @discord.ui.button(
        label="Previous",
        style=discord.ButtonStyle.secondary,
        emoji="◀️",
        disabled=True,
    )
    async def prev_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button,
    ) -> None:
        """Show the previous page of sources."""
        self.current_page = max(0, self.current_page - 1)
        self.update_buttons()
        await call_with_embed_limits(
            interaction.response.edit_message,
            embed=self.build_embed(),
            view=self,
        )

    @discord.ui.button(
        label="Next",
        style=discord.ButtonStyle.secondary,
        emoji="▶️",
    )
    async def next_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button,
    ) -> None:
        """Show the next page of sources."""
        self.current_page = min(self.total_pages - 1, self.current_page + 1)
        self.update_buttons()
        await call_with_embed_limits(
            interaction.response.edit_message,
            embed=self.build_embed(),
            view=self,
        )

    @discord.ui.button(
        label="Go to Page",
        style=discord.ButtonStyle.primary,
        emoji="🔢",
    )
    async def goto_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button,
    ) -> None:
        """Open a modal to select a page."""
        modal = GoToPageModal(self)
        await interaction.response.send_modal(modal)


class GoToPageModal(discord.ui.Modal, title="Go to Page"):
    """Modal for entering a specific page number."""

    page_input: discord.ui.TextInput = discord.ui.TextInput(
        label="Page Number",
        placeholder="Enter page number...",
        required=True,
        min_length=1,
        max_length=5,
    )

    def __init__(self, sources_view: TavilySourcesView) -> None:
        """Initialize the modal for the specified sources view."""
        super().__init__()
        self.sources_view = sources_view
        self.page_input.placeholder = f"Enter 1-{sources_view.total_pages}"

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Validate the page number and update the embed."""
        try:
            page_num = int(self.page_input.value)
            if 1 <= page_num <= self.sources_view.total_pages:
                # Convert to 0-indexed.
                self.sources_view.current_page = page_num - 1
                self.sources_view.update_buttons()
                await call_with_embed_limits(
                    interaction.response.edit_message,
                    embed=self.sources_view.build_embed(),
                    view=self.sources_view,
                )
            else:
                await call_with_embed_limits(
                    interaction.response.send_message,
                    embed=build_error_embed(
                        (
                            "Invalid page number. Please enter a number "
                            f"between 1 and {self.sources_view.total_pages}."
                        ),
                    ),
                    ephemeral=True,
                )
        except ValueError:
            await call_with_embed_limits(
                interaction.response.send_message,
                embed=build_error_embed("Please enter a valid number."),
                ephemeral=True,
            )


class TavilySourceButton(discord.ui.Button):
    """Button to show sources from web search (supports Tavily and Exa)."""

    def __init__(
        self,
        search_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the web-search sources button."""
        super().__init__(
            label="Show Sources",
            style=discord.ButtonStyle.secondary,
            emoji="🔍",
            custom_id=TAVILY_SOURCES_ID,
        )
        self.search_metadata = search_metadata

    async def callback(self, interaction: discord.Interaction) -> None:
        """Show web-search sources in a paginated embed."""
        search_metadata = self.search_metadata
        if search_metadata is None and interaction.message:
            response_data = get_response_data(interaction.message.id)
            search_metadata = response_data.tavily_metadata

        if not search_metadata:
            await call_with_embed_limits(
                interaction.response.send_message,
                embed=build_error_embed(
                    "No web search sources available for this message.",
                ),
                ephemeral=True,
            )
            return

        view = TavilySourcesView(search_metadata)
        embed = view.build_embed()
        await call_with_embed_limits(
            interaction.response.send_message,
            embed=embed,
            view=view,
            ephemeral=True,
        )


class SourceView(discord.ui.View):
    """Legacy view for backwards compatibility - now uses ResponseView."""

    def __init__(self, metadata: object) -> None:
        """Initialize the legacy sources view."""
        super().__init__(timeout=None)
        self.metadata = metadata

    @discord.ui.button(
        label="Show Sources",
        style=discord.ButtonStyle.secondary,
        custom_id=GROUNDING_SOURCES_ID,
    )
    async def show_sources(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button,
    ) -> None:
        """Show sources for legacy responses."""
        metadata = self.metadata
        if metadata is None and interaction.message:
            response_data = get_response_data(interaction.message.id)
            metadata = response_data.grounding_metadata

        if not metadata:
            await call_with_embed_limits(
                interaction.response.send_message,
                embed=build_error_embed(
                    "No source information available for this message.",
                ),
                ephemeral=True,
            )
            return

        embed = build_grounding_sources_embed(metadata)
        await call_with_embed_limits(
            interaction.response.send_message,
            embed=embed,
            ephemeral=True,
        )
