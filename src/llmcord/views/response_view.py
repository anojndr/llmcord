"""Response view and web-search source views."""

import logging
from collections.abc import Awaitable, Callable, Mapping

import discord
import httpx
from bs4 import BeautifulSoup

from llmcord.config import get_config, get_or_create_httpx_client

from .grounding import SourceButton, _has_grounding_data, add_chunked_embed_field

LOGGER = logging.getLogger(__name__)

# Shared httpx clients for text.is uploads - uses factory pattern for DRY
_textis_proxy_client_holder: list[httpx.AsyncClient] = []
_textis_direct_client_holder: list[httpx.AsyncClient] = []
HTTP_OK = 200
URL_MAX_LENGTH = 150


def _get_textis_client(proxy_url: str | None = None) -> httpx.AsyncClient:
    """Get or create the shared text.is httpx client using the DRY factory pattern."""
    client_holder = (
        _textis_proxy_client_holder if proxy_url else _textis_direct_client_holder
    )
    return get_or_create_httpx_client(
        client_holder,
        timeout=30.0,
        connect_timeout=10.0,
        max_connections=10,
        max_keepalive=5,
        proxy_url=proxy_url,
        follow_redirects=False,  # text.is needs redirect handling
    )


async def _upload_to_textis_with_client(
    client: httpx.AsyncClient,
    text: str,
) -> str | None:
    """Upload text to text.is using the provided client."""
    # Get the CSRF token from the main page
    response = await client.get("https://text.is/", timeout=30)

    # Extract CSRF token from the form
    soup = BeautifulSoup(response.text, "lxml")  # lxml is faster than html.parser
    csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if not csrf_input:
        # Debug: log a snippet of the response to help diagnose
        LOGGER.error("Could not find CSRF token on text.is")
        LOGGER.debug(
            "Response status: %s, Content preview: %s",
            response.status_code,
            response.text[:500],
        )
        return None

    csrf_token = csrf_input.get("value")

    # Get cookies from the response
    cookies = response.cookies

    # POST the text content
    form_data = {
        "csrfmiddlewaretoken": csrf_token,
        "text": text,
    }

    headers = {
        "Referer": "https://text.is/",
        "Origin": "https://text.is",
    }

    post_response = await client.post(
        "https://text.is/",
        data=form_data,
        headers=headers,
        cookies=cookies,
        timeout=30,
    )

    # The response should be a redirect (302) to the paste URL
    if post_response.status_code in (301, 302, 303, 307, 308):
        paste_url = post_response.headers.get("Location")
        if paste_url:
            # Handle relative URLs
            if paste_url.startswith("/"):
                paste_url = f"https://text.is{paste_url}"
            return paste_url

    # If we got a 200, the paste might have been created and we're on the page
    if post_response.status_code == HTTP_OK:
        # Check if the URL changed (we might be on the paste page)
        final_url = str(post_response.url)
        if final_url != "https://text.is/" and "text.is/" in final_url:
            return final_url

    LOGGER.error(
        "Unexpected response from text.is: %s",
        post_response.status_code,
    )

    return None


async def upload_to_textis(text: str) -> str | None:
    """Upload text to text.is and return the paste URL.

    Returns None if upload fails.
    """
    # Proxy configuration from config (optional)
    config = get_config()
    proxy_url = config.get("proxy_url") or None

    try:
        client = _get_textis_client(proxy_url)
        return await _upload_to_textis_with_client(client, text)
    except httpx.ConnectError as exc:
        if proxy_url:
            LOGGER.warning(
                "Proxy connection to text.is failed; retrying without proxy: %s",
                exc,
            )
            try:
                direct_client = _get_textis_client(None)
                return await _upload_to_textis_with_client(direct_client, text)
            except Exception:
                LOGGER.exception("Error uploading to text.is without proxy")
                return None

        LOGGER.exception("Error uploading to text.is")
    except Exception:
        LOGGER.exception("Error uploading to text.is")

    return None


class RetryButton(discord.ui.Button):
    """Button to retry the generation."""

    def __init__(
        self,
        callback_fn: Callable[[], Awaitable[None]],
        user_id: int,
    ) -> None:
        """Initialize a retry button for the given user."""
        super().__init__(
            style=discord.ButtonStyle.secondary,
            label="Retry",
            emoji="🔄",
            custom_id="llmcord:retry",
        )
        self.callback_fn = callback_fn
        self.user_id = user_id

    async def callback(self, interaction: discord.Interaction) -> None:
        """Retry the generation when allowed for the requesting user."""
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "❌ You cannot retry this message.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()
        await self.callback_fn()


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

        # Add Retry button if callback is provided
        if retry_callback and user_id:
            self.add_item(RetryButton(retry_callback, user_id))

        # Add Gemini grounding sources button if we have grounding metadata
        if metadata and _has_grounding_data(metadata):
            self.add_item(SourceButton(metadata))

        # Add Tavily sources button if we have tavily metadata with URLs or queries
        if tavily_metadata and (
            tavily_metadata.get("urls") or tavily_metadata.get("queries")
        ):
            self.add_item(TavilySourceButton(tavily_metadata))

    @discord.ui.button(
        label="View Response Better",
        style=discord.ButtonStyle.secondary,
        emoji="📄",
        custom_id="llmcord:view_response_better",
    )
    async def view_response_better(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button,
    ) -> None:
        """Upload the response to text.is and share the link."""
        await interaction.response.defer(ephemeral=True)

        if not self.full_response:
            await interaction.followup.send(
                "❌ No stored response content is available for this message.",
                ephemeral=True,
            )
            return

        paste_url = await upload_to_textis(self.full_response)

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


class TavilySourceButton(discord.ui.Button):
    """Button to show sources from web search (supports Tavily and Exa)."""

    def __init__(self, search_metadata: Mapping[str, object]) -> None:
        """Initialize the web-search sources button."""
        super().__init__(
            label="Show Sources",
            style=discord.ButtonStyle.secondary,
            emoji="🔍",
            custom_id="llmcord:web_sources",
        )
        self.search_metadata = search_metadata

    async def callback(self, interaction: discord.Interaction) -> None:
        """Show web-search sources in a paginated embed."""
        view = TavilySourcesView(self.search_metadata)
        embed = view.build_embed()
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


class TavilySourcesView(discord.ui.View):
    """Paginated view for displaying web search sources (supports Tavily and Exa)."""

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
        self.queries = self.search_metadata.get("queries") or []
        self.urls = self.search_metadata.get("urls") or []
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
            sources.append(f"{i+1}. [{title}]({url})")
        return sources

    def _paginate_sources(self) -> list[list[str]]:
        """Split sources into pages that fit within embed limits."""
        if not self.sources:
            return [[]]

        pages = []
        current_page = []
        current_size = 0

        # Account for queries field size (on every page)
        queries_size = 0
        if self.queries:
            queries_text = "\n".join(f"• {q}" for q in self.queries)[:self.FIELD_LIMIT]
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
        embed = discord.Embed(title="Web Search Sources", color=discord.Color.blue())

        # Display search queries (on every page)
        if self.queries:
            queries_text = "\n".join(f"• {q}" for q in self.queries)[:self.FIELD_LIMIT]
            embed.add_field(name="Search Queries", value=queries_text, inline=False)

        # Display sources for current page
        if self.pages and self.pages[self.current_page]:
            page_sources = self.pages[self.current_page]

            # Split into multiple fields if needed
            add_chunked_embed_field(embed, page_sources, "Sources", self.FIELD_LIMIT)
        elif not self.urls:
            embed.add_field(name="Sources", value="No URLs available", inline=False)

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

        return embed

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
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

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
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

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

    page_input = discord.ui.TextInput(
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
                self.sources_view.current_page = page_num - 1  # Convert to 0-indexed
                self.sources_view.update_buttons()
                await interaction.response.edit_message(
                    embed=self.sources_view.build_embed(),
                    view=self.sources_view,
                )
            else:
                await interaction.response.send_message(
                    (
                        "❌ Invalid page number. Please enter a number between 1 "
                        f"and {self.sources_view.total_pages}."
                    ),
                    ephemeral=True,
                )
        except ValueError:
            await interaction.response.send_message(
                "❌ Please enter a valid number.",
                ephemeral=True,
            )
