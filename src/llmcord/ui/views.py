"""Discord UI components (Views and Buttons) for llmcord."""
import logging
from dataclasses import dataclass
from collections.abc import Awaitable, Callable, Mapping, Sequence

import discord
import httpx
from bs4 import BeautifulSoup

from llmcord.config import get_or_create_httpx_client
from llmcord.services.database import get_bad_keys_db

LOGGER = logging.getLogger(__name__)

# Shared httpx client for text.is uploads - uses factory pattern for DRY
_textis_client_holder: list[httpx.AsyncClient] = []
HTTP_OK = 200
URL_MAX_LENGTH = 150

VIEW_RESPONSE_BETTER_ID = "llmcord:response:better"
RETRY_RESPONSE_ID = "llmcord:response:retry"
GROUNDING_SOURCES_ID = "llmcord:response:sources:grounding"
TAVILY_SOURCES_ID = "llmcord:response:sources:tavily"

RetryHandler = Callable[[discord.Interaction, int, int], Awaitable[None]]
_retry_handler: RetryHandler | None = None


def set_retry_handler(handler: RetryHandler | None) -> None:
    """Set the global retry handler used by persistent buttons."""
    global _retry_handler
    _retry_handler = handler


@dataclass(slots=True)
class ResponseData:
    full_response: str | None
    grounding_metadata: object | None
    tavily_metadata: Mapping[str, object] | None
    request_message_id: int | None
    request_user_id: int | None


def _get_response_data(message_id: int) -> ResponseData:
    db = get_bad_keys_db()
    (
        full_response,
        grounding_metadata,
        tavily_metadata,
        request_message_id,
        request_user_id,
    ) = db.get_message_response_data(str(message_id))

    return ResponseData(
        full_response=full_response,
        grounding_metadata=grounding_metadata,
        tavily_metadata=tavily_metadata,
        request_message_id=int(request_message_id)
        if request_message_id and str(request_message_id).isdigit()
        else None,
        request_user_id=int(request_user_id)
        if request_user_id and str(request_user_id).isdigit()
        else None,
    )


def _get_textis_client() -> httpx.AsyncClient:
    """Get or create the shared text.is httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _textis_client_holder,
        timeout=30.0,
        connect_timeout=10.0,
        max_connections=10,
        max_keepalive=5,
        proxy_url=None,
        follow_redirects=False,  # text.is needs redirect handling
    )


async def upload_to_textis(text: str) -> str | None:
    """Upload text to text.is and return the paste URL.

    Returns None if upload fails.
    """
    try:
        client = _get_textis_client()

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
    except Exception:
        LOGGER.exception("Error uploading to text.is")
    else:
        return None

    return None


def _normalize_queries(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(query) for query in value if query]
    if value:
        return [str(value)]
    return []


def _extract_queries_from_mapping(mapping: Mapping[str, object]) -> list[str]:
    for key in ("web_search_queries", "searchQueries", "webSearchQueries"):
        value = mapping.get(key)
        if value:
            return _normalize_queries(value)
    return []


def _get_grounding_queries(metadata: object | None) -> list[str]:
    """Extract web search queries from grounding metadata.

    Handles GenAI types.GroundingMetadata, LiteLLM dict formats, and list formats.
    """
    if metadata is None:
        return []

    # Handle list format (LiteLLM streaming returns list of grounding results)
    if isinstance(metadata, list):
        queries = []
        for item in metadata:
            if isinstance(item, Mapping):
                queries.extend(_extract_queries_from_mapping(item))
        return queries

    # Handle dict format (LiteLLM)
    if isinstance(metadata, Mapping):
        return _extract_queries_from_mapping(metadata)

    # Handle object format (GenAI GroundingMetadata)
    if hasattr(metadata, "web_search_queries"):
        return _normalize_queries(getattr(metadata, "web_search_queries", None))

    return []


def _extract_chunks_from_mapping(mapping: Mapping[str, object]) -> list[object]:
    for key in ("grounding_chunks", "groundingChunks", "chunks"):
        value = mapping.get(key)
        if value:
            if isinstance(value, list):
                return list(value)
            return [value]
    return []


def _normalize_chunk(chunk: object) -> dict[str, str] | None:
    if isinstance(chunk, Mapping):
        web = chunk.get("web")
        if isinstance(web, Mapping):
            title = str(web.get("title") or "")
            uri = str(web.get("uri") or "")
            if title or uri:
                return {"title": title, "uri": uri}
        title = str(chunk.get("title") or "")
        uri = str(chunk.get("uri") or "")
        if title or uri:
            return {"title": title, "uri": uri}
        return None

    web = getattr(chunk, "web", None)
    if web:
        title = str(getattr(web, "title", "") or "")
        uri = str(getattr(web, "uri", "") or "")
        if title or uri:
            return {"title": title, "uri": uri}
    return None


def _normalize_chunks(chunks: Sequence[object]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for chunk in chunks:
        result = _normalize_chunk(chunk)
        if result:
            normalized.append(result)
    return normalized


def _get_grounding_chunks(metadata: object | None) -> list[dict[str, str]]:
    """Extract grounding chunks from grounding metadata.

    Returns list of dicts with 'title' and 'uri' keys.
    Handles GenAI types.GroundingMetadata, LiteLLM dict formats, and list formats.
    """
    if metadata is None:
        return []

    # Handle list format (LiteLLM streaming returns list of grounding results)
    if isinstance(metadata, list):
        result: list[dict[str, str]] = []
        for item in metadata:
            if isinstance(item, Mapping):
                chunks = _extract_chunks_from_mapping(item)
                result.extend(_normalize_chunks(chunks))
        return result

    # Handle dict format (LiteLLM)
    if isinstance(metadata, Mapping):
        chunks = _extract_chunks_from_mapping(metadata)
        return _normalize_chunks(chunks)

    # Handle object format (GenAI GroundingMetadata)
    if hasattr(metadata, "grounding_chunks"):
        chunks = getattr(metadata, "grounding_chunks", []) or []
        return _normalize_chunks(chunks)

    return []


def _has_grounding_data(metadata: object | None) -> bool:
    """Check if metadata has any grounding data (queries or chunks).

    Used to determine if the Show Sources button should be displayed.
    Only returns True if actual grounding queries or chunks exist.
    """
    if metadata is None:
        return False

    return bool(_get_grounding_queries(metadata)) or bool(
        _get_grounding_chunks(metadata),
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
            response_data = _get_response_data(interaction.message.id)
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

        if not _retry_handler:
            await interaction.followup.send(
                "❌ Retry is unavailable right now. Please try again later.",
                ephemeral=True,
            )
            return

        if not response_data:
            response_data = _get_response_data(interaction.message.id)

        if not response_data.request_message_id or not response_data.request_user_id:
            await interaction.followup.send(
                "❌ Missing retry context for this message.",
                ephemeral=True,
            )
            return

        await _retry_handler(
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
            response_data = _get_response_data(interaction.message.id)
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
        if metadata and _has_grounding_data(metadata):
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


def add_chunked_embed_field(
    embed: discord.Embed,
    items: list[str],
    base_name: str,
    field_limit: int = 1024,
) -> None:
    """Add items to embed fields, splitting into multiple fields if needed.

    Args:
        embed: The Discord embed to add fields to
        items: List of strings to add
        base_name: Base name for the field (e.g., "Sources", "Search Results")
        field_limit: Maximum characters per field (default: 1024)

    """
    if not items:
        return

    current_chunk = ""
    field_count = 1

    for item in items:
        if len(current_chunk) + len(item) + 1 > field_limit:
            field_name = (
                f"{base_name} ({field_count})" if field_count > 1 else base_name
            )
            embed.add_field(name=field_name, value=current_chunk, inline=False)
            current_chunk = item
            field_count += 1
        else:
            current_chunk = (current_chunk + "\n" + item) if current_chunk else item

    if current_chunk:
        field_name = f"{base_name} ({field_count})" if field_count > 1 else base_name
        embed.add_field(name=field_name, value=current_chunk, inline=False)


def build_grounding_sources_embed(metadata: object) -> discord.Embed:
    """Build a Discord embed showing sources from grounding metadata.

    Args:
        metadata: Grounding metadata (either GenAI GroundingMetadata or LiteLLM dict)

    Returns:
        A Discord Embed with the formatted sources

    """
    embed = discord.Embed(title="Sources", color=discord.Color.blue())

    queries = _get_grounding_queries(metadata)
    if queries:
        embed.add_field(
            name="Search Queries",
            value="\n".join(f"• {q}" for q in queries),
            inline=False,
        )

    chunks = _get_grounding_chunks(metadata)
    if chunks:
        sources = []
        for i, chunk in enumerate(chunks):
            if chunk.get("uri"):
                sources.append(f"{i+1}. [{chunk['title']}]({chunk['uri']})")
        add_chunked_embed_field(embed, sources, "Search Results")

    # Handle edge case where no content was added to the embed
    if not embed.fields:
        embed.add_field(
            name="Sources",
            value="No source information available.",
            inline=False,
        )

    return embed


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
        if metadata is None:
            response_data = _get_response_data(interaction.message.id)
            metadata = response_data.grounding_metadata

        if not metadata:
            await interaction.response.send_message(
                "❌ No source information available for this message.",
                ephemeral=True,
            )
            return

        embed = build_grounding_sources_embed(metadata)
        await interaction.response.send_message(embed=embed, ephemeral=True)


class TavilySourceButton(discord.ui.Button):
    """Button to show sources from web search (supports Tavily and Exa)."""

    def __init__(self, search_metadata: Mapping[str, object] | None = None) -> None:
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
        if search_metadata is None:
            response_data = _get_response_data(interaction.message.id)
            search_metadata = response_data.tavily_metadata

        if not search_metadata:
            await interaction.response.send_message(
                "❌ No web search sources available for this message.",
                ephemeral=True,
            )
            return

        view = TavilySourcesView(search_metadata)
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
        if metadata is None:
            response_data = _get_response_data(interaction.message.id)
            metadata = response_data.grounding_metadata

        if not metadata:
            await interaction.response.send_message(
                "❌ No source information available for this message.",
                ephemeral=True,
            )
            return

        embed = build_grounding_sources_embed(metadata)
        await interaction.response.send_message(embed=embed, ephemeral=True)

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
        self.type = discord.ComponentType.text_display if hasattr(discord.ComponentType, "text_display") else discord.ComponentType.button


class LayoutView(discord.ui.View):
    """View to hold text displays."""

    def __init__(self) -> None:
        """Initialize the layout view."""
        super().__init__(timeout=None)
