"""
Discord UI components (Views and Buttons) for llmcord.
"""
import logging
from typing import Optional

import discord
import httpx
from bs4 import BeautifulSoup
from google.genai import types

from config import get_config


async def upload_to_textis(text: str) -> Optional[str]:
    """
    Upload text to text.is and return the paste URL.
    Returns None if upload fails.
    """
    try:
        # Browser-like headers to avoid being blocked
        # Note: Don't include Accept-Encoding - let httpx handle decompression automatically
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Proxy configuration from config (optional)
        config = get_config()
        proxy_url = config.get("proxy_url") or None
        
        async with httpx.AsyncClient(follow_redirects=False, headers=browser_headers, proxy=proxy_url) as client:
            # Get the CSRF token from the main page
            response = await client.get("https://text.is/", timeout=30)
            
            # Extract CSRF token from the form
            soup = BeautifulSoup(response.text, "lxml")  # lxml is faster than html.parser
            csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
            if not csrf_input:
                # Debug: log a snippet of the response to help diagnose
                logging.error("Could not find CSRF token on text.is")
                logging.debug(f"Response status: {response.status_code}, Content preview: {response.text[:500]}")
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
                timeout=30
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
            if post_response.status_code == 200:
                # Check if the URL changed (we might be on the paste page)
                final_url = str(post_response.url)
                if final_url != "https://text.is/" and "text.is/" in final_url:
                    return final_url
            
            logging.error(f"Unexpected response from text.is: {post_response.status_code}")
            return None
            
    except Exception as e:
        logging.exception(f"Error uploading to text.is: {e}")
        return None


class ResponseView(discord.ui.View):
    """View with 'View Response Better' button that uploads to text.is"""
    
    def __init__(self, full_response: str, metadata: Optional[types.GroundingMetadata] = None, tavily_metadata: Optional[dict] = None):
        super().__init__(timeout=None)
        self.full_response = full_response
        self.metadata = metadata
        self.tavily_metadata = tavily_metadata
        
        # Add Gemini grounding sources button if we have metadata with search queries
        if metadata and metadata.web_search_queries:
            self.add_item(SourceButton(metadata))
        
        # Add Tavily sources button if we have tavily metadata with URLs or queries
        if tavily_metadata and (tavily_metadata.get("urls") or tavily_metadata.get("queries")):
            self.add_item(TavilySourceButton(tavily_metadata))

    @discord.ui.button(label="View Response Better", style=discord.ButtonStyle.secondary, emoji="📄")
    async def view_response_better(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        
        paste_url = await upload_to_textis(self.full_response)
        
        if paste_url:
            embed = discord.Embed(
                title="View Response",
                description=f"Your response has been uploaded for better viewing:\n\n**[Click here to view]({paste_url})**",
                color=discord.Color.green()
            )
            embed.set_footer(text="Powered by text.is")
            await interaction.followup.send(embed=embed, ephemeral=True)
        else:
            await interaction.followup.send(
                "❌ Failed to upload response to text.is. Please try again later.",
                ephemeral=True
            )


def add_chunked_embed_field(embed: discord.Embed, items: list[str], base_name: str, field_limit: int = 1024) -> None:
    """
    Add items to embed fields, splitting into multiple fields if content exceeds limit.
    
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
            field_name = f"{base_name} ({field_count})" if field_count > 1 else base_name
            embed.add_field(name=field_name, value=current_chunk, inline=False)
            current_chunk = item
            field_count += 1
        else:
            current_chunk = (current_chunk + "\n" + item) if current_chunk else item
    
    if current_chunk:
        field_name = f"{base_name} ({field_count})" if field_count > 1 else base_name
        embed.add_field(name=field_name, value=current_chunk, inline=False)


def build_grounding_sources_embed(metadata: types.GroundingMetadata) -> discord.Embed:
    """
    Build a Discord embed showing sources from Gemini grounding metadata.
    
    Args:
        metadata: Gemini GroundingMetadata containing search queries and chunks
    
    Returns:
        A Discord Embed with the formatted sources
    """
    embed = discord.Embed(title="Sources", color=discord.Color.blue())
    
    if metadata and metadata.web_search_queries:
        embed.add_field(
            name="Search Queries", 
            value="\n".join(f"• {q}" for q in metadata.web_search_queries), 
            inline=False
        )
    
    if metadata and metadata.grounding_chunks:
        sources = [
            f"{i+1}. [{chunk.web.title}]({chunk.web.uri})"
            for i, chunk in enumerate(metadata.grounding_chunks)
            if chunk.web
        ]
        add_chunked_embed_field(embed, sources, "Search Results")
    
    # Handle edge case where no content was added to the embed
    if not embed.fields:
        embed.add_field(name="Sources", value="No source information available.", inline=False)
    
    return embed


class SourceButton(discord.ui.Button):
    """Button to show sources from grounding metadata"""
    
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary)
        self.metadata = metadata
    
    async def callback(self, interaction: discord.Interaction):
        embed = build_grounding_sources_embed(self.metadata)
        await interaction.response.send_message(embed=embed, ephemeral=True)


class TavilySourceButton(discord.ui.Button):
    """Button to show sources from Tavily web search"""
    
    def __init__(self, tavily_metadata: dict):
        super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary, emoji="🔍")
        self.tavily_metadata = tavily_metadata
    
    async def callback(self, interaction: discord.Interaction):
        view = TavilySourcesView(self.tavily_metadata)
        embed = view.build_embed()
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


class TavilySourcesView(discord.ui.View):
    """Paginated view for displaying Tavily search sources"""
    
    # Limits for embed content
    MAX_EMBED_SIZE = 5500  # Leave buffer below 6000
    FIELD_LIMIT = 1024
    SOURCES_PER_PAGE = 10  # Reasonable default
    
    def __init__(self, tavily_metadata: dict):
        super().__init__(timeout=300)  # 5 minute timeout
        # Handle None or malformed metadata
        self.tavily_metadata = tavily_metadata or {}
        self.current_page = 0
        
        # Prepare source entries with defensive defaults
        self.queries = self.tavily_metadata.get("queries") or []
        self.urls = self.tavily_metadata.get("urls") or []
        self.sources = self._prepare_sources()
        self.pages = self._paginate_sources()
        self.total_pages = len(self.pages)
        
        # Update button states
        self._update_buttons()
    
    def _prepare_sources(self) -> list[str]:
        """Prepare formatted source strings"""
        sources = []
        for i, url_info in enumerate(self.urls):
            title = url_info.get("title", "No title")[:80]  # Truncate long titles
            url = url_info.get("url", "")
            # Truncate very long URLs
            if len(url) > 150:
                url = url[:150] + "..."
            sources.append(f"{i+1}. [{title}]({url})")
        return sources
    
    def _paginate_sources(self) -> list[list[str]]:
        """Split sources into pages that fit within embed limits"""
        if not self.sources:
            return [[]]
        
        pages = []
        current_page = []
        current_size = 0
        
        # Account for queries field size (on every page)
        queries_size = 0
        if self.queries:
            queries_text = "\n".join(f"• {q}" for q in self.queries)[:self.FIELD_LIMIT]
            queries_size = len("Search Queries") + len(queries_text) + 50  # field overhead
        
        base_size = len("Web Search Sources") + queries_size + 100  # title + queries + buffer
        
        for source in self.sources:
            source_size = len(source) + 2  # +2 for newline
            
            # Check if adding this source would exceed limits
            if current_size + source_size > self.MAX_EMBED_SIZE - base_size or len(current_page) >= self.SOURCES_PER_PAGE:
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
    
    def _update_buttons(self):
        """Update button disabled states based on current page"""
        self.prev_button.disabled = self.current_page <= 0
        self.next_button.disabled = self.current_page >= self.total_pages - 1
    
    def build_embed(self) -> discord.Embed:
        """Build the embed for the current page"""
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
        
        # Footer with pagination info
        footer_text = f"Page {self.current_page + 1}/{self.total_pages} • {len(self.urls)} total sources • Powered by Tavily Search"
        embed.set_footer(text=footer_text)
        
        return embed
    
    @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary, emoji="◀️", disabled=True)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = max(0, self.current_page - 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)
    
    @discord.ui.button(label="Next", style=discord.ButtonStyle.secondary, emoji="▶️")
    async def next_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = min(self.total_pages - 1, self.current_page + 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)
    
    @discord.ui.button(label="Go to Page", style=discord.ButtonStyle.primary, emoji="🔢")
    async def goto_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = GoToPageModal(self)
        await interaction.response.send_modal(modal)


class GoToPageModal(discord.ui.Modal, title="Go to Page"):
    """Modal for entering a specific page number"""
    
    page_input = discord.ui.TextInput(
        label="Page Number",
        placeholder="Enter page number...",
        required=True,
        min_length=1,
        max_length=5
    )
    
    def __init__(self, sources_view: TavilySourcesView):
        super().__init__()
        self.sources_view = sources_view
        self.page_input.placeholder = f"Enter 1-{sources_view.total_pages}"
    
    async def on_submit(self, interaction: discord.Interaction):
        try:
            page_num = int(self.page_input.value)
            if 1 <= page_num <= self.sources_view.total_pages:
                self.sources_view.current_page = page_num - 1  # Convert to 0-indexed
                self.sources_view._update_buttons()
                await interaction.response.edit_message(
                    embed=self.sources_view.build_embed(), 
                    view=self.sources_view
                )
            else:
                await interaction.response.send_message(
                    f"❌ Invalid page number. Please enter a number between 1 and {self.sources_view.total_pages}.",
                    ephemeral=True
                )
        except ValueError:
            await interaction.response.send_message(
                "❌ Please enter a valid number.",
                ephemeral=True
            )


class SourceView(discord.ui.View):
    """Legacy view for backwards compatibility - now using ResponseView instead"""
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(timeout=None)
        self.metadata = metadata

    @discord.ui.button(label="Show Sources", style=discord.ButtonStyle.secondary)
    async def show_sources(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = build_grounding_sources_embed(self.metadata)
        await interaction.response.send_message(embed=embed, ephemeral=True)
