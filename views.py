"""
Discord UI components (Views and Buttons) for llmcord.
"""
import logging
from typing import Optional

import discord
import httpx
from bs4 import BeautifulSoup
from google.genai import types


async def upload_to_textis(text: str) -> Optional[str]:
    """
    Upload text to text.is and return the paste URL.
    Returns None if upload fails.
    """
    try:
        # Browser-like headers to avoid being blocked
        browser_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        async with httpx.AsyncClient(follow_redirects=False, headers=browser_headers) as client:
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


class SourceButton(discord.ui.Button):
    """Button to show sources from grounding metadata"""
    
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary)
        self.metadata = metadata
    
    async def callback(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Sources", color=discord.Color.blue())

        if self.metadata.web_search_queries:
            embed.add_field(name="Search Queries", value="\n".join(f"• {q}" for q in self.metadata.web_search_queries), inline=False)

        if self.metadata.grounding_chunks:
            sources = []
            for i, chunk in enumerate(self.metadata.grounding_chunks):
                if chunk.web:
                    sources.append(f"{i+1}. [{chunk.web.title}]({chunk.web.uri})")

            if sources:
                current_chunk = ""
                field_count = 1
                for source in sources:
                    if len(current_chunk) + len(source) + 1 > 1024:
                        embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)
                        current_chunk = source
                        field_count += 1
                    else:
                        current_chunk = (current_chunk + "\n" + source) if current_chunk else source

                if current_chunk:
                    embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)


class TavilySourceButton(discord.ui.Button):
    """Button to show sources from Tavily web search"""
    
    def __init__(self, tavily_metadata: dict):
        super().__init__(label="Show Sources", style=discord.ButtonStyle.secondary, emoji="🔍")
        self.tavily_metadata = tavily_metadata
    
    async def callback(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Web Search Sources", color=discord.Color.blue())
        
        # Display search queries
        queries = self.tavily_metadata.get("queries", [])
        if queries:
            queries_text = "\n".join(f"• {q}" for q in queries)
            embed.add_field(name="Search Queries", value=queries_text[:1024], inline=False)
        
        # Display URLs
        urls = self.tavily_metadata.get("urls", [])
        if urls:
            sources = []
            for i, url_info in enumerate(urls):
                title = url_info.get("title", "No title")[:50]  # Truncate long titles
                url = url_info.get("url", "")
                sources.append(f"{i+1}. [{title}]({url})")
            
            # Split into multiple fields if needed (Discord field limit is 1024 chars)
            current_chunk = ""
            field_count = 1
            for source in sources:
                if len(current_chunk) + len(source) + 1 > 1024:
                    embed.add_field(
                        name=f"URLs Used ({field_count})" if field_count > 1 else "URLs Used", 
                        value=current_chunk, 
                        inline=False
                    )
                    current_chunk = source
                    field_count += 1
                else:
                    current_chunk = (current_chunk + "\n" + source) if current_chunk else source
            
            if current_chunk:
                embed.add_field(
                    name=f"URLs Used ({field_count})" if field_count > 1 else "URLs Used", 
                    value=current_chunk, 
                    inline=False
                )
        
        embed.set_footer(text="Powered by Tavily Search")
        await interaction.response.send_message(embed=embed, ephemeral=True)


class SourceView(discord.ui.View):
    """Legacy view for backwards compatibility - now using ResponseView instead"""
    def __init__(self, metadata: types.GroundingMetadata):
        super().__init__(timeout=None)
        self.metadata = metadata

    @discord.ui.button(label="Show Sources", style=discord.ButtonStyle.secondary)
    async def show_sources(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(title="Sources", color=discord.Color.blue())

        if self.metadata.web_search_queries:
            embed.add_field(name="Search Queries", value="\n".join(f"• {q}" for q in self.metadata.web_search_queries), inline=False)

        if self.metadata.grounding_chunks:
            sources = []
            for i, chunk in enumerate(self.metadata.grounding_chunks):
                if chunk.web:
                    sources.append(f"{i+1}. [{chunk.web.title}]({chunk.web.uri})")

            if sources:
                current_chunk = ""
                field_count = 1
                for source in sources:
                    if len(current_chunk) + len(source) + 1 > 1024:
                        embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)
                        current_chunk = source
                        field_count += 1
                    else:
                        current_chunk = (current_chunk + "\n" + source) if current_chunk else source

                if current_chunk:
                    embed.add_field(name=f"Search Results ({field_count})" if field_count > 1 else "Search Results", value=current_chunk, inline=False)

        await interaction.response.send_message(embed=embed, ephemeral=True)
