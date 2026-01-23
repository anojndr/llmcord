"""
Web search functionality including search decision and Tavily integration.
Uses LiteLLM for unified LLM API access via shared litellm_utils.
"""
import asyncio
import json
import logging
from datetime import datetime

import httpx
import litellm

from bad_keys import get_bad_keys_db, KeyRotator
from config import get_or_create_httpx_client
from litellm_utils import prepare_litellm_kwargs


def get_current_datetime_strings() -> tuple[str, str]:
    """
    Get current date and time strings for system prompts.
    Returns (date_str, time_str) tuple.
    
    Date format: "January 21 2026"
    Time format: "20:00:00 +0800"
    """
    now = datetime.now().astimezone()
    date_str = now.strftime("%B %d %Y")
    time_str = now.strftime("%H:%M:%S %Z%z")
    return date_str, time_str


def convert_messages_to_openai_format(
    messages: list, 
    system_prompt: str = None, 
    reverse: bool = True,
    include_analysis_prompt: bool = False
) -> list[dict]:
    """
    Convert internal message format to OpenAI-compatible message format.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        reverse: Whether to reverse the message order (default True for chronological)
        include_analysis_prompt: Whether to append the analysis instruction prompt
    
    Returns:
        List of OpenAI-compatible message dicts
    """
    openai_messages = []
    
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    
    message_list = messages[::-1] if reverse else messages
    
    for msg in message_list:
        role = msg.get("role", "user")
        if role == "system":
            continue  # Skip system messages from chat history
        
        content = msg.get("content", "")
        if isinstance(content, list):
            # Keep multimodal format for OpenAI
            openai_messages.append({"role": role, "content": content})
        elif content:
            openai_messages.append({"role": role, "content": str(content)})
    
    if include_analysis_prompt:
        openai_messages.append({
            "role": "user",
            "content": "Based on the conversation above, analyze the last user query and respond with your JSON decision."
        })
    
    return openai_messages


# Web Search Decider and Tavily Search Functions
SEARCH_DECIDER_SYSTEM_PROMPT = """You are a web search query optimizer. Your job is to analyze user queries and determine if they need web search for up-to-date or factual information.

RESPOND ONLY with a valid JSON object. No other text, no markdown.

If web search is NOT needed (e.g., creative writing, opinions, general knowledge, math, coding):
{"needs_search": false}

If web search IS needed (e.g., current events, recent news, product specs, prices, real-time info):
{"needs_search": true, "queries": ["query1", "query2", ...]}

RULES for generating queries:
1. Keep queries concise—under 400 characters. Think of it as a query for an agent performing web search, not long-form prompts.
2. EXTRACT CONCRETE ENTITIES, NOT META-DESCRIPTIONS. Your queries must be actual searchable terms, NOT descriptions of what to search for.
   - NEVER output queries like "events mentioned by X", "topics in the conversation", "things the user asked about", etc.
   - ALWAYS extract the ACTUAL entities, names, events, or topics from the conversation and use those as queries.
   - If the user says "search for the events John mentioned" and John mentioned "Russia Ukraine war" and "Trump policies", your queries should be ["Russia Ukraine war", "Trump policies"], NOT ["events mentioned by John"].
3. SINGLE ENTITY = SINGLE QUERY. If the user asks about ONE entity, output exactly ONE search query. Do NOT split into multiple queries.
   Example: "latest news" → ["latest news today"] (ONE query only)
   Example: "iPhone 16 price" → ["iPhone 16 price"] (ONE query only)
4. MULTIPLE ENTITIES = MULTIPLE QUERIES. Only if the user asks about multiple entities, create separate queries for EACH entity PLUS a query containing all entities.
   Example: "which is the best? B&K 5128 Diffuse Field Target, VDSF 5128 Demo Target Response On-Ear, VDSF 5128 Demo Target Response In-Ear, 5128 Harman In-Ear 2024 Beta, or 4128/4195 VDSF Target Response?" → ["B&K 5128 Diffuse Field Target", "VDSF 5128 Demo Target Response On-Ear", "VDSF 5128 Demo Target Response In-Ear", "5128 Harman In-Ear 2024 Beta", "4128/4195 VDSF Target Response", "B&K 5128 Diffuse Field Target vs VDSF 5128 Demo Target Response On-Ear vs VDSF 5128 Demo Target Response In-Ear vs 5128 Harman In-Ear 2024 Beta vs 4128/4195 VDSF Target Response"]
5. Make queries search-engine friendly
6. Preserve the user's original intent

BAD QUERIES (never output these):
- "events mentioned by Joeii in their reply" ❌ (meta-description, not searchable)
- "topics discussed in the conversation" ❌ (vague, not extracting actual content)
- "things the user wants to know about" ❌ (self-referential)
- "information from the image" ❌ (not extracting actual content from image)

GOOD QUERIES (extract actual content):
- If someone mentions "Russia invading Ukraine" → ["Russia Ukraine war 2024"]
- If someone mentions "Trump's policies" → ["Trump policies"]
- If someone mentions "China and Taiwan conflict" → ["China Taiwan relations"]

Examples:
- "What's the weather today?" → {"needs_search": true, "queries": ["weather today"]}
- "Who won the 2024 Super Bowl?" → {"needs_search": true, "queries": ["2024 Super Bowl winner"]}
- "latest news" → {"needs_search": true, "queries": ["latest news today"]}
- "Write me a poem about cats" → {"needs_search": false}
- "Compare RTX 4090 and RTX 4080" → {"needs_search": true, "queries": ["RTX 4090", "RTX 4080", "RTX 4090 vs RTX 4080"]}
- User shares image with text about "Biden's economic plan" and says "search for this" → {"needs_search": true, "queries": ["Biden economic plan"]}
- Conversation mentions "Greenland invasion by Russia" and user says "look that up" → {"needs_search": true, "queries": ["Russia Greenland invasion"]}
"""


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """
    Uses a configurable model to decide if web search is needed and generates optimized queries.
    Uses LiteLLM for unified API access across all providers.
    Uses KeyRotator for consistent key rotation and bad key tracking.
    Returns: {"needs_search": bool, "queries": list[str]} or {"needs_search": False}
    
    decider_config should contain:
        - provider: "gemini", "github_copilot", or other (OpenAI-compatible)
        - model: model name
        - api_keys: list of API keys
        - base_url: (optional) for OpenAI-compatible providers
    """
    provider = decider_config.get("provider", "gemini")
    model = decider_config.get("model", "gemini-3-flash-preview")
    api_keys = decider_config.get("api_keys", [])
    base_url = decider_config.get("base_url")
    
    if not api_keys:
        return {"needs_search": False}
    
    # Use KeyRotator for consistent key rotation with synced bad key tracking
    rotator = KeyRotator(provider, api_keys)
    
    async for current_api_key in rotator.get_keys_async():
        try:
            date_str, time_str = get_current_datetime_strings()
            system_prompt_with_date = f"{SEARCH_DECIDER_SYSTEM_PROMPT}\n\nCurrent date: {date_str}. Current time: {time_str}."
            
            # Convert messages to OpenAI format (LiteLLM uses OpenAI format)
            litellm_messages = convert_messages_to_openai_format(
                messages, 
                system_prompt=system_prompt_with_date,
                reverse=True,
                include_analysis_prompt=True
            )
            
            if len(litellm_messages) <= 2:  # Only system prompt and analysis instruction
                return {"needs_search": False}
            
            # Use shared utility to prepare kwargs with all provider-specific config
            litellm_kwargs = prepare_litellm_kwargs(
                provider=provider,
                model=model,
                messages=litellm_messages,
                api_key=current_api_key,
                base_url=base_url,
                temperature=0.1,
            )
            
            # Make the LiteLLM call
            response = await litellm.acompletion(**litellm_kwargs)
            
            response_text = (response.choices[0].message.content or "").strip()
            
            # Parse response
            if response_text.startswith("```"):
                # Remove markdown code blocks if present
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            try:
                result = json.loads(response_text)
                # Validate response structure
                if not isinstance(result, dict):
                    logging.warning(f"Web search decider returned non-dict response: {response_text[:100]}")
                    return {"needs_search": False}
                return result
            except json.JSONDecodeError as json_err:
                logging.warning(f"Failed to parse JSON response from search decider: {json_err}. Response: {response_text[:200]}")
                # Attempt to extract needs_search from malformed response
                if '"needs_search": false' in response_text.lower() or '"needs_search":false' in response_text.lower():
                    return {"needs_search": False}
                return {"needs_search": False}
        except Exception as e:
            logging.exception(f"Error in web search decider: {e}")
            rotator.mark_current_bad(str(e))
    
    # All keys exhausted
    logging.error("Web search decider failed after exhausting all keys, skipping web search")
    return {"needs_search": False}


# Shared httpx client for Tavily API calls - uses DRY factory pattern
_tavily_client_holder: list = []


def _get_tavily_client() -> httpx.AsyncClient:
    """Get or create the shared Tavily httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _tavily_client_holder,
        timeout=30.0,
        connect_timeout=10.0,
        max_connections=20,
        max_keepalive=10,
        follow_redirects=True,
        headers={},  # Tavily doesn't need browser headers, just defaults
    )


async def tavily_search(
    query: str, 
    tavily_api_key: str, 
    max_results: int = 5, 
    search_depth: str = "basic"
) -> dict:
    """
    Execute a single Tavily search query.
    Returns the search results with page content or an error dict.
    
    Best practices applied:
    - search_depth configurable (basic/advanced/fast/ultra-fast)
    - Reuses shared httpx client for connection pooling
    
    Args:
        query: Search query (keep under 400 characters)
        tavily_api_key: Tavily API key
        max_results: Maximum results to return (1-20)
        search_depth: "basic", "advanced", "fast", or "ultra-fast"
    """
    try:
        client = _get_tavily_client()
        
        # Build request payload
        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": "markdown",  # Get full page content in markdown format
        }
        
        # Increase timeout for advanced depth which takes longer
        timeout = 45.0 if search_depth == "advanced" else 30.0
        
        logging.info(f"Tavily API request for query '{query}': depth={search_depth}, max_results={max_results}")
        
        response = await client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json"
            },
            timeout=timeout,
        )
        logging.info(f"Tavily API response status: {response.status_code}")
        
        # Log raw response for debugging (first 1000 chars)
        raw_text = response.text[:1000] if len(response.text) > 1000 else response.text
        logging.debug(f"Tavily API raw response: {raw_text}")
        
        response.raise_for_status()
        result = response.json()
        logging.info(f"Tavily API response for query '{query}': {len(result.get('results', []))} results")
        if not result.get("results"):
            logging.warning(f"Tavily returned empty results for query '{query}'. Full response: {result}")
        return result
    except httpx.HTTPStatusError as e:
        logging.error(f"Tavily HTTP error for query '{query}': {e.response.status_code} - {e.response.text[:500]}")
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}", "query": query}
    except httpx.TimeoutException as e:
        logging.error(f"Tavily timeout for query '{query}': {e}")
        return {"error": f"Timeout: {e}", "query": query}
    except httpx.RequestError as e:
        logging.error(f"Tavily connection error for query '{query}': {e}")
        return {"error": f"Connection error: {e}", "query": query}
    except Exception as e:
        logging.exception(f"Tavily search error for query '{query}': {e}")
        return {"error": str(e), "query": query}


async def perform_web_search(
    queries: list[str], 
    tavily_api_keys: list[str], 
    max_results_per_query: int = 5, 
    max_chars_per_url: int = 2000,
    min_score: float = 0.3
) -> tuple[str, dict]:
    """
    Perform concurrent web searches for multiple queries using Tavily.
    Uses KeyRotator for consistent key rotation and bad key tracking.
    Returns a tuple of (formatted_results, metadata).
    
    Best practices applied:
    - Concurrent requests with asyncio.gather()
    - KeyRotator for synced bad key tracking with database persistence
    - Always uses "advanced" search depth for highest quality results
    
    Args:
        queries: List of search queries
        tavily_api_keys: List of Tavily API keys for rotation
        max_results_per_query: Maximum number of URLs per query (default: 5)
        max_chars_per_url: Maximum characters per URL content (default: 2000)
        min_score: Minimum relevance score to include a result (0.0-1.0, default: 0.3)
    
    Returns:
        tuple: (formatted_results_string, {"queries": [...], "urls": [{...}, ...]})
    """
    if not queries or not tavily_api_keys:
        return "", {}
    
    # Pre-fetch good keys once instead of creating KeyRotator per query
    db = get_bad_keys_db()
    good_keys = db.get_good_keys_synced("tavily", tavily_api_keys)
    if not good_keys:
        db.reset_provider_keys_synced("tavily")
        good_keys = tavily_api_keys.copy()
    
    async def search_single_query(query: str, depth: str, keys: list[str]) -> dict:
        """
        Search with retry logic using pre-fetched keys.
        """
        for key in keys:
            result = await tavily_search(query, key, max_results_per_query, depth)
            
            if "error" not in result:
                return result
            
            # Mark the key as bad
            error_msg = result.get("error", "Unknown error")[:200]
            db.mark_key_bad_synced("tavily", key, error_msg)
        
        # All keys failed
        logging.error(f"All Tavily API keys failed for query '{query}'")
        return {"error": "All API keys exhausted", "query": query}
    
    async def execute_searches(depth: str) -> tuple[list, list]:
        """Execute searches with given parameters and return results and URLs."""
        # Execute searches concurrently with shared key list
        search_tasks = [
            search_single_query(query, depth, good_keys)
            for query in queries
        ]
        results = await asyncio.gather(*search_tasks)
        
        formatted = []
        urls = []
        
        logging.info(f"Tavily search returned {len(results)} result sets for {len(queries)} queries")
        
        for i, result in enumerate(results):
            if "error" in result:
                logging.warning(f"Tavily search error for query '{queries[i]}': {result.get('error')}")
                continue
            
            query = queries[i]
            query_results = []
            
            result_items = result.get("results", [])
            logging.info(f"Query '{query}' returned {len(result_items)} items")
            
            for item in result_items:
                score = item.get("score", 0)
                
                title = item.get("title", "No title")
                url = item.get("url", "")
                
                if url:
                    urls.append({"title": title, "url": url, "score": score})
                
                # Prefer raw_content (full page content) over content (snippet)
                raw_content = item.get("raw_content", "")
                content = item.get("content", "")
                
                page_content = raw_content if raw_content else content
                page_content = page_content[:max_chars_per_url] if page_content else ""
                
                result_text = f"\n**{title}** (relevance: {score:.2f})\n{url}\n{page_content}\n"
                query_results.append(result_text)
            
            if query_results:
                formatted.append(f"\n### Search Results for: {query}")
                formatted.extend(query_results)
        
        logging.info(f"Total URLs collected: {len(urls)}")
        return formatted, urls
    
    # Always use advanced depth for best results
    formatted_results, all_urls = await execute_searches("advanced")
    
    metadata = {
        "queries": queries,
        "urls": all_urls
    }
    
    if formatted_results:
        return "\n\n---\nHere are the web search results in case the user asked you to search the net or something:\n\n**Web Search Results:**" + "".join(formatted_results), metadata
    return "", metadata
