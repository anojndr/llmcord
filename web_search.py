"""Provide web search decision and provider integration.

Supports both Tavily and Exa MCP as web search backends. Uses LiteLLM for
unified LLM API access via shared litellm_utils.
"""
import asyncio
import json
import logging
from datetime import datetime

import httpx
import litellm

from bad_keys import KeyRotator, get_bad_keys_db
from config import get_or_create_httpx_client
from litellm_utils import LiteLLMOptions, prepare_litellm_kwargs

logger = logging.getLogger(__name__)

MIN_DECIDER_MESSAGES = 2
MAX_LOG_CHARS = 1000
MAX_ERROR_CHARS = 500
HTTP_OK = 200
EXA_MCP_URL = (
    "https://mcp.exa.ai/mcp?tools=web_search_exa,web_search_advanced_exa,"
    "get_code_context_exa,deep_search_exa,crawling_exa,company_research_exa,"
    "linkedin_search_exa,deep_researcher_start,deep_researcher_check"
)


def get_current_datetime_strings() -> tuple[str, str]:
    """Get current date and time strings for system prompts.

    Returns a tuple of (date_str, time_str).

    Date format: "January 21 2026"
    Time format: "20:00:00 +0800"
    """
    now = datetime.now().astimezone()
    date_str = now.strftime("%B %d %Y")
    time_str = now.strftime("%H:%M:%S %Z%z")
    return date_str, time_str


def convert_messages_to_openai_format(
    messages: list,
    system_prompt: str | None = None,
    *,
    reverse: bool = True,
    include_analysis_prompt: bool = False,
) -> list[dict]:
    """Convert internal message format to OpenAI-compatible message format.

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
            # Filter to only include types supported by OpenAI-compatible APIs
            # GitHub Copilot and others only accept 'text' and 'image_url' types
            filtered_content = [
                part for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "image_url")
            ]
            if filtered_content:
                openai_messages.append({"role": role, "content": filtered_content})
        elif content:
            openai_messages.append({"role": role, "content": str(content)})

    if include_analysis_prompt:
        openai_messages.append(
            {
                "role": "user",
                "content": (
                    "Based on the conversation above, analyze the last user "
                    "query and respond with your JSON decision."
                ),
            },
        )

    return openai_messages


# Web Search Decider and Tavily Search Functions
SEARCH_DECIDER_SYSTEM_PROMPT = "\n".join(  # noqa: FLY002
     [
          "You are a web search query optimizer. Your job is to analyze user queries",
          "and determine if they need web search for up-to-date or factual",
          "information.",
          "",
          "RESPOND ONLY with a valid JSON object. No other text, no markdown.",
          "",
          "If web search is NOT needed (e.g., creative writing, opinions, general",
          "knowledge, math, coding):",
        '{"needs_search": false}',
          "",
          "If web search IS needed (e.g., current events, recent news, product",
          "specs, prices, real-time info):",
        '{"needs_search": true, "queries": ["query1", "query2", ...]}',
          "",
          "RULES for generating queries:",
          "1. Keep queries concise—under 400 characters. Think of it as a query",
          "for an agent performing web search, not long-form prompts.",
          "2. EXTRACT CONCRETE ENTITIES, NOT META-DESCRIPTIONS. Your queries",
          "must be actual searchable terms, NOT descriptions of what to search",
          "for.",
        '   - NEVER output queries like "events mentioned by X", "topics in',
        'the conversation", "things the user asked about", etc.',
          "   - ALWAYS extract the ACTUAL entities, names, events, or topics",
          "from the conversation and use those as queries.",
        '   - If the user says "search for the events John mentioned" and',
        'John mentioned "Russia Ukraine war" and "Trump policies", your',
        'queries should be ["Russia Ukraine war", "Trump policies"], NOT',
        '["events mentioned by John"].',
          "3. SINGLE ENTITY = SINGLE QUERY. If the user asks about ONE entity,",
          "output exactly ONE search query. Do NOT split into multiple queries.",
        '   Example: "latest news" → ["latest news today"] (ONE query only)',
        '   Example: "iPhone 16 price" → ["iPhone 16 price"] (ONE query only)',
          "4. MULTIPLE ENTITIES = MULTIPLE QUERIES. Only if the user asks about",
          "multiple entities, create separate queries for EACH entity PLUS a",
          "query containing all entities.",
        '   Example: "which is the best? B&K 5128 Diffuse Field Target, VDSF',
          "5128 Demo Target Response On-Ear, VDSF 5128 Demo Target Response",
          "In-Ear, 5128 Harman In-Ear 2024 Beta, or 4128/4195 VDSF Target",
        'Response?" → ["B&K 5128 Diffuse Field Target", "VDSF 5128 Demo',
        'Target Response On-Ear", "VDSF 5128 Demo Target Response In-Ear",',
        '"5128 Harman In-Ear 2024 Beta", "4128/4195 VDSF Target Response",',
        '"B&K 5128 Diffuse Field Target vs VDSF 5128 Demo Target Response',
          "On-Ear vs VDSF 5128 Demo Target Response In-Ear vs 5128 Harman",
        'In-Ear 2024 Beta vs 4128/4195 VDSF Target Response"]',
          "5. Make queries search-engine friendly",
        "6. Preserve the user's original intent",
          "",
          "BAD QUERIES (never output these):",
        '- "events mentioned by Joeii in their reply" ❌ (meta-description, not',
          "searchable)",
        '- "topics discussed in the conversation" ❌ (vague, not extracting',
          "actual content)",
        '- "things the user wants to know about" ❌ (self-referential)',
        '- "information from the image" ❌ (not extracting actual content',
          "from image)",
          "",
          "GOOD QUERIES (extract actual content):",
        '- If someone mentions "Russia invading Ukraine" → ["Russia Ukraine',
        'war 2024"]',
        '- If someone mentions "Trump\'s policies" → ["Trump policies"]',
        '- If someone mentions "China and Taiwan conflict" → ["China Taiwan',
        'relations"]',
          "",
          "Examples:",
        '- "What\'s the weather today?" → {"needs_search": true,',
        '  "queries": ["weather today"]}',
        '- "Who won the 2024 Super Bowl?" → {"needs_search": true,',
        '  "queries": ["2024 Super Bowl winner"]}',
        '- "latest news" → {"needs_search": true,',
        '  "queries": ["latest news today"]}',
        '- "Write me a poem about cats" → {"needs_search": false}',
        '- "Compare RTX 4090 and RTX 4080" → {"needs_search": true,',
        '  "queries": ["RTX 4090", "RTX 4080", "RTX 4090 vs RTX 4080"]}',
        '- User shares image with text about "Biden\'s economic plan" and says',
        '  "search for this" → {"needs_search": true,',
        '  "queries": ["Biden economic plan"]}',
        '- Conversation mentions "Greenland invasion by Russia" and user says',
        '  "look that up" → {"needs_search": true,',
        '  "queries": ["Russia Greenland invasion"]}',
        "",
    ],
)


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """Decide whether web search is needed and generate optimized queries.

    Uses LiteLLM for unified API access across all providers. Uses KeyRotator
    for consistent key rotation and bad key tracking.

    Returns: {"needs_search": bool, "queries": list[str]} or
    {"needs_search": False}.

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

    default_result = {"needs_search": False}
    if not api_keys:
        return default_result

    # Use KeyRotator for consistent key rotation with synced bad key tracking
    rotator = KeyRotator(provider, api_keys)

    exhausted_keys = True

    async for current_api_key in rotator.get_keys_async():
        try:
            date_str, time_str = get_current_datetime_strings()
            system_prompt_with_date = (
                f"{SEARCH_DECIDER_SYSTEM_PROMPT}\n\nCurrent date: {date_str}. "
                f"Current time: {time_str}."
            )

            # Convert messages to OpenAI format (LiteLLM uses OpenAI format)
            litellm_messages = convert_messages_to_openai_format(
                messages,
                system_prompt=system_prompt_with_date,
                reverse=True,
                include_analysis_prompt=True,
            )

            if len(litellm_messages) <= MIN_DECIDER_MESSAGES:
                exhausted_keys = False
                break

            # Use shared utility to prepare kwargs with all provider-specific config
            litellm_kwargs = prepare_litellm_kwargs(
                provider=provider,
                model=model,
                messages=litellm_messages,
                api_key=current_api_key,
                options=LiteLLMOptions(
                    base_url=base_url,
                    temperature=0.1,
                ),
            )

            # Make the LiteLLM call
            response = await litellm.acompletion(**litellm_kwargs)

            response_text = (response.choices[0].message.content or "").strip()

            # Parse response
            if response_text.startswith("```"):
                # Remove markdown code blocks if present
                response_text = response_text.split("```")[1]
                response_text = response_text.removeprefix("json")
                response_text = response_text.strip()

            try:
                result = json.loads(response_text)
                # Validate response structure
                if isinstance(result, dict):
                    return result
                logger.warning(
                    "Web search decider returned non-dict response: %s",
                    response_text[:100],
                )
                exhausted_keys = False
                break
            except json.JSONDecodeError as json_err:
                logger.warning(
                    "Failed to parse JSON response from search decider: %s. "
                    "Response: %s",
                    json_err,
                    response_text[:200],
                )
                # Attempt to extract needs_search from malformed response
                if (
                    '"needs_search": false' in response_text.lower()
                    or '"needs_search":false' in response_text.lower()
                ):
                    exhausted_keys = False
                    break
                exhausted_keys = False
                break
        except Exception as exc:
            logger.exception("Error in web search decider")
            rotator.mark_current_bad(str(exc))
            continue

        exhausted_keys = False
        break

    # All keys exhausted
    if exhausted_keys:
        logger.error(
            "Web search decider failed after exhausting all keys, skipping web "
            "search",
        )
    return default_result


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
    search_depth: str = "basic",
) -> dict:
    """Execute a single Tavily search query.

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
            "include_raw_content": "markdown",
        }

        # Increase timeout for advanced depth which takes longer
        timeout = 45.0 if search_depth == "advanced" else 30.0

        logger.info(
            "Tavily API request for query '%s': depth=%s, max_results=%s",
            query,
            search_depth,
            max_results,
        )

        response = await client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        logger.info("Tavily API response status: %s", response.status_code)

        # Log raw response for debugging (first 1000 chars)
        raw_text = (
            response.text[:MAX_LOG_CHARS]
            if len(response.text) > MAX_LOG_CHARS
            else response.text
        )
        logger.debug("Tavily API raw response: %s", raw_text)

        response.raise_for_status()
        result = response.json()
        logger.info(
            "Tavily API response for query '%s': %s results",
            query,
            len(result.get("results", [])),
        )
        if not result.get("results"):
            logger.warning(
                "Tavily returned empty results for query '%s'. Full response: %s",
                query,
                result,
            )
        else:
            return result
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Tavily HTTP error for query '%s': %s - %s",
            query,
            exc.response.status_code,
            exc.response.text[:MAX_ERROR_CHARS],
        )
        return {
            "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            "query": query,
        }
    except httpx.TimeoutException as exc:
        logger.exception("Tavily timeout for query '%s'", query)
        return {"error": f"Timeout: {exc}", "query": query}
    except httpx.RequestError as exc:
        logger.exception("Tavily connection error for query '%s'", query)
        return {"error": f"Connection error: {exc}", "query": query}
    except Exception as exc:
        logger.exception("Tavily search error for query '%s'", query)
        return {"error": str(exc), "query": query}

    return {"results": [], "query": query}


# Shared httpx client for Exa MCP API calls - uses DRY factory pattern
_exa_client_holder: list = []


def _get_exa_client() -> httpx.AsyncClient:
    """Get or create the shared Exa MCP httpx client using the DRY factory pattern."""
    return get_or_create_httpx_client(
        _exa_client_holder,
        timeout=60.0,  # Exa can take longer for deep searches
        connect_timeout=15.0,
        max_connections=20,
        max_keepalive=10,
        follow_redirects=True,
        headers={},
    )


def parse_exa_text_format(text_content: str) -> list[dict]:  # noqa: C901
    """Parse Exa MCP's structured text response format into a list of result dicts.

    Exa returns results in this format (multiple results separated by blank lines):
    Title: ...
    Author: ...
    Published Date: ...
    URL: ...
    Text: ...

    Returns a list of dicts with 'title', 'url', and 'content' keys.
    """
    if not text_content:
        return []

    results = []

    # Split by double newlines to separate individual results
    # Each result block starts with "Title:"
    blocks = text_content.split("\n\nTitle:")

    for i, block in enumerate(blocks):
        # Add back "Title:" prefix for all blocks except the first if it starts
        # with "Title:".
        if i == 0:
            if not block.strip().startswith("Title:"):
                continue  # Skip if first block doesn't start with Title
            current_block = block.strip()
        else:
            current_block = "Title:" + block

        # Parse the block
        title = ""
        url = ""
        content = ""

        lines = current_block.split("\n")
        text_started = False
        text_lines = []

        for line in lines:
            if line.startswith("Title:"):
                title = line[6:].strip()
            elif line.startswith("URL:"):
                url = line[4:].strip()
            elif line.startswith("Text:"):
                text_started = True
                text_lines.append(line[5:].strip())
            elif text_started:
                text_lines.append(line)
            # Skip Author and Published Date as they're not needed

        content = "\n".join(text_lines).strip()

        if title or url:  # Only add if we have at least a title or URL
            results.append({
                "title": title or "Untitled",
                "url": url,
                "content": content,
            })

    return results


async def exa_search(  # noqa: C901, PLR0911, PLR0912, PLR0915
    query: str,
    exa_mcp_url: str = EXA_MCP_URL,
    max_results: int = 5,
) -> dict:
    """Execute a single Exa MCP web search query.

    Uses the Exa MCP HTTP endpoint for web search. Handles SSE (Server-Sent
    Events) streaming responses. Returns the search results with page content or
    an error dict.

    Args:
        query: Search query
        exa_mcp_url: The Exa MCP endpoint URL (default: https://mcp.exa.ai/mcp)
        max_results: Maximum results to return

    """
    try:
        client = _get_exa_client()

        # MCP uses JSON-RPC 2.0 format for tool calls
        # The web_search_exa tool is enabled by default on Exa MCP
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": query,
                    "numResults": max_results,
                },
            },
        }

        logger.info(
            "Exa MCP request for query '%s': max_results=%s",
            query,
            max_results,
        )

        # Use streaming request to handle SSE responses
        async with client.stream(
            "POST",
            exa_mcp_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=60.0,
        ) as response:
            logger.info("Exa MCP response status: %s", response.status_code)

            if response.status_code != HTTP_OK:
                error_text = await response.aread()
                error_text = error_text.decode("utf-8", errors="replace")
                error_text = error_text[:MAX_ERROR_CHARS]
                logger.error(
                    "Exa MCP HTTP error for query '%s': %s - %s",
                    query,
                    response.status_code,
                    error_text,
                )
                return {
                    "error": f"HTTP {response.status_code}: {error_text[:200]}",
                    "query": query,
                }

            # Check content type to determine how to parse
            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # Handle SSE stream - collect events and find the result
                full_response = ""
                current_event_data = []

                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\n").rstrip("\r")

                    if not line:
                        # End of event (blank line)
                        if current_event_data:
                            # Join without newlines to reconstruct split JSON
                            # without injecting invalid control chars.
                            event_body = "".join(current_event_data)
                            logger.debug(
                                "Exa MCP SSE event body length: %s",
                                len(event_body),
                            )

                            # We are looking for the JSON result. It should start with {
                            if event_body.strip().startswith("{"):
                                full_response = event_body

                            current_event_data = []
                    elif line.startswith("data:"):
                        # Extract data, handling optional space
                        data = line[5:]
                        data = data.removeprefix(" ")
                        current_event_data.append(data)
                    elif line.startswith("event:"):
                        # Log event type for debugging
                        event_type = line[6:].strip()
                        logger.debug("Exa MCP SSE event: %s", event_type)

                # Check if there's any remaining data after the loop finishes
                if current_event_data and not full_response:
                    full_response = "".join(current_event_data)

                if not full_response:
                    logger.warning(
                        "Exa MCP returned empty SSE stream for query '%s'",
                        query,
                    )
                    return {"results": [], "query": query}

                try:
                    result = json.loads(full_response)
                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Exa MCP SSE JSON parse error. Data: %s",
                        full_response[:MAX_ERROR_CHARS],
                    )
                    return {"error": f"JSON parse error: {exc}", "query": query}
            else:
                # Regular JSON response
                response_text = await response.aread()
                response_text = response_text.decode("utf-8")

                if not response_text:
                    logger.warning(
                        "Exa MCP returned empty response for query '%s'",
                        query,
                    )
                    return {"results": [], "query": query}

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    logger.exception(
                        "Exa MCP JSON parse error. Response: %s",
                        response_text[:MAX_ERROR_CHARS],
                    )
                    return {"error": f"JSON parse error: {exc}", "query": query}

        # MCP JSON-RPC response format
        if "error" in result:
            error_msg = result.get("error", {}).get("message", "Unknown MCP error")
            logger.error("Exa MCP error for query '%s': %s", query, error_msg)
            return {"error": error_msg, "query": query}

        # Log the full result structure for debugging
        logger.debug(
            "Exa MCP full result keys: %s",
            result.keys() if isinstance(result, dict) else type(result),
        )

        # Extract results from the MCP response
        # The result is in result.result.content[0].text as JSON
        mcp_result = result.get("result", {})
        logger.debug(
            "Exa MCP mcp_result keys: %s",
            mcp_result.keys()
            if isinstance(mcp_result, dict)
            else type(mcp_result),
        )

        content = mcp_result.get("content", [])
        logger.debug(
            "Exa MCP content type: %s, length: %s",
            type(content),
            len(content) if isinstance(content, list) else "N/A",
        )

        if not (content and isinstance(content, list) and len(content) > 0):
            logger.warning("Exa MCP returned empty content for query '%s'", query)
            return {"results": [], "query": query}

        # Log first content item structure
        first_content = content[0]
        logger.debug(
            "Exa MCP first content item: %s",
            first_content.keys()
            if isinstance(first_content, dict)
            else type(first_content),
        )

        # Parse the text content which contains the actual search results
        if isinstance(first_content, dict):
            text_content = first_content.get("text", "")
        else:
            text_content = str(first_content)
        preview = text_content[:MAX_ERROR_CHARS] if text_content else "empty"
        logger.info("Exa MCP text_content preview: %s...", preview)

        try:
            search_data = json.loads(text_content) if text_content else {}
            logger.debug(
                "Exa MCP search_data keys: %s",
                search_data.keys()
                if isinstance(search_data, dict)
                else type(search_data),
            )

            # Normalize to match Tavily response format
            # Exa might return results directly or under a different key
            results = search_data.get("results", [])
            if not results and isinstance(search_data, list):
                # If search_data is a list directly, use it as results
                results = search_data

            logger.info(
                "Exa MCP response for query '%s': %s results",
                query,
                len(results),
            )
        except json.JSONDecodeError:
            # Exa MCP returns a structured text format, not JSON
            # Parse the structured text format returned by Exa MCP.
            logger.info(
                "Exa MCP returned text format, parsing structured text for "
                "query '%s'",
                query,
            )
            results = parse_exa_text_format(text_content)
            if results:
                logger.info(
                    "Exa MCP parsed %s results from text format",
                    len(results),
                )
                return {"results": results, "query": query}
            # Fallback: treat the entire text as a single result
            logger.warning(
                "Could not parse Exa text format, using as single result",
            )
            return {
                "results": [
                    {
                        "title": "Search Result",
                        "url": "",
                        "content": text_content,
                    },
                ],
                "query": query,
            }
        else:
            return {"results": results, "query": query}

    except httpx.TimeoutException as exc:
        logger.exception("Exa MCP timeout for query '%s'", query)
        return {"error": f"Timeout: {exc}", "query": query}
    except httpx.RequestError as exc:
        logger.exception("Exa MCP connection error for query '%s'", query)
        return {"error": f"Connection error: {exc}", "query": query}
    except Exception as exc:
        logger.exception("Exa MCP search error for query '%s'", query)
        return {"error": str(exc), "query": query}


async def perform_web_search(  # noqa: C901, PLR0913, PLR0915
    queries: list[str],
    api_keys: list[str] | None = None,
    max_results_per_query: int = 5,
    max_chars_per_url: int = 2000,
    min_score: float = 0.3,
    search_depth: str = "advanced",
    web_search_provider: str = "tavily",
    exa_mcp_url: str = EXA_MCP_URL,
) -> tuple[str, dict]:
    """Perform concurrent web searches for multiple queries.

    Supports both Tavily and Exa MCP as search providers. Returns a tuple of
    (formatted_results, metadata).

    Best practices applied:
    - Concurrent requests with asyncio.gather()
        - KeyRotator for synced bad key tracking with database persistence
            (Tavily only)
    - Configurable search depth (Tavily: "basic", "advanced", "fast", "ultra-fast")

    Args:
        queries: List of search queries
        api_keys: List of API keys for rotation (required for Tavily, optional
            for Exa)
        max_results_per_query: Maximum number of URLs per query (default: 5)
        max_chars_per_url: Maximum characters per URL content (default: 2000)
        min_score: Minimum relevance score to include a result (0.0-1.0, default:
            0.3)
        search_depth: Tavily search depth - "basic", "advanced", "fast", or
            "ultra-fast" (default: "advanced")
        web_search_provider: Which provider to use - "tavily" or "exa" (default:
            "tavily")
        exa_mcp_url: The Exa MCP endpoint URL (default: https://mcp.exa.ai/mcp)

    Returns:
        tuple: (formatted_results_string, {"queries": [...], "urls": [{...}, ...],
            "provider": "..."})

    """
    if not queries:
        return "", {}

    # Validate provider and requirements
    if web_search_provider == "tavily" and not api_keys:
        logger.warning("Tavily requires API keys but none provided")
        return "", {}

    db = get_bad_keys_db()

    # Provider-specific search functions
    async def search_single_query_tavily(
        query: str,
        depth: str,
        keys: list[str],
    ) -> dict:
        """Search with Tavily using retry logic and key rotation."""
        for key in keys:
            result = await tavily_search(query, key, max_results_per_query, depth)

            if "error" not in result:
                return result

            # Mark the key as bad
            error_msg = result.get("error", "Unknown error")[:200]
            db.mark_key_bad_synced("tavily", key, error_msg)

        # All keys failed
        logger.error("All Tavily API keys failed for query '%s'", query)
        return {"error": "All API keys exhausted", "query": query}

    async def search_single_query_exa(query: str) -> dict:
        """Search with Exa MCP."""
        return await exa_search(query, exa_mcp_url, max_results_per_query)

    async def execute_searches() -> tuple[list, list]:
        """Execute searches with the configured provider and return results and URLs."""
        if web_search_provider == "tavily":
            # Pre-fetch good keys for Tavily
            good_keys = db.get_good_keys_synced("tavily", api_keys)
            if not good_keys:
                db.reset_provider_keys_synced("tavily")
                good_keys = api_keys.copy()

            search_tasks = [
                search_single_query_tavily(query, search_depth, good_keys)
                for query in queries
            ]
        else:  # exa
            search_tasks = [
                search_single_query_exa(query)
                for query in queries
            ]

        results = await asyncio.gather(*search_tasks)

        formatted = []
        urls = []

        provider_name = web_search_provider.capitalize()
        logger.info(
            "%s search returned %s result sets for %s queries",
            provider_name,
            len(results),
            len(queries),
        )

        for i, result in enumerate(results):
            if "error" in result:
                logger.warning(
                    "%s search error for query '%s': %s",
                    provider_name,
                    queries[i],
                    result.get("error"),
                )
                continue

            query = queries[i]
            query_results = []

            result_items = result.get("results", [])
            logger.info(
                "Query '%s' returned %s items",
                query,
                len(result_items),
            )

            for item in result_items:
                # Handle both Tavily and Exa result formats
                score = item.get("score")
                if score is not None and score < min_score:
                    continue

                title = item.get("title", "No title")
                url = item.get("url", "")

                if url:
                    urls.append({"title": title, "url": url, "score": score})

                # Prefer raw_content (full page content) over content (snippet)
                # Tavily uses raw_content, Exa might use text or content
                raw_content = item.get("raw_content", "")
                content = item.get("content", "") or item.get("text", "")

                page_content = raw_content if raw_content else content
                page_content = (
                    page_content[:max_chars_per_url]
                    if page_content
                    else ""
                )

                # Format score display - handle missing scores gracefully
                score_str = f" (relevance: {score:.2f})" if score else ""
                result_text = f"\n**{title}**{score_str}\n{url}\n{page_content}\n"
                query_results.append(result_text)

            if query_results:
                formatted.append(f"\n### Search Results for: {query}")
                formatted.extend(query_results)

        logger.info("Total URLs collected: %s", len(urls))
        return formatted, urls

    # Execute the searches
    formatted_results, all_urls = await execute_searches()

    metadata = {
        "queries": queries,
        "urls": all_urls,
        "provider": web_search_provider,
    }

    if formatted_results:
        return (
            "\n\n---\nHere are the web search results in case the user asked you "
            "to search the net or something:\n\n**Web Search Results:**"
            + "".join(formatted_results),
            metadata,
        )
    return "", metadata

