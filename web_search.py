"""
Web search functionality including search decision and Tavily integration.
"""
import asyncio
import json
import logging

import httpx
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from bad_keys import get_bad_keys_db


# Web Search Decider and Tavily Search Functions
SEARCH_DECIDER_SYSTEM_PROMPT = """You are a web search query optimizer. Your job is to analyze user queries and determine if they need web search for up-to-date or factual information.

RESPOND ONLY with a valid JSON object. No other text, no markdown.

If web search is NOT needed (e.g., creative writing, opinions, general knowledge, math, coding):
{"needs_search": false}

If web search IS needed (e.g., current events, recent news, product specs, prices, real-time info):
{"needs_search": true, "queries": ["query1", "query2", ...]}

RULES for generating queries:
1. Keep queries concise—under 400 characters. Think of it as a query for an agent performing web search, not long-form prompts.
2. SINGLE ENTITY = SINGLE QUERY. If the user asks about ONE entity, output exactly ONE search query. Do NOT split into multiple queries.
   Example: "latest news" → ["latest news today"] (ONE query only)
   Example: "iPhone 16 price" → ["iPhone 16 price"] (ONE query only)
3. MULTIPLE ENTITIES = MULTIPLE QUERIES. Only if the user asks about multiple entities, create separate queries for EACH entity PLUS a query containing all entities.
   Example: "which is the best? B&K 5128 Diffuse Field Target, VDSF 5128 Demo Target Response On-Ear, VDSF 5128 Demo Target Response In-Ear, 5128 Harman In-Ear 2024 Beta, or 4128/4195 VDSF Target Response?" → ["B&K 5128 Diffuse Field Target", "VDSF 5128 Demo Target Response On-Ear", "VDSF 5128 Demo Target Response In-Ear", "5128 Harman In-Ear 2024 Beta", "4128/4195 VDSF Target Response", "B&K 5128 Diffuse Field Target vs VDSF 5128 Demo Target Response On-Ear vs VDSF 5128 Demo Target Response In-Ear vs 5128 Harman In-Ear 2024 Beta vs 4128/4195 VDSF Target Response"]
4. Make queries search-engine friendly
5. Preserve the user's original intent

Examples:
- "What's the weather today?" → {"needs_search": true, "queries": ["weather today"]}
- "Who won the 2024 Super Bowl?" → {"needs_search": true, "queries": ["2024 Super Bowl winner"]}
- "latest news" → {"needs_search": true, "queries": ["latest news today"]}
- "Write me a poem about cats" → {"needs_search": false}
- "Compare RTX 4090 and RTX 4080" → {"needs_search": true, "queries": ["RTX 4090", "RTX 4080", "RTX 4090 vs RTX 4080"]}
"""


async def decide_web_search(messages: list, decider_config: dict) -> dict:
    """
    Uses a configurable model to decide if web search is needed and generates optimized queries.
    Supports both Gemini and OpenAI-compatible APIs.
    Implements retry mechanism with API key rotation and bad key tracking.
    Returns: {"needs_search": bool, "queries": list[str]} or {"needs_search": False}
    
    decider_config should contain:
        - provider: "gemini" or other (OpenAI-compatible)
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
    
    # Use synced bad key tracking so keys marked bad by main model are also recognized here
    # and keys marked bad here are recognized by main model
    
    # Get good keys (filter out known bad ones from BOTH main and decider)
    good_keys = get_bad_keys_db().get_good_keys_synced(provider, api_keys)
    
    # If all keys are bad, reset and try again with all keys
    if not good_keys:
        logging.warning(f"All API keys for '{provider}' (synced) are marked as bad. Resetting...")
        get_bad_keys_db().reset_provider_keys_synced(provider)
        good_keys = api_keys.copy()
    
    attempt_count = 0
    
    while good_keys:
        attempt_count += 1
        current_api_key = good_keys[(attempt_count - 1) % len(good_keys)]
        
        try:
            if provider == "gemini":
                # Build Gemini-format contents
                search_decider_contents = []
                for msg in messages[::-1]:  # Reverse to get chronological order
                    role = "model" if msg.get("role") == "assistant" else "user"
                    content = msg.get("content", "")
                    parts = []
                    
                    if isinstance(content, list):
                        # Handle multimodal content (text + images)
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text" and part.get("text"):
                                    parts.append(types.Part.from_text(text=str(part["text"])[:5000]))
                                elif part.get("type") == "image_url" and part.get("image_url", {}).get("url"):
                                    # Extract base64 image data
                                    image_url = part["image_url"]["url"]
                                    if image_url.startswith("data:"):
                                        try:
                                            header, b64_data = image_url.split(",", 1)
                                            mime_type = header.split(":")[1].split(";")[0]
                                            import base64
                                            image_bytes = base64.b64decode(b64_data)
                                            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                                        except Exception:
                                            pass
                    elif content:
                        parts.append(types.Part.from_text(text=str(content)[:5000]))
                    
                    if parts:
                        search_decider_contents.append(types.Content(role=role, parts=parts))
                
                if not search_decider_contents:
                    return {"needs_search": False}
                
                # Add instruction to analyze the conversation
                search_decider_contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Based on the conversation above, analyze the last user query and respond with your JSON decision.")]
                ))
                
                config_kwargs = dict(
                    system_instruction=SEARCH_DECIDER_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                
                # Add thinking config for Gemini 3 models
                if "gemini-3" in model:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")
                
                gemini_config = types.GenerateContentConfig(**config_kwargs)
                
                client = genai.Client(api_key=current_api_key, http_options=dict(api_version="v1beta"))
                
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=search_decider_contents,
                    config=gemini_config
                )
                
                response_text = response.text.strip()
            else:
                # OpenAI-compatible API
                openai_messages = [{"role": "system", "content": SEARCH_DECIDER_SYSTEM_PROMPT}]
                
                for msg in messages[::-1]:  # Reverse to get chronological order
                    role = msg.get("role", "user")
                    if role == "system":
                        continue  # Skip system messages from chat history
                    
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Keep multimodal format for OpenAI
                        openai_messages.append({"role": role, "content": content})
                    elif content:
                        openai_messages.append({"role": role, "content": str(content)[:5000]})
                
                if len(openai_messages) <= 1:  # Only system prompt
                    return {"needs_search": False}
                
                # Add instruction to analyze the conversation
                openai_messages.append({
                    "role": "user",
                    "content": "Based on the conversation above, analyze the last user query and respond with your JSON decision."
                })
                
                openai_client = AsyncOpenAI(base_url=base_url, api_key=current_api_key)
                
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=openai_messages,
                    temperature=0.1,
                )
                
                response_text = response.choices[0].message.content.strip()
            
            # Parse response (common for both providers)
            if response_text.startswith("```"):
                # Remove markdown code blocks if present
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            logging.exception(f"Error in web search decider (attempt {attempt_count}): {e}")
            
            # Mark the current key as bad (synced for both main and decider)
            error_msg = str(e)[:200] if e else "Unknown error"
            get_bad_keys_db().mark_key_bad_synced(provider, current_api_key, error_msg)
            
            # Remove the bad key from good_keys list for this session
            if current_api_key in good_keys:
                good_keys.remove(current_api_key)
            
            # If all keys are exhausted, try resetting once
            if not good_keys:
                if attempt_count >= len(api_keys) * 2:
                    logging.error("Web search decider failed after exhausting all keys, skipping web search")
                    return {"needs_search": False}
                else:
                    logging.warning(f"All decider keys exhausted. Resetting synced keys for retry...")
                    get_bad_keys_db().reset_provider_keys_synced(provider)
                    good_keys = api_keys.copy()
    
    return {"needs_search": False}


# Shared httpx client for Tavily API calls - reuses connections for better performance
_tavily_client: httpx.AsyncClient | None = None


def _get_tavily_client() -> httpx.AsyncClient:
    """Get or create the shared Tavily httpx client."""
    global _tavily_client
    if _tavily_client is None or _tavily_client.is_closed:
        _tavily_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
    return _tavily_client


async def tavily_search(query: str, tavily_api_key: str, max_results: int = 5, search_depth: str = "basic") -> dict:
    """
    Execute a single Tavily search query.
    Returns the search results with page content or an error dict.
    
    Best practices applied:
    - auto_parameters enabled for automatic query optimization
    - search_depth configurable (basic/advanced/fast/ultra-fast)
    - Reuses shared httpx client for connection pooling
    """
    try:
        client = _get_tavily_client()
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": False,
                "include_raw_content": "markdown",  # Get full page content in markdown format
                "auto_parameters": True,  # Let Tavily automatically optimize parameters based on query intent
            },
            headers={
                "Authorization": f"Bearer {tavily_api_key}",
                "Content-Type": "application/json"
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.exception(f"Tavily search error for query '{query}': {e}")
        return {"error": str(e), "query": query}


async def perform_web_search(
    queries: list[str], 
    tavily_api_keys: list[str], 
    max_results_per_query: int = 5, 
    max_chars_per_url: int = 2000,
    search_depth: str = "basic",
    min_score: float = 0.5
) -> tuple[str, dict]:
    """
    Perform concurrent web searches for multiple queries using Tavily.
    Rotates through multiple API keys for load distribution.
    Returns a tuple of (formatted_results, metadata).
    
    Best practices applied:
    - Concurrent requests with asyncio.gather()
    - API key rotation for load distribution
    - Score-based filtering (min_score) for relevance
    - Configurable search_depth (basic/advanced/fast/ultra-fast)
    
    Args:
        queries: List of search queries
        tavily_api_keys: List of Tavily API keys for rotation
        max_results_per_query: Maximum number of URLs per query (default: 5)
        max_chars_per_url: Maximum characters per URL content (default: 2000)
        search_depth: Tavily search depth - "basic" (balanced), "advanced" (highest relevance), 
                      "fast" (low latency), or "ultra-fast" (lowest latency)
        min_score: Minimum relevance score to include a result (0.0-1.0, default: 0.5)
    
    Returns:
        tuple: (formatted_results_string, {"queries": [...], "urls": [{...}, ...]})
    """
    if not queries or not tavily_api_keys:
        return "", {}
    
    # Execute all searches concurrently, rotating through API keys
    search_tasks = [
        tavily_search(query, tavily_api_keys[i % len(tavily_api_keys)], max_results_per_query, search_depth) 
        for i, query in enumerate(queries)
    ]
    results = await asyncio.gather(*search_tasks)
    
    formatted_results = []
    all_urls = []  # Track all URLs used for the "Show Sources" button
    
    for i, result in enumerate(results):
        if "error" in result:
            continue
        
        query = queries[i]
        query_results = []
        
        for item in result.get("results", []):
            # Best practice: Score-based filtering for relevance
            score = item.get("score", 0)
            if score < min_score:
                logging.debug(f"Skipping low-score result (score={score:.2f}): {item.get('title', 'No title')}")
                continue
            
            title = item.get("title", "No title")
            url = item.get("url", "")
            
            # Track URL for sources (with score for reference)
            if url:
                all_urls.append({"title": title, "url": url, "score": score})
            
            # Prefer raw_content (full page content) over content (snippet)
            raw_content = item.get("raw_content", "")
            content = item.get("content", "")
            
            # Use raw_content if available, otherwise fall back to content snippet
            page_content = raw_content if raw_content else content
            # Limit to max_chars_per_url (default 2000 chars per URL)
            page_content = page_content[:max_chars_per_url] if page_content else ""
            
            result_text = f"\n**{title}** (relevance: {score:.2f})\n{url}\n{page_content}\n"
            query_results.append(result_text)
        
        if query_results:
            formatted_results.append(f"\n### Search Results for: {query}")
            formatted_results.extend(query_results)
    
    metadata = {
        "queries": queries,
        "urls": all_urls
    }
    
    if formatted_results:
        return "\n\n---\n**Web Search Results:**" + "".join(formatted_results), metadata
    return "", metadata
