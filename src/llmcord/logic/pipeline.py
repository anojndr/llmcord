"""Message processing orchestration."""
import asyncio
import io
import logging
import re
from base64 import b64encode

import discord
import httpx


from llmcord.core.config import (
    BROWSER_HEADERS,
    EMBED_COLOR_INCOMPLETE,
    PROVIDERS_SUPPORTING_USERNAMES,
    VISION_MODEL_TAGS,
    ensure_list,
    get_config,
    is_gemini_model,
)
from llmcord.globals import reddit_client
from llmcord.core.models import MsgNode
from llmcord.services.database import get_bad_keys_db
from llmcord.services.search import (
    decide_web_search,
    get_current_datetime_strings,
    perform_tavily_research,
    perform_web_search,
)
from llmcord.services.extractors import (
    TwitterApiProtocol,
    download_attachments,
    extract_pdf_text,
    extract_reddit_post,
    extract_youtube_transcript,
    fetch_tweet_with_replies,
    perform_yandex_lookup,
    process_attachments,
)
from llmcord.logic.generation import generate_response
from llmcord.logic.utils import (
    append_search_to_content,
    build_node_text_parts,
    extract_research_command,
    replace_content_text,
)
from llmcord.logic.permissions import should_process_message

logger = logging.getLogger(__name__)


async def process_message(  # noqa: C901, PLR0912, PLR0913, PLR0915
    new_msg: discord.Message,
    discord_bot: discord.Client,
    httpx_client: httpx.AsyncClient,
    twitter_api: TwitterApiProtocol,
    msg_nodes: dict[int, MsgNode],
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
    override_provider_slash_model: str | None = None,
    fallback_chain: list[tuple[str, str, str]] | None = None,
) -> None:
    """Process a message."""
    # Per-request edit timing to avoid interference between concurrent requests
    last_edit_time = 0

    should_process, processing_msg_or_none = await should_process_message(
        new_msg, discord_bot
    )
    if not should_process:
        return

    # should_process guarantees processing_msg is not None if it returns True
    processing_msg = processing_msg_or_none
    assert processing_msg is not None

    config = get_config()  # Now cached, no need for to_thread

    # Thread-safe read of current model (unless an override is provided)
    if override_provider_slash_model:
        provider_slash_model = override_provider_slash_model
    else:
        async with curr_model_lock:
            provider_slash_model = curr_model_ref[0]

    # Validate provider/model format
    try:
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    except ValueError:
        logger.exception(
            "Invalid model format: %s. Expected 'provider/model'.",
            provider_slash_model,
        )
        embed = discord.Embed(
            description=(
                "❌ Invalid model configuration: "
                f"'{provider_slash_model}'. Expected format: 'provider/model'.\n"
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    # Validate provider exists in config
    providers = config.get("providers", {})
    if provider not in providers:
        logger.error("Provider '%s' not found in config.", provider)
        embed = discord.Embed(
            description=(
                f"❌ Provider '{provider}' is not configured. "
                "Please contact an administrator."
            ),
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    provider_config = providers[provider]

    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key")) or ["sk-no-key-required"]

    model_parameters = config["models"].get(provider_slash_model, None)

    # Support model aliasing: if config specifies a different actual model name, use it
    # This allows e.g. "gemini-3-flash-minimal" to map to "gemini-3-flash-preview"
    actual_model = model_parameters.get("model", model) if model_parameters else model

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (
        (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None
    )

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(
        provider_slash_model.lower().startswith(x)
        for x in PROVIDERS_SUPPORTING_USERNAMES
    )

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(
                    discord_bot.user.mention
                ).lstrip()
                cleaned_content = re.sub(
                    r"\bat ai\b", "", cleaned_content, flags=re.IGNORECASE
                ).lstrip()

                if cleaned_content.lower().startswith("googlelens"):
                    cleaned_content = cleaned_content[10:].strip()

                    # Check for cached lens results first
                    _, _, cached_lens_results = get_bad_keys_db().get_message_search_data(
                        curr_msg.id
                    )
                    if cached_lens_results:
                        # Use cached lens results
                        cleaned_content = cleaned_content + cached_lens_results
                        curr_node.lens_results = cached_lens_results
                        logger.debug(
                            "Using cached lens results for message %s",
                            curr_msg.id,
                        )
                    elif image_url := next(
                        (
                            att.url
                            for att in curr_msg.attachments
                            if att.content_type
                            and att.content_type.startswith("image")
                        ),
                        None,
                    ):
                        try:
                            lens_results, twitter_content = await perform_yandex_lookup(
                                image_url,
                                httpx_client,
                                twitter_api,
                                max_tweet_replies,
                            )

                            if lens_results:
                                result_text = (
                                    "\n\nanswer the user's query based on the yandex reverse image results:\n"
                                    + "\n".join(lens_results)
                                )
                                if twitter_content:
                                    result_text += (
                                        "\n\n--- Extracted Twitter/X Content ---"
                                        + "".join(twitter_content)
                                    )
                                cleaned_content += result_text

                                # Store lens results for persistence
                                curr_node.lens_results = result_text

                                # Save lens results to database for persistence in chat history
                                get_bad_keys_db().save_message_search_data(
                                    curr_msg.id,
                                    lens_results=result_text,
                                )
                                logger.info(
                                    "Saved lens results for message %s",
                                    curr_msg.id,
                                )
                        except Exception:
                            logger.exception("Error fetching Yandex results")

                allowed_types = (
                    "text",
                    "image",
                    "application/pdf",
                )  # PDF always allowed - Gemini uses native, others get text extraction
                if is_gemini_model(actual_model):
                    allowed_types += ("audio", "video")

                good_attachments = [
                    att
                    for att in curr_msg.attachments
                    if att.content_type
                    and any(att.content_type.startswith(x) for x in allowed_types)
                ]

                # Download attachments
                successful_pairs = await download_attachments(
                    good_attachments, httpx_client
                )

                # Update collections based on successful downloads
                good_attachments = [pair[0] for pair in successful_pairs]
                attachment_responses = [pair[1] for pair in successful_pairs]

                processed_attachments = await process_attachments(successful_pairs)

                # Initial text building before async content fetches (using DRY helper)
                curr_node.text = build_node_text_parts(
                    cleaned_content,
                    curr_msg.embeds,
                    curr_msg.components,
                    text_attachments=[
                        att["text"]
                        for att in processed_attachments
                        if att["content_type"]
                        and att["content_type"].startswith("text")
                        and att["text"]
                    ],
                )

                curr_node.images = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{att['content_type']};base64,"
                                f"{b64encode(att['content']).decode('utf-8')}"
                            ),
                        },
                    }
                    for att in processed_attachments
                    if att["content_type"]
                    and att["content_type"].startswith("image")
                ]

                curr_node.raw_attachments = [
                    {"content_type": att["content_type"], "content": att["content"]}
                    for att in processed_attachments
                ]

                # Fetch YouTube transcripts
                video_ids = re.findall(
                    r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})",
                    cleaned_content,
                )
                if video_ids:
                    yt_results = await asyncio.gather(
                        *[
                            extract_youtube_transcript(vid, httpx_client)
                            for vid in video_ids
                        ],
                    )
                    yt_transcripts = [t for t in yt_results if t is not None]
                else:
                    yt_transcripts = []

                # Fetch tweets
                tweets = []
                for tweet_id in re.findall(
                    r"(?:https?:\/\/)?(?:[a-zA-Z0-9-]+\.)?(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)",
                    cleaned_content,
                ):
                    tweet_text = await fetch_tweet_with_replies(
                        twitter_api,
                        int(tweet_id),
                        max_replies=max_tweet_replies,
                        include_url=False,
                    )
                    if tweet_text:
                        tweets.append(tweet_text)

                reddit_posts = []
                for post_url in re.findall(
                    r"(https?:\/\/(?:[a-zA-Z0-9-]+\.)?(?:reddit\.com\/r\/[a-zA-Z0-9_]+\/comments\/[a-zA-Z0-9_]+(?:[\w\-\.\/\?\=\&%]*)|redd\.it\/[a-zA-Z0-9_]+))",
                    cleaned_content,
                ):
                    post_text = await extract_reddit_post(post_url, httpx_client, reddit_client)
                    if post_text:
                        reddit_posts.append(post_text)

                # PDF text extraction for non-Gemini models
                pdf_texts = []
                if not is_gemini_model(actual_model):
                    pdf_attachments = [
                        att
                        for att in processed_attachments
                        if att["content_type"] == "application/pdf"
                    ]
                    if pdf_attachments:
                        # Extract text from PDFs in parallel
                        pdf_extraction_tasks = [
                            extract_pdf_text(att["content"]) for att in pdf_attachments
                        ]
                        pdf_results = await asyncio.gather(*pdf_extraction_tasks)

                        for i, pdf_text in enumerate(pdf_results):
                            if pdf_text:
                                # Format PDF content nicely for the LLM
                                pdf_texts.append(
                                    f"--- PDF Attachment {i + 1} Content ---\n{pdf_text}",
                                )
                                logger.info(
                                    "Extracted text from PDF attachment (%s chars)",
                                    len(pdf_text),
                                )
                            else:
                                logger.warning(
                                    "Could not extract text from PDF attachment %s",
                                    i + 1,
                                )

                # Final text building with all async content (using DRY helper)
                curr_node.text = build_node_text_parts(
                    cleaned_content,
                    curr_msg.embeds,
                    curr_msg.components,
                    text_attachments=[
                        resp.text
                        for att, resp in zip(
                            good_attachments,
                            attachment_responses,
                            strict=False,
                        )
                        if att.content_type.startswith("text") and resp.text
                    ],
                    extra_parts=yt_transcripts + tweets + reddit_posts + pdf_texts,
                )

                if not curr_node.text and curr_node.images:
                    curr_node.text = "What is in this image?"

                if (
                    not curr_node.text
                    and not curr_node.images
                    and not curr_node.raw_attachments
                    and curr_msg == new_msg
                ):
                    curr_node.text = "Hello"

                curr_node.role = (
                    "assistant"
                    if curr_msg.author == discord_bot.user
                    else "user"
                )

                curr_node.user_id = (
                    curr_msg.author.id if curr_node.role == "user" else None
                )

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(
                    good_attachments,
                )

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (
                            prev_msg_in_channel := (
                                [
                                    m
                                    async for m in curr_msg.channel.history(
                                        before=curr_msg,
                                        limit=1,
                                    )
                                ]
                                or [None]
                            )[0]
                        )
                        and prev_msg_in_channel.type
                        in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author
                        == (
                            discord_bot.user
                            if curr_msg.channel.type == discord.ChannelType.private
                            else curr_msg.author
                        )
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = (
                            curr_msg.channel.type == discord.ChannelType.public_thread
                        )
                        parent_is_thread_start = (
                            is_public_thread
                            and curr_msg.reference is None
                            and curr_msg.channel.parent.type == discord.ChannelType.text
                        )

                        if parent_msg_id := (
                            curr_msg.channel.id
                            if parent_is_thread_start
                            else getattr(curr_msg.reference, "message_id", None)
                        ):
                            if parent_is_thread_start:
                                curr_node.parent_msg = (
                                    curr_msg.channel.starter_message
                                    or await curr_msg.channel.parent.fetch_message(
                                        parent_msg_id,
                                    )
                                )
                            else:
                                curr_node.parent_msg = (
                                    curr_msg.reference.cached_message
                                    or await curr_msg.channel.fetch_message(
                                        parent_msg_id,
                                    )
                                )

                except (discord.NotFound, discord.HTTPException):
                    logger.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Load stored search data from database if not already cached
            # This is outside the text==None block so it runs even for cached nodes
            if curr_node.search_results is None or curr_node.lens_results is None:
                (
                    stored_search_results,
                    stored_tavily_metadata,
                    stored_lens_results,
                ) = get_bad_keys_db().get_message_search_data(curr_msg.id)
                if stored_search_results and curr_node.search_results is None:
                    curr_node.search_results = stored_search_results
                    curr_node.tavily_metadata = stored_tavily_metadata
                    logger.info(
                        "Loaded stored search data for message %s",
                        curr_msg.id,
                    )
                if stored_lens_results and curr_node.lens_results is None:
                    curr_node.lens_results = stored_lens_results
                    logger.info(
                        "Loaded stored lens results for message %s",
                        curr_msg.id,
                    )

            # Build message content (now unified for all providers via LiteLLM)
            # Check if there are Gemini-specific file attachments (audio, video, PDF)
            gemini_file_attachments = []
            if is_gemini_model(actual_model) and curr_node.raw_attachments:
                for att in curr_node.raw_attachments:
                    if att["content_type"].startswith(
                        ("audio", "video"),
                    ) or att["content_type"] == "application/pdf":
                        # LiteLLM supports inline data format for Gemini
                        encoded_data = b64encode(att["content"]).decode("utf-8")
                        gemini_file_attachments.append(
                            {
                                "type": "file",
                                "file": {
                                    "file_data": (
                                        f"data:{att['content_type']};base64,{encoded_data}"
                                    ),
                                },
                            },
                        )

            # Determine if we need multimodal content format
            has_images = bool(curr_node.images[:max_images])
            has_gemini_files = bool(gemini_file_attachments)

            if has_images or has_gemini_files:
                # Build multimodal content array
                content = []

                # Add text part if present
                if curr_node.text[:max_text]:
                    content.append(
                        {"type": "text", "text": curr_node.text[:max_text]},
                    )

                # Add images
                content.extend(curr_node.images[:max_images])

                # Add Gemini file attachments (audio, video, PDF)
                content.extend(gemini_file_attachments)
                if gemini_file_attachments:
                    logger.info(
                        "Added %s Gemini file attachment(s) (audio/video/PDF) to message",
                        len(gemini_file_attachments),
                    )

                # Ensure we have at least some content
                if not content:
                    content = [{"type": "text", "text": "What is in this file?"}]
            else:
                content = curr_node.text[:max_text]

            # Include stored search results from history messages
            if curr_node.search_results and curr_node.role == "user":
                content = append_search_to_content(content, curr_node.search_results)

            if content:
                message = {"content": content, "role": curr_node.role}
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if curr_node.text and len(curr_node.text) > max_text:
                user_warnings.add(
                    f"⚠️ Max {max_text:,} characters per message",
                )
            if len(curr_node.images) > max_images:
                user_warnings.add(
                    (
                        f"⚠️ Max {max_images} image"
                        f"{'' if max_images == 1 else 's'} per message"
                    )
                    if max_images > 0
                    else "⚠️ Can't see images",
                )
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (
                curr_node.parent_msg is not None and len(messages) == max_messages
            ):
                user_warnings.add(
                    f"⚠️ Only using last {len(messages)} message"
                    f"{'' if len(messages) == 1 else 's'}",
                )

            curr_msg = curr_node.parent_msg

    logger.info(
        "Message received (user ID: %s, attachments: %s, conversation length: %s):\n%s",
        new_msg.author.id,
        len(new_msg.attachments),
        len(messages),
        new_msg.content,
    )

    # Handle edge case: no valid messages could be built
    if not messages:
        logger.warning("No valid messages could be built from the conversation.")
        embed = discord.Embed(
            description="❌ Could not process your message. Please try again.",
            color=EMBED_COLOR_INCOMPLETE,
        )
        await processing_msg.edit(embed=embed)
        return

    system_prompt = config.get("system_prompt")

    if system_prompt:
        date_str, time_str = get_current_datetime_strings()

        system_prompt = (
            system_prompt.replace("{date}", date_str)
            .replace("{time}", time_str)
            .strip()
        )
        if accept_usernames:
            system_prompt += (
                "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."
            )

        messages.append({"role": "system", "content": system_prompt})

    # Web Search Integration for non-Gemini models and Gemini preview models
    # Supports both Tavily and Exa MCP as search providers
    tavily_api_keys = ensure_list(config.get("tavily_api_key"))
    exa_mcp_url = config.get("exa_mcp_url", "")
    web_search_provider = config.get(
        "web_search_provider",
        "tavily",
    )  # Default to Tavily

    # Determine if web search is available based on provider config
    web_search_available = False
    if (web_search_provider == "tavily" and tavily_api_keys) or (
        web_search_provider == "exa" and exa_mcp_url
    ):
        web_search_available = True
    elif web_search_provider == "auto":
        # Auto-detect: prefer Tavily if keys are available, otherwise try Exa
        if tavily_api_keys:
            web_search_provider = "tavily"
            web_search_available = True
        elif exa_mcp_url:
            web_search_provider = "exa"
            web_search_available = True

    is_preview_model = "preview" in actual_model.lower()
    is_non_gemini = not is_gemini_model(actual_model)
    # Check for googlelens query - strip both mention and "at ai" prefix
    content_for_lens_check = (
        new_msg.content.lower()
        .removeprefix(discord_bot.user.mention.lower())
        .strip()
    )
    if content_for_lens_check.startswith("at ai"):
        content_for_lens_check = content_for_lens_check[5:].strip()
    is_googlelens_query = content_for_lens_check.startswith("googlelens")

    search_metadata = None
    research_model, research_query = extract_research_command(
        new_msg.content,
        discord_bot.user.mention if discord_bot.user else "",
    )

    # Check for existing search results to handle retries correctly
    if new_msg.id in msg_nodes and msg_nodes[new_msg.id].search_results:
        search_metadata = msg_nodes[new_msg.id].tavily_metadata
        has_existing_search = True
    else:
        has_existing_search = False

    if research_model and not has_existing_search:
        if not research_query:
            user_warnings.add(
                "⚠️ Provide a research query after researchpro/researchmini",
            )
        elif not tavily_api_keys:
            user_warnings.add("⚠️ Tavily API key missing for research")
        else:
            for msg in messages:
                if msg.get("role") == "user":
                    msg["content"] = replace_content_text(
                        msg.get("content", ""),
                        research_query,
                    )
                    break

            research_results, search_metadata = await perform_tavily_research(
                query=research_query,
                api_keys=tavily_api_keys,
                model=research_model,
            )

            if research_results:
                for msg in messages:
                    if msg.get("role") == "user":
                        msg["content"] = append_search_to_content(
                            msg.get("content", ""),
                            research_results,
                        )

                        logger.info(
                            "Tavily research results appended to user message",
                        )

                        get_bad_keys_db().save_message_search_data(
                            new_msg.id,
                            research_results,
                            search_metadata,
                        )

                        if new_msg.id in msg_nodes:
                            msg_nodes[new_msg.id].search_results = research_results
                            msg_nodes[new_msg.id].tavily_metadata = search_metadata
                        break

    if (
        not research_model
        and web_search_available
        and (is_non_gemini or is_preview_model)
        and not is_googlelens_query
        and not has_existing_search
    ):
        # Get web search decider model - first check user preference, then config default
        db = get_bad_keys_db()
        user_id = str(new_msg.author.id)
        user_decider_model = db.get_user_search_decider_model(user_id)
        default_decider = config.get(
            "web_search_decider_model",
            "gemini/gemini-3-flash-preview",
        )

        # Use user preference if set and valid, otherwise use config default
        if user_decider_model and user_decider_model in config.get("models", {}):
            decider_model_str = user_decider_model
        else:
            decider_model_str = default_decider

        decider_provider, decider_model = (
            decider_model_str.split("/", 1)
            if "/" in decider_model_str
            else ("gemini", decider_model_str)
        )

        # Get provider config for the decider
        decider_provider_config = config.get("providers", {}).get(decider_provider, {})
        decider_api_keys = ensure_list(decider_provider_config.get("api_key"))

        decider_config = {
            "provider": decider_provider,
            "model": decider_model,
            "api_keys": decider_api_keys,
            "base_url": decider_provider_config.get("base_url"),
        }

        if decider_api_keys:
            # Run the search decider with chat history (uses all keys with rotation)
            search_decision = await decide_web_search(messages, decider_config)

            if search_decision.get("needs_search") and search_decision.get("queries"):
                queries = search_decision["queries"]
                logger.info(
                    "Web search triggered with %s. Queries: %s",
                    web_search_provider,
                    queries,
                )

                # Perform web search with the configured provider
                search_depth = config.get("tavily_search_depth", "advanced")
                search_results, search_metadata = await perform_web_search(
                    queries,
                    api_keys=tavily_api_keys,
                    max_results_per_query=5,
                    max_chars_per_url=2000,
                    search_depth=search_depth,
                    web_search_provider=web_search_provider,
                    exa_mcp_url=exa_mcp_url
                    or "https://mcp.exa.ai/mcp?tools=web_search_exa,web_search_advanced_exa,get_code_context_exa,deep_search_exa,crawling_exa,company_research_exa,linkedin_search_exa,deep_researcher_start,deep_researcher_check",
                )

                if search_results:
                    # Append search results to the first (most recent) user message
                    for msg in messages:
                        if msg.get("role") == "user":
                            msg["content"] = append_search_to_content(
                                msg.get("content", ""),
                                search_results,
                            )

                            logger.info(
                                "Web search results appended to user message",
                            )

                            # Save search results to database for persistence in chat history
                            get_bad_keys_db().save_message_search_data(
                                new_msg.id,
                                search_results,
                                search_metadata,
                            )

                            # Also update the cached MsgNode so follow-up requests in the same session get the data
                            if new_msg.id in msg_nodes:
                                msg_nodes[new_msg.id].search_results = search_results
                                msg_nodes[new_msg.id].tavily_metadata = search_metadata
                            break

    # Continue with response generation
    async def retry_callback() -> None:
        await process_message(
            new_msg=new_msg,
            discord_bot=discord_bot,
            httpx_client=httpx_client,
            twitter_api=twitter_api,

            msg_nodes=msg_nodes,
            curr_model_lock=curr_model_lock,
            curr_model_ref=curr_model_ref,
            override_provider_slash_model="gemini/gemma-3-27b-it",
            fallback_chain=[
                (
                    "mistral",
                    "mistral-large-latest",
                    "mistral/mistral-large-latest",
                ),
            ],
        )

    await generate_response(
        new_msg=new_msg,
        _discord_bot=discord_bot,
        msg_nodes=msg_nodes,
        messages=messages,
        user_warnings=user_warnings,
        provider=provider,
        _model=model,
        actual_model=actual_model,
        provider_slash_model=provider_slash_model,
        base_url=base_url,
        api_keys=api_keys,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        _extra_query=extra_query,
        _extra_body=extra_body,
        _system_prompt=system_prompt,
        config=config,
        _max_text=max_text,
        tavily_metadata=search_metadata,
        last_edit_time=last_edit_time,
        processing_msg=processing_msg,
        retry_callback=retry_callback,
        fallback_chain=fallback_chain,
    )
