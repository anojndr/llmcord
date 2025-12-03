import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import os
import re
import logging
from typing import Any, Literal, Optional

from aiohttp import web
import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from google import genai
from google.genai import types
import httpx
from openai import AsyncOpenAI
from twscrape import API, gather
from youtube_transcript_api import YouTubeTranscriptApi
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        with open(os.path.join("/etc/secrets", filename), encoding="utf-8") as file:
            return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None, allowed_mentions=discord.AllowedMentions(replied_user=False))

httpx_client = httpx.AsyncClient()
twitter_api = API(proxy=config.get("twitter_proxy"))


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    raw_attachments: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        curr_model = model
        output = f"Model switched to: `{model}`"
        logging.info(output)

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()

    if twitter_accounts := config.get("twitter_accounts"):
        for acc in twitter_accounts:
            if await twitter_api.pool.get_account(acc["username"]):
                continue
            await twitter_api.pool.add_account(
                acc["username"],
                acc["password"],
                acc["email"],
                acc["email_password"],
                cookies=acc.get("cookies"),
            )
        await twitter_api.pool.login_all()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions and "at ai" not in new_msg.content.lower()) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config.get("base_url")
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_tweet_replies = config.get("max_tweet_replies", 50)

    # Build message chain and set user warnings
    messages = []
    gemini_contents = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                if cleaned_content.lower().startswith("at ai"):
                    cleaned_content = cleaned_content[5:].lstrip()

                if cleaned_content.lower().startswith("googlelens") and (serpapi_key := config.get("serpapi_api_key")):
                    lens_query = cleaned_content[10:].strip()

                    if image_url := next((att.url for att in curr_msg.attachments if att.content_type and att.content_type.startswith("image")), None):
                        try:
                            params = {
                                "engine": "google_lens",
                                "api_key": serpapi_key,
                                "url": image_url,
                            }
                            if lens_query:
                                params["q"] = lens_query

                            lens_resp = await httpx_client.get("https://serpapi.com/search", params=params, timeout=60)
                            lens_data = lens_resp.json()

                            if not lens_data.get("error"):
                                lens_results = []
                                if visual_matches := lens_data.get("visual_matches"):
                                    lens_results.append("Visual Matches:")
                                    for match in visual_matches:
                                        lens_results.append(f"- [{match.get('title')}]({match.get('link')}) ({match.get('source')})")

                                if related_content := lens_data.get("related_content"):
                                    lens_results.append("\nRelated Content:")
                                    for content in related_content:
                                        lens_results.append(f"- [{content.get('query')}]({content.get('link')})")

                                if lens_results:
                                    cleaned_content = (cleaned_content + "\n\nGoogle Lens Results:\n" + "\n".join(lens_results))
                        except Exception:
                            logging.exception("Error fetching Google Lens results")

                allowed_types = ("text", "image")
                if provider == "gemini":
                    allowed_types += ("audio", "video", "application/pdf")

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in allowed_types)]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.raw_attachments = [
                    dict(content_type=att.content_type, content=resp.content)
                    for att, resp in zip(good_attachments, attachment_responses)
                ]

                yt_transcripts = []
                for video_id in re.findall(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})", cleaned_content):
                    try:
                        ytt_api = YouTubeTranscriptApi()
                        transcript_obj = await asyncio.to_thread(ytt_api.fetch, video_id)
                        transcript = transcript_obj.to_raw_data()

                        response = await httpx_client.get(f"https://www.youtube.com/watch?v={video_id}", follow_redirects=True)
                        html = response.text
                        title_match = re.search(r'<meta name="title" content="(.*?)">', html)
                        title = title_match.group(1) if title_match else "Unknown Title"
                        channel_match = re.search(r'<link itemprop="name" content="(.*?)">', html)
                        channel = channel_match.group(1) if channel_match else "Unknown Channel"

                        yt_transcripts.append(f"YouTube Video ID: {video_id}\nTitle: {title}\nChannel: {channel}\nTranscript:\n" + " ".join(x["text"] for x in transcript))
                    except Exception:
                        pass

                tweets = []
                for tweet_id in re.findall(r"(?:https?:\/\/)?(?:www\.)?(?:twitter\.com|x\.com)\/[a-zA-Z0-9_]+\/status\/([0-9]+)", cleaned_content):
                    try:
                        tweet = await twitter_api.tweet_details(int(tweet_id))
                        tweets.append(f"Tweet from @{tweet.user.username}:\n{tweet.rawContent}")

                        if max_tweet_replies > 0:
                            replies = await gather(twitter_api.tweet_replies(int(tweet_id), limit=max_tweet_replies))
                            if replies:
                                tweets.append("\nReplies:")
                                for reply in replies:
                                    tweets.append(f"- @{reply.user.username}: {reply.rawContent}")
                    except Exception:
                        pass

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    + yt_transcripts
                    + tweets
                )

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if provider == "gemini":
                parts = []
                if curr_node.text:
                    parts.append(types.Part.from_text(text=curr_node.text[:max_text]))

                for att in curr_node.raw_attachments:
                    if att["content_type"].startswith("text"):
                        continue

                    parts.append(types.Part.from_bytes(data=att["content"], mime_type=att["content_type"]))

                if parts:
                    role = "model" if curr_node.role == "assistant" else "user"
                    gemini_contents.append(types.Content(role=role, parts=parts))

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    system_prompt = config.get("system_prompt")

    if provider != "gemini" and (exa_key := config.get("exa_api_key")):
        search_query = new_msg.content.removeprefix(discord_bot.user.mention).lstrip()
        if search_query.lower().startswith("at ai"):
            search_query = search_query[5:].lstrip()

        if search_query:
            try:
                exa_resp = await httpx_client.post(
                    "https://api.exa.ai/search",
                    headers={"x-api-key": exa_key, "Content-Type": "application/json"},
                    json={
                        "query": search_query,
                        "type": "auto",
                        "contents": {"text": True}
                    },
                    timeout=10
                )
                exa_data = exa_resp.json()
                if results := exa_data.get("results"):
                    search_context = "\n\nSearch Results:\n"
                    for res in results[:10]:
                        search_context += f"- [{res.get('title')}]({res.get('url')}): {res.get('text')}\n"

                    if system_prompt:
                        system_prompt += search_context
                    else:
                        system_prompt = search_context
            except Exception:
                logging.exception("Error fetching Exa search results")

    if system_prompt:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    async def get_stream():
        if provider == "gemini":
            client = genai.Client(api_key=api_key, http_options=dict(api_version="v1beta"))
            tools = [types.Tool(google_search=types.GoogleSearch()), types.Tool(url_context=types.UrlContext())]
            gemini_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools,
                temperature=model_parameters.get("temperature") if model_parameters else None,
            )
            async for chunk in await client.aio.models.generate_content_stream(
                model=model,
                contents=gemini_contents[::-1],
                config=gemini_config
            ):
                text = chunk.text or ""
                reason = chunk.candidates[0].finish_reason if chunk.candidates else None
                grounding_metadata = chunk.candidates[0].grounding_metadata if chunk.candidates else None
                yield text, str(reason) if reason else None, grounding_metadata
        else:
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue
                yield choice.delta.content or "", choice.finish_reason, None

    grounding_metadata = None
    attempt_count = 0

    while True:
        curr_content = finish_reason = None
        response_contents = []
        attempt_count += 1

        try:
            async with new_msg.channel.typing():
                async for delta_content, new_finish_reason, new_grounding_metadata in get_stream():
                    if new_grounding_metadata:
                        grounding_metadata = new_grounding_metadata

                    if finish_reason != None:
                        break

                    finish_reason = new_finish_reason

                    prev_content = curr_content or ""
                    curr_content = delta_content

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now().timestamp() - last_task_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and any(x in str(finish_reason).lower() for x in ("stop", "end_turn"))

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            view = SourceView(grounding_metadata) if is_final_edit and grounding_metadata and grounding_metadata.web_search_queries else None

                            msg_index = len(response_contents) - 1
                            if start_next_msg:
                                if msg_index < len(response_msgs):
                                    await response_msgs[msg_index].edit(embed=embed, view=view)
                                else:
                                    await reply_helper(embed=embed, silent=True, view=view)
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[msg_index].edit(embed=embed, view=view)

                            last_task_time = datetime.now().timestamp()

            if not response_contents:
                raise Exception("Response stream ended with no content")

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

            break

        except Exception:
            logging.exception("Error while generating response")
            if attempt_count >= 10:
                error_text = "I encountered too many errors and couldn't generate a response. Please try again later."

                if use_plain_responses:
                    await reply_helper(content=error_text)
                    response_contents = [error_text]
                else:
                    embed.description = error_text
                    embed.color = EMBED_COLOR_INCOMPLETE
                    if response_msgs:
                        await response_msgs[-1].edit(embed=embed, view=None)
                    else:
                        await reply_helper(embed=embed)
                    response_contents = [error_text]
                break

    if not use_plain_responses and len(response_msgs) > len(response_contents):
        for msg in response_msgs[len(response_contents):]:
            await msg.delete()
            if msg.id in msg_nodes:
                msg_nodes[msg.id].lock.release()
                del msg_nodes[msg.id]
        response_msgs = response_msgs[:len(response_contents)]

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


class SourceView(discord.ui.View):
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


async def health_check(request):
    return web.Response(text="I'm alive")


async def start_server():
    app = web.Application()
    app.add_routes([web.get('/', health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()


async def main() -> None:
    await asyncio.gather(start_server(), discord_bot.start(config["bot_token"]))


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
