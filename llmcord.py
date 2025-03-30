import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional
import base64

import discord
import httpx
from openai import AsyncOpenAI
import yaml

from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "claude-3", "gemini", "gemma", "pixtral", "mistral-small", "llava", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

GEMINI_PROVIDERS = ("google")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100

def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)

cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if ((not is_dm and discord_client.user not in new_msg.mentions and "at ai" not in new_msg.content.lower()) or new_msg.author.bot):
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider, model = cfg["model"].split("/", 1)
    is_gemini = provider.lower() in GEMINI_PROVIDERS

    if is_gemini:

        api_key = cfg["providers"][provider].get("api_key")
        genai_client = genai.Client(api_key=api_key)

        enable_grounding = cfg["providers"][provider].get("enable_grounding", False)
    else:

        base_url = cfg["providers"][provider]["base_url"]
        api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:

                cleaned_content = curr_msg.content
                if discord_client.user.mention in cleaned_content:
                    cleaned_content = cleaned_content.removeprefix(discord_client.user.mention).lstrip()
                elif cleaned_content.lower().startswith("at ai"):
                    cleaned_content = cleaned_content[5:].lstrip()  

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(type) for type in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and "at ai" not in curr_msg.content.lower()
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
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

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        messages.append(dict(role="system", content=full_system_prompt))

    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    try:
        async with new_msg.channel.typing():
            if is_gemini:

                gemini_messages = []
                system_instruction = None

                for msg in messages[::-1]:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "system":
                        system_instruction = content
                        continue

                    gemini_role = "user" if role == "user" else "model"

                    if isinstance(content, list):

                        parts = []
                        for item in content:
                            if item["type"] == "text":
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item["type"] == "image_url":

                                image_url = item["image_url"]["url"]
                                if image_url.startswith("data:"):

                                    mime_type = image_url.split(';')[0].split(':')[1]
                                    base64_data = image_url.split(',')[1]

                                    image_bytes = base64.b64decode(base64_data)
                                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

                        if parts:
                            gemini_messages.append(types.Content(role=gemini_role, parts=parts))
                    else:

                        gemini_messages.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)]))

                extra_params = cfg["extra_api_parameters"].copy()
                temperature = extra_params.pop("temperature", 1.0)
                max_output_tokens = extra_params.pop("max_tokens", 4096)

                safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    )
                ]

                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    safety_settings=safety_settings,
                    **extra_params
                )

                if system_instruction:
                    generate_content_config.system_instruction = system_instruction

                tools = []
                if enable_grounding:
                    google_search_tool = types.Tool(google_search=types.GoogleSearch())
                    tools.append(google_search_tool)

                if tools:
                    generate_content_config.tools = tools

                async for chunk in await genai_client.aio.models.generate_content_stream(
                    model=model,
                    contents=gemini_messages,
                    config=generate_content_config,
                ):
                    new_content = chunk.text

                    if new_content:
                        prev_content = curr_content or ""
                        curr_content = new_content

                        if response_contents == [] and new_content == "":
                            continue

                        if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                            response_contents.append("")

                        response_contents[-1] += new_content

                        if not use_plain_responses:
                            ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                            msg_split_incoming = len(response_contents[-1] + new_content) > max_message_length

                            if start_next_msg or ready_to_edit or msg_split_incoming:
                                if edit_task != None:
                                    await edit_task

                                embed.description = response_contents[-1] + (STREAMING_INDICATOR if not msg_split_incoming else "")
                                embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming else EMBED_COLOR_INCOMPLETE

                                if start_next_msg:
                                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                                    response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                                    response_msgs.append(response_msg)

                                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                    await msg_nodes[response_msg.id].lock.acquire()
                                else:
                                    edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                                last_task_time = dt.now().timestamp()

                if not use_plain_responses and response_msgs:
                    embed.description = response_contents[-1]
                    embed.color = EMBED_COLOR_COMPLETE
                    await response_msgs[-1].edit(embed=embed)

                finish_reason = "stop"  

                if enable_grounding and response_msgs and not use_plain_responses:
                    try:

                        grounding_response = await genai_client.aio.models.generate_content(
                            model=model,
                            contents=gemini_messages,
                            config=generate_content_config,
                        )

                        if (hasattr(grounding_response.candidates[0], 'grounding_metadata') and 
                            grounding_response.candidates[0].grounding_metadata):

                            grounding_meta = grounding_response.candidates[0].grounding_metadata

                            grounding_embed = discord.Embed(
                                title="📚 Sources",
                                description="This response was enhanced with Google Search results.",
                                color=discord.Color.blue()
                            )

                            if hasattr(grounding_meta, 'web_search_queries') and grounding_meta.web_search_queries:
                                search_queries = []
                                for query in grounding_meta.web_search_queries:
                                    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
                                    search_queries.append(f"[{query}]({search_url})")

                                if search_queries:
                                    grounding_embed.add_field(
                                        name="Search Queries",
                                        value="\n".join(search_queries),
                                        inline=False
                                    )

                            if hasattr(grounding_meta, 'grounding_chunks') and grounding_meta.grounding_chunks:
                                sources = []
                                for i, chunk in enumerate(grounding_meta.grounding_chunks[:10]):  
                                    if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri') and hasattr(chunk.web, 'title'):
                                        sources.append(f"[{chunk.web.title}]({chunk.web.uri})")

                                if sources:
                                    grounding_embed.add_field(
                                        name="Top Sources",
                                        value="\n".join(sources),
                                        inline=False
                                    )

                            if grounding_embed.fields:
                                await response_msgs[-1].reply(embed=grounding_embed, silent=True)
                                logging.info("Sent grounding metadata as a separate message")
                    except Exception as e:
                        logging.exception(f"Error sending grounding metadata: {str(e)}")
            else:

                kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=cfg["extra_api_parameters"])
                async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                    if finish_reason != None:
                        break

                    finish_reason = curr_chunk.choices[0].finish_reason

                    prev_content = curr_content or ""
                    curr_content = curr_chunk.choices[0].delta.content or ""

                    new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason != None or msg_split_incoming
                        is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            if edit_task != None:
                                await edit_task

                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            if start_next_msg:
                                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                                response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                                response_msgs.append(response_msg)

                                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                                await msg_nodes[response_msg.id].lock.acquire()
                            else:
                                edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                            last_task_time = dt.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception as e:
        logging.exception(f"Error while generating response: {str(e)}")

        try:
            error_embed = discord.Embed(
                title="Error",
                description=f"An error occurred while generating a response. Please try again or check the logs.",
                color=discord.Color.red()
            )
            await new_msg.reply(embed=error_embed)
        except:
            logging.exception("Failed to send error message")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

async def main():
    await discord_client.start(cfg["bot_token"])

asyncio.run(main())