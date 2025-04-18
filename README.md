<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/jakobdylanc/llmcord/assets/38699060/789d49fe-ef5c-470e-b60e-48ac03057443" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based chat system
Just @ the bot (or start your message with `at ai`) to start a conversation and reply to continue. Build conversations with reply chains!

*(Why "at ai"? It's an inside joke from a Discord server I'm in!)*

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot (or use `at ai`) while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot or use `at ai`. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot (or use `at ai`) inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

### Choose any LLM
llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [Google Gemini API](https://ai.google.dev/docs)
- [xAI API](https://docs.x.ai/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or run a local model with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

### And more:
- **API Key Rotation & Rate Limiting:** Automatically rotates through multiple API keys per provider (defined as a list in `config.yaml`). If a key encounters a rate limit error (during request or streaming), it's marked in a persistent SQLite database and avoided for 24 hours (even across bot restarts). The system retries with other available keys until a request succeeds. An error is only sent to Discord if all keys for that provider fail. The rate limit database for a specific provider automatically resets if all its keys become rate-limited, or every 24 hours.
- Supports image attachments when using a vision model (like gpt-4.1, claude-3, gemini-2.0-flash, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- **YouTube URL Processing:** Automatically extracts the title, description, channel name, transcription, and top comments from YouTube URLs in the query. This information is appended to the query before sending it to the LLM, providing context for summarization or discussion. (Requires a YouTube Data API v3 key for metadata/comments).
- **General URL Processing:** Automatically extracts text content from non-YouTube/Reddit URLs in the query using Beautiful Soup. This content is appended to the query before sending it to the LLM.
- **Reddit URL Processing:** Automatically extracts the title, author, score, body (or URL for link posts), and top comments from Reddit submission URLs in the query. This information is appended to the query before sending it to the LLM, providing context for summarization or discussion. (Requires Reddit API credentials).
- **Grounding with Google Search (Gemini only):** Enhances Gemini responses with up-to-date information from Google Search. Includes a "Show Sources" button on grounded responses to display search queries and web sources used.
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- ~2 Python files, ~1600 lines of code (main logic + key/rate limit management)

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile. **Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. **Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often. **Also disables streamed responses, warning messages, and the "Show Sources" button.**<br />(Default: `false`) |
| **allow_dms** | Set to `false` to disable direct message access.<br />(Default: `true`) |
| **permissions** | Configure permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`. **Leave `allowed_ids` empty to allow ALL. Role and channel permissions do not affect DMs. You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control grouped channel permissions.** |
| **youtube_data_api_key** | **(Optional)** Your YouTube Data API v3 key from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials). Required to fetch YouTube video titles, descriptions, channel names, and comments. Transcripts are fetched using `youtube-transcript-api` and do not require this key. |
| **reddit_client_id** | **(Optional)** Your Reddit application's client ID. Create a "script" application at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). Required for processing Reddit URLs. |
| **reddit_client_secret** | **(Optional)** Your Reddit application's client secret. Required for processing Reddit URLs. |
| **reddit_user_agent** | **(Optional)** A unique user agent string for your Reddit application (e.g., `discord:llmcord:v1.0 (by u/your_reddit_username)`). Required for processing Reddit URLs. See [Reddit's API rules](https://github.com/reddit-archive/reddit/wiki/API) for guidelines. |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with an optional `api_key` entry. For OpenAI-compatible APIs, also add a `base_url`. Popular providers (`openai`, `google`, `ollama`, etc.) are already included. **The `google` provider uses the `google-genai` library and does not require a `base_url`. Other providers require OpenAI-compatible APIs.** |
| **providers -> <provider_name> -> api_key** | Your API key(s) for the provider. **This can be a single key (string) or a list of keys (strings)** to enable key rotation. Leave empty or omit for local models that don't require keys. |
| **model** | Set to `<provider name>/<model name>`, e.g:<br /><br />-`openai/gpt-4.1`<br />-`google/gemini-2.0-flash`<br />-`ollama/llama3.3`<br />-`openrouter/anthropic/claude-3.7-sonnet` |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed. **Refer to your provider's documentation for supported API parameters.**<br />(Default: `max_tokens=4096, temperature=1.0`) |
| **system_prompt** | Write anything you want to customize the bot's behavior! **Leave blank for no system prompt.** |

3. Run the bot:

   **No Docker:**
   ```bash
   python -m pip install -U -r requirements.txt
   python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- **API Key Rotation & Rate Limiting:**
    - If you provide a list of API keys for a provider in `config.yaml`, the bot will randomly select a key for each request.
    - If a key encounters a rate limit error (HTTP 429 or equivalent, detected during request or streaming), it will be temporarily disabled for 24 hours. The bot will automatically retry the request with a different, non-limited key.
    - An error message is only sent to Discord if *all* configured keys for the requested provider have been tried and failed (either due to rate limits or other persistent errors).
    - Rate limit information is stored persistently in SQLite database files within the `ratelimit_db` directory (one `.db` file per provider). This ensures the 24-hour cooldown persists even if the bot restarts.
    - The rate limit database for a provider is automatically cleared if *all* keys for that provider become rate-limited simultaneously, allowing immediate retries. The database also clears expired entries automatically.
    - The `.gitignore` is configured to ignore the `ratelimit_db` directory and its contents.

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- The "Show Sources" button only appears on responses from Gemini models that were enhanced by Google Search grounding.

- YouTube URL processing requires a `youtube_data_api_key` in `config.yaml` to fetch metadata (title, description, channel) and comments. Transcripts are fetched via `youtube-transcript-api` and work without the key. If the key is missing or invalid, only the transcript (if available) will be appended.

- Reddit URL processing requires `reddit_client_id`, `reddit_client_secret`, and `reddit_user_agent` in `config.yaml`. If these are missing or invalid, Reddit URLs will not be processed. Be mindful of Reddit's API rate limits.
- General URL processing uses Beautiful Soup to extract text content. It attempts to get the main content but might include boilerplate or fail on complex sites. Content length is limited.

- Gemini safety settings are configured to `BLOCK_NONE` for all categories. Be mindful of the content generated.

- PRs are welcome :)

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>
