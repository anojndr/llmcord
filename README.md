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
Just **@ the bot** or start your message with **"at ai"** (case-insensitive) to start a conversation, and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot or use "at ai" while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot or use "at ai". You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot or use "at ai" inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

*(Note: The "at ai" trigger is an inside joke from a Discord server the author is in. It functions identically to mentioning the bot.)*

### Choose any LLM
llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [Google Gemini API](https://ai.google.dev/models/gemini) (via `google-genai` library)
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
- **Robust API Key Rotation & Rate Limiting:** Automatically rotates through multiple API keys for LLM providers (OpenAI, Google Gemini, etc.) and SerpAPI. If a key encounters an error (especially rate limits, including during response streaming), it retries with the next available key. Errors are only reported to Discord if all keys for a service fail for a single request.
- **Persistent Rate Limit Cooldown:** Uses SQLite databases (in the `ratelimit_db` directory) to track rate-limited keys, preventing their reuse for 24 hours, even across bot restarts. Databases automatically reset if all keys for a service become limited or after the 24-hour cooldown expires for individual keys.
- **YouTube URL Processing:** Automatically extracts title, description, channel name, transcript, and top comments from YouTube URLs in the user's query, appending this information as context for the LLM. Handles multiple URLs simultaneously.
- **Reddit URL Processing:** Automatically extracts submission title, self-text, and top comments from Reddit URLs (including share links like `/r/subreddit/s/...`) in the user's query, appending this information as context for the LLM. Handles multiple URLs simultaneously.
- Supports image attachments when using a vision model (like gpt-4.1, claude-3, gemini-1.5-flash, etc.)
- **Generic URL Processing:** Fetches and extracts text content from other web URLs (excluding YouTube and Reddit) found in the query, appending this as context. Handles multiple URLs concurrently.
- **Google Lens Image Search:** When a query starts with "lens" (after the bot mention or "at ai"), the bot uses SerpAPI's Google Lens API to analyze attached images and appends the results to the query before sending it to the LLM. Handles multiple images and rotates through `serpapi_api_keys`.
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- **Google Search grounding** for enhanced factuality and recency with Gemini models.
- Conditional **"Show Sources" button** displays grounding metadata (search queries, sources) for Gemini responses, appearing only if grounding was used.
- Configurable safety settings (defaults to `block_none` for Gemini in the code).
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded, or YouTube extraction errors)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~1300 lines of code (including comments and blank lines)

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

   **Important:** API keys for LLM providers and SerpAPI are now configured as *lists* to support rotation. See the updated sections below. A `ratelimit_db` directory will be created automatically in the project root to store rate limit information.

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile. **Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. **Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often. **Also disables streamed responses and warning messages.**<br />(Default: `false`) |
| **allow_dms** | Set to `false` to disable direct message access.<br />(Default: `true`) |
| **permissions** | Configure permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`. **Leave `allowed_ids` empty to allow ALL. Role and channel permissions do not affect DMs. You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control grouped channel permissions.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use. For OpenAI-compatible APIs, include a `base_url` and a list of `api_keys`. For Google Gemini, add a `google` entry with just a list of `api_keys`. Popular providers (`openai`, `ollama`, etc.) are already included. **Provide API keys as a list under `api_keys:` for each provider you want to use keys with.** For providers like Ollama that don't require keys, use an empty list `api_keys: []`. The bot will rotate through the provided keys for each request. |
| **model** | Set to `<provider name>/<model name>`, e.g:<br /><br />-`openai/gpt-4.1`<br />-`google/gemini-2.0-flash`<br />-`ollama/llama3.1`<br />-`openrouter/anthropic/claude-3.5-sonnet` |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed. **Refer to your provider's documentation for supported API parameters.**<br />(Default: `max_tokens=4096, temperature=1.0`) |
| **system_prompt** | Write anything you want to customize the bot's behavior! **Leave blank for no system prompt.** |

### YouTube Data API settings:

| Setting | Description |
| --- | --- |
| **youtube_api_key** | **Required for YouTube URL processing.** Create a project on [Google Cloud Console](https://console.cloud.google.com/), enable the "YouTube Data API v3", and generate an API key under "Credentials". This key is used to fetch video details (title, description, channel) and comments. Transcripts are fetched using `youtube-transcript-api` and do not require this key. **Note: Key rotation is NOT implemented for the YouTube Data API.** |

### Reddit API settings:

| Setting | Description |
| --- | --- |
| **reddit** | **Required for Reddit URL processing.** Contains `client_id`, `client_secret`, and `user_agent`. Create a script app on [Reddit's app preferences page](https://www.reddit.com/prefs/apps) to get these credentials. The `user_agent` should be a unique string, e.g., `discord:your-app-name:v1.0 (by /u/your-reddit-username)`. **Note: Key rotation is NOT implemented for the Reddit API.** |

### SerpApi settings:

| Setting | Description |
| --- | --- |
| **serpapi_api_keys** | **Required for Google Lens image search.** Provide a *list* of API keys from [SerpApi](https://serpapi.com/) under `serpapi_api_keys:`. The bot will rotate through these keys when making Google Lens requests. |

#### Gemini Specifics:
- **Grounding:** Google Search grounding is automatically enabled in the code when using a `google/` model to improve factual accuracy and access recent information.
- **Show Sources Button:** If a Gemini response was enhanced by grounding, a "Show Sources" button will appear below the message. Clicking it reveals the search queries used and the web sources cited by the model for that response.
- **Safety Settings:** All Gemini safety settings are currently hardcoded to `BLOCK_NONE` within `llmcord.py`.
- **Rate Limiting:** Rate limiting and key rotation apply to Gemini API keys just like other providers.

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

- **API Key Rotation:** The bot now rotates through the API keys provided in the `api_keys` list for each LLM provider and the `serpapi_api_keys` list. If a request fails (e.g., due to rate limits, server errors, including during response streaming), it automatically retries with the next available key. Error messages are only sent to Discord if *all* configured keys for a service have been tried and failed for a single request.

- **Rate Limit Database:** A `ratelimit_db` directory will be created in the project root to store SQLite databases (e.g., `openai_ratelimit.db`, `google_ratelimit.db`, `serpapi_ratelimit.db`). These track rate-limited keys.

- **Rate Limit Persistence & Reset:** Rate-limited keys are prevented from reuse for 24 hours. This cooldown persists even if the bot restarts. If all keys for a specific service (e.g., all configured Google keys) become rate-limited, the database for that service will be automatically cleared, allowing retries sooner. Individual keys are also cleared after their 24-hour cooldown expires.

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Google Gemini API does not currently support this specific parameter. Hopefully more providers support this in the future.

- The "Show Sources" button only appears for Gemini responses where grounding was actively used by the model to generate the answer. If the model answered from its internal knowledge without searching, the button will not be shown.

- YouTube URL processing requires a `youtube_api_key` in `config.yaml` to fetch video details and comments. Transcripts are fetched via `youtube-transcript-api` and don't need the key. The bot will still function without the key, but YouTube URLs won't be processed. Be mindful of YouTube Data API quotas. Errors during extraction (e.g., disabled transcripts/comments, private videos, quota exceeded) will be noted in the appended context. Key rotation is *not* implemented for the YouTube Data API.

- PRs are welcome :)

- Reddit URL processing requires Reddit API credentials (`client_id`, `client_secret`, `user_agent`) in `config.yaml`. The bot will still function without these, but Reddit URLs won't be processed. Be mindful of Reddit API rate limits. Errors during extraction (e.g., private subreddits, deleted posts, API issues) will be noted in the appended context. Key rotation is *not* implemented for the Reddit API.

- Generic URL processing uses `httpx` and `BeautifulSoup` to fetch and parse web pages. It attempts to extract the main textual content. Errors during fetching or parsing (e.g., network issues, non-HTML content, complex JavaScript sites) will be noted in the appended context. This feature does not execute JavaScript.

- Google Lens integration requires a list of `serpapi_api_keys` in `config.yaml`. The bot will function without it, but the "lens" command won't work. The bot rotates through these keys and handles rate limits. Be mindful of SerpApi usage limits. Errors during the API call will be noted in the appended context.

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>