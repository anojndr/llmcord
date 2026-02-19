<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based chat system:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

---

### Model switching with `/model`:
![image](https://github.com/user-attachments/assets/568e2f5c-bf32-4b77-ab57-198d9120f3d2)

llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [xAI API](https://docs.x.ai/docs/models)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/models)
- **Google Cloud Code Assist (Gemini CLI OAuth)**
- **Google Antigravity (Gemini 3 / Claude / GPT-OSS via OAuth)**
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter](https://openrouter.ai/models)

Or run local models with:
- [Ollama](https://ollama.com)
- [LM Studio](LM Studio)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

---

### And more:
- **Web Search**: Automatic web search with Tavily or Exa MCP (for non-Gemini models) or native Gemini grounding. Uses an LLM to intelligently decide when search is needed.
- **Research Commands**: Use `researchpro` or `researchmini` for deep multi-step research via Tavily.
- **"View Response Better" button**: Upload long responses to rentry.co for easier reading.
- **"Show Sources" button**: View search queries and source URLs with pagination for web searches.
- **"Show Thought Process" button**: Reveal the hidden chain-of-thought for reasoning models (e.g., Gemini 3, o1, o3).
- Supports image attachments when using a vision model.
- Supports Reverse Image Search (Google Lens via SerpApi + Yandex) (start your message with `googlelens` and attach an image).
- Supports YouTube video transcripts (just paste a YouTube link).
- Supports Twitter/X and Reddit link expansion.
- **Social Video Support**: Automatically downloads and processes TikTok and Facebook video links (Gemini only).
- Supports text file attachments (.txt, .py, .c, etc.) and PDF attachments (native for Gemini, text extraction for others).
- Customizable personality (aka system prompt) with `{date}` and `{time}` support.
- Hot reloading config (change settings without restarting the bot).
- Fully asynchronous and mutex-protected message caching to minimize Discord API calls.

## Architecture overview

- **Entry point:** [src/llmcord/__main__.py](src/llmcord/__main__.py) starts the bot via [src/llmcord/entrypoint.py](src/llmcord/entrypoint.py).
- **Discord layer:** [src/llmcord/discord/](src/llmcord/discord/) handles slash commands, events, and UI interactions.
- **Message pipeline:** [src/llmcord/logic/pipeline.py](src/llmcord/logic/pipeline.py) orchestrates context building, web search, and response generation.
- **Provider glue:** [src/llmcord/services/llm/](src/llmcord/services/llm/) centralizes LiteLLM integration and provider-specific configurations.
- **Persistence:** [src/llmcord/services/database/](src/llmcord/services/database/) manages user preferences and bad API key tracking with local SQLite.
- **External Services:** [src/llmcord/services/](src/llmcord/services/) contains logic for scraping TikTok, Facebook, Twitter, and performing web searches.

## Commands

- `/model` — View or switch your active model.
- `/searchdecidermodel` — View or switch the model used to decide when web search is needed.
- `/resetallpreferences` — [Owner only] Reset all saved user model preferences to defaults.

## Configuration notes

### Profiles
`config.yaml` supports `main` and `test` profiles. Set `profile: main` to select which one to run. Profile-specific settings include `port` and `bot_token`.

### Time To First Token
- Configure `first_token_timeout_seconds` in `config.yaml` to control how long llmcord waits for the first streamed token before rotating keys or falling back.
- Default is `30` seconds when omitted or invalid.

### Web Search
- **Tavily** (requires API key) and **Exa MCP** are supported.
- Set `web_search_provider: auto` to prefer Tavily when keys exist, falling back to Exa.
- The `web_search_decider_model` dictates which LLM determines search necessity.

### Provider naming
- Models are configured as `provider/model` (e.g., `gemini/gemini-2.5-pro`).
- Add `:vision` to a model key to force vision support if not auto-detected.

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord
   ```

2. Install dependencies (using [uv](https://github.com/astral-sh/uv) is recommended):
  ```bash
  uv venv
  # Windows:
  .venv\Scripts\activate
  # Linux/macOS:
  source .venv/bin/activate

  uv pip install -r pyproject.toml
  uv pip install -e .
  ```

3. Create a copy of `config-example.yaml` named `config.yaml` and set your `bot_token`, `providers`, and `models`.

#### For Google Gemini CLI support:
Run the login helper to generate your API key JSON:
```bash
uv run python -c "from llmcord.services.llm.providers.gemini_cli import cli_login_main; raise SystemExit(cli_login_main())"
```
Paste the resulting JSON into `providers.google-gemini-cli.api_key`.

#### For Google Antigravity support:
Run the Antigravity login helper to generate your API key JSON:
```bash
uv run python -c "from llmcord.services.llm.providers.gemini_cli import cli_login_antigravity_main; raise SystemExit(cli_login_antigravity_main())"
```
Paste the resulting JSON into `providers.google-antigravity.api_key`.

4. Run the bot:
  ```bash
  python -m llmcord
  ```

## Development

- Run tests: `pytest`
- Linting: `ruff check --select ALL . --fix`
- Type checking: `ty check`

## Notes

- For improved PDF layout analysis, install [pymupdf-layout](https://pypi.org/project/pymupdf-layout/).
- User identity awareness (the `name` parameter) is currently supported for OpenAI and xAI providers.
- PRs are welcome!

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>