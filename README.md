# llmcord

Talk to LLMs with your friends on Discord.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="llmcord screenshot">
</p>

`llmcord` turns Discord into a collaborative LLM frontend with reply-chain
conversations, per-user model preferences, web search, file support, and a
small set of Discord-native controls for retrying and inspecting responses.

It supports standard LiteLLM/OpenAI-compatible backends plus the
OAuth-backed providers in this repo, including Google Gemini CLI,
Google Antigravity, and OpenAI Codex.

## Features

### Conversation flow

- Mention the bot or say `at ai` to start a conversation in servers.
- Reply to keep the conversation going from any earlier message.
- Continue another user's chain or branch a conversation indefinitely.
- In DMs, conversations continue automatically without requiring replies.
- In threads, the bot can continue from the thread starter.
- Trigger-only messages such as just a mention or `at ai` still continue the
  conversation.

### Model and provider support

- Switch models per user with `/model`.
- Configure models as `provider/model`.
- Use standard LiteLLM/OpenAI-compatible providers such as OpenAI, xAI, Groq,
  Ollama, LM Studio, vLLM, OpenRouter, or any compatible gateway you expose via
  `base_url`.
- Use the built-in OAuth-backed providers in this repo:
  `google-gemini-cli`, `google-antigravity`, and `openai-codex`.
- Add model-level overrides such as `model`, `disable_system_prompt`, or other
  provider request fields directly under `models:`.

### Search and research

- Automatic web search with Tavily or Exa MCP.
- Search-decider model selection with `/searchdecidermodel`.
- Native Gemini grounding when the provider/model path supports it.
- Deep research commands with `researchpro` and `researchmini`.
- Source inspection with a paginated `Show Sources` button.

### Attachments and extracted content

- Image attachments for vision-capable models.
- Text attachments and PDF support.
- Reverse image search with `googlelens` plus an attached image.
- YouTube transcript extraction.
- Twitter/X and Reddit link expansion.
- TikTok and Facebook video downloading for Gemini requests.
- Audio/video preprocessing for non-Gemini target models using a configured
  Gemini model.

### Response UX

- `View Response Better` uploads long responses to `rentry.co`.
- `Show Thought Process` exposes provider-supplied thinking output when
  available.
- `Retry` reruns a response for the original requesting user.
- `failed urls` shows any URLs that could not be extracted.

### Runtime behavior

- Hot-reloads `config.yaml` changes after file updates.
- Stores user model preferences and cached search/media metadata in SQLite.
- Starts a simple HTTP health-check server.
- Uses asynchronous message processing and cached message nodes to reduce
  Discord API churn.

## Architecture Overview

- `src/llmcord/__main__.py`: process entrypoint and shutdown handling.
- `src/llmcord/entrypoint.py`: bootstraps the database, health-check server,
  slash commands, and Discord events.
- `src/llmcord/core/config/`: config loading, profile normalization, shared
  constants, and HTTP client setup.
- `src/llmcord/discord/`: slash commands, event handlers, and Discord UI
  components.
- `src/llmcord/logic/`: conversation building, attachment handling, search
  orchestration, compaction, and generation pipeline logic.
- `src/llmcord/services/`: provider integrations, search backends, extractors,
  database access, and external service helpers.

## Slash Commands

- `/model`: view or switch your current model.
- `/searchdecidermodel`: view or switch the model used to decide whether web
  search should run.
- `/humanize`: humanize text through QuillBot.
- `/resetallpreferences`: owner-only reset for saved user model preferences and
  search-decider preferences.

## Configuration Notes

### Config file location

The app loads `config.yaml` from the current working directory, or from
`/etc/secrets/config.yaml` if present there instead.

### Profiles and health checks

- `config.yaml` supports `main` and `test` profiles.
- Each profile defines `bot_token`, `port`, and optional `host`.
- `profile: main` chooses the active profile.
- The health-check server listens on the selected profile's `host` and `port`.
- `HOST` and `PORT` environment variables override the configured health-check
  address.

### Permissions

Use `permissions.users`, `permissions.roles`, and `permissions.channels` to
allow or block access. DMs can also be disabled with `allow_dms: false`.

### Providers and models

- Models are declared as `provider/model`.
- For OpenAI-compatible backends, set `providers.<name>.base_url` and list
  models under `models:`.
- Model entries can override the actual sent model name through `model`,
  `model_name`, or `modelName`.
- Append `:vision` to a configured model key if you need to force image support
  for UI/model selection behavior.
- Reasoning aliases are supported, for example
  `openai-codex/gpt-5.2-xhigh`.

### Search

- `web_search_provider` can be `tavily`, `exa`, or `auto`.
- `auto` prefers Tavily when `tavily_api_key` is configured and otherwise falls
  back to Exa MCP.
- `web_search_decider_model` controls the model that decides whether search is
  necessary.
- `web_search_max_chars_per_url` caps how much text is retained from each
  search result.

### Media preprocessing and prompt controls

- `media_preprocessor_model` can pin the Gemini model used to analyze
  audio/video for non-Gemini target models.
- `model_token_limits` configures prompt compaction context windows.
- `channel_model_overrides` locks channels to specific models and disables
  `/model` there.
- `disable_system_prompt_models` disables the global system prompt for specific
  configured models.
- Individual model entries can also set `disable_system_prompt: true`.

## Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord
   ```

2. Install dependencies:

   ```bash
   uv sync --extra dev
   ```

3. Copy the example config and fill in your Discord token, providers, and
   models:

   ```bash
   cp config-example.yaml config.yaml
   # PowerShell: Copy-Item config-example.yaml config.yaml
   ```

4. Optional OAuth login helpers:

   For Google Gemini CLI:

   ```bash
   uv run python -c "from llmcord.services.llm.providers.gemini_cli import cli_login_main; raise SystemExit(cli_login_main())"
   ```

   Paste the resulting JSON into `providers.google-gemini-cli.api_key`.

   For Google Antigravity:

   ```bash
   uv run python -c "from llmcord.services.llm.providers.gemini_cli import cli_login_antigravity_main; raise SystemExit(cli_login_antigravity_main())"
   ```

   Paste the resulting JSON into `providers.google-antigravity.api_key`.

   For OpenAI Codex:

   ```bash
   uv run python -c "from llmcord.services.llm.providers.openai_codex import cli_login_openai_codex_main; raise SystemExit(cli_login_openai_codex_main())"
   ```

   Paste the resulting JSON into `providers.openai-codex.api_key`.

5. Run the bot:

   ```bash
   uv run python -m llmcord
   ```

## Development

```bash
uvx ruff check --select ALL . --fix
uvx ruff format
uvx ty check
uv run pytest
```

## Notes

- For improved PDF layout analysis, install
  [pymupdf-layout](https://pypi.org/project/pymupdf-layout/).
- User identity forwarding via the `name` field is currently enabled for
  `openai`, `openai-codex`, and `x-ai`.
- PRs are welcome.

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
<img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
</picture>
</a>
