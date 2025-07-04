import re
import discord

# --- LLM & Vision ---
VISION_MODEL_TAGS = (
    "gpt-4",
    "o3",
    "o4",
    "claude-3",
    "gemini",
    "gemma",
    "llama",
    "pixtral",
    "mistral-small",
    "vision",
    "vl",
    "flash",
    "grok",
    "mistral",
)
PROVIDERS_SUPPORTING_USERNAMES = (
    "openai"  # x-ai might support it too, needs verification
)

# --- Discord Embeds & UI ---
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS_CONFIG_KEY = "edit_delay_seconds"

MAX_MESSAGE_NODES_CONFIG_KEY = "max_message_node_cache"
MAX_TEXT_LIMITS_CONFIG_KEY = "max_text_limits"  # New
DEFAULT_MAX_TEXT_KEY = "default"  # New
MODEL_SPECIFIC_MAX_TEXT_KEY = "models"  # New
MAX_TEXT_SAFETY_MARGIN_CONFIG_KEY = "max_text_safety_margin"  # New
MIN_TOKEN_LIMIT_AFTER_SAFETY_MARGIN_CONFIG_KEY = (
    "min_token_limit_after_safety_margin"  # New
)
# --- Limits ---
MAX_EMBED_FIELD_VALUE_LENGTH = 1024
MAX_EMBED_FIELDS = 25
MAX_EMBED_DESCRIPTION_LENGTH = 4096 - len(STREAMING_INDICATOR)  # Adjusted for indicator
MAX_EMBED_TOTAL_SIZE = 5900  # Safety margin below Discord's 6000 limit
MAX_PLAIN_TEXT_LENGTH = 2000  # Discord message character limit

# --- URL Regex Patterns ---
GENERAL_URL_PATTERN = re.compile(
    r'https?://[^\s<>"`]+|www\.[^\s<>"`]+'
)  # Added ` to negated set
YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11}))"
)
REDDIT_URL_PATTERN = re.compile(
    r"(https?://(?:www\.)?reddit\.com/r/[a-zA-Z0-9_]+/comments/([a-zA-Z0-9]+))"
)

# --- ADDED: Image URL Detection ---
COMMON_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff")
# Pattern to match URLs ending with common image extensions, ignoring query parameters or fragments
IMAGE_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"?#]+(?:"
    + "|".join(re.escape(ext) for ext in COMMON_IMAGE_EXTENSIONS)
    + r")",
    re.IGNORECASE,
)

AT_AI_PATTERN = re.compile(r"\bat ai\b", re.IGNORECASE)
GOOGLE_LENS_KEYWORD = "googlelens"
GOOGLE_LENS_PATTERN = re.compile(rf"^{GOOGLE_LENS_KEYWORD}\s+", re.IGNORECASE)
# --- ADDED CONSTANTS ---
IMGUR_HEADER = "--- Generated Images ---"
IMGUR_URL_PREFIX = "https://i.imgur.com/"
IMGUR_URL_PATTERN = re.compile(
    r"^(https://i\.imgur\.com/[a-zA-Z0-9]+\.(?:jpeg|jpg|png|gif))$"
)

# --- SearxNG and Grounding ---
SEARXNG_NUM_RESULTS_CONFIG_KEY = "searxng_num_results_fetch"
GROUNDING_MODEL_CONFIG_KEY = "grounding_model"
SEARXNG_BASE_URL_CONFIG_KEY = "searxng_base_url"
SEARXNG_DEFAULT_URL = "http://localhost:18088"  # Default if not in config
# --- ADDED CONSTANT ---
GROUNDING_SYSTEM_PROMPT_CONFIG_KEY = "grounding_system_prompt"
# --- ADDED CONSTANTS FOR SEARXNG URL CONTENT LENGTH ---
SEARXNG_URL_CONTENT_MAX_LENGTH_CONFIG_KEY = "searxng_url_content_max_length"
SEARXNG_DEFAULT_URL_CONTENT_MAX_LENGTH = 20000

# --- Grounding Model Parameters ---
GROUNDING_MODEL_TEMPERATURE_CONFIG_KEY = "grounding_model_temperature"
DEFAULT_GROUNDING_MODEL_TEMPERATURE = 0.7
GROUNDING_MODEL_TOP_K_CONFIG_KEY = "grounding_model_top_k"
DEFAULT_GROUNDING_MODEL_TOP_K = 40
GROUNDING_MODEL_TOP_P_CONFIG_KEY = "grounding_model_top_p"
DEFAULT_GROUNDING_MODEL_TOP_P = 0.95
# Grounding Model Thinking Budget (Imported from below, but belongs with other grounding params)
GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY = "grounding_model_use_thinking_budget"
GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY = (
    "grounding_model_thinking_budget_value"
)
GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET = False
GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE = 0
# Grounding optimization settings
GROUNDING_FORCE_STOP_OPTIMIZATION_CONFIG_KEY = "grounding_force_stop_optimization"
DEFAULT_GROUNDING_FORCE_STOP_OPTIMIZATION = False

# --- General URL Content Extraction ---
MAIN_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY = "main_general_url_content_extractor"
FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR_CONFIG_KEY = (
    "fallback_general_url_content_extractor"
)
DEFAULT_MAIN_GENERAL_URL_CONTENT_EXTRACTOR = "crawl4ai"
DEFAULT_FALLBACK_GENERAL_URL_CONTENT_EXTRACTOR = "beautifulsoup"
VALID_URL_EXTRACTORS = ["crawl4ai", "beautifulsoup", "jina"]

# --- Jina Reader Settings ---
JINA_ENGINE_MODE_CONFIG_KEY = "jina_engine_mode"
DEFAULT_JINA_ENGINE_MODE = "direct"
VALID_JINA_ENGINE_MODES = ["direct", "browser", "default"]
JINA_WAIT_FOR_SELECTOR_CONFIG_KEY = "jina_wait_for_selector"  # Added
DEFAULT_JINA_WAIT_FOR_SELECTOR = None  # Added - None means not used by default
JINA_TIMEOUT_CONFIG_KEY = "jina_timeout"  # Added
DEFAULT_JINA_TIMEOUT = (
    None  # Added - None means not used by default, Jina might have its own default
)

# --- Crawl4AI Settings ---
CRAWL4AI_CACHE_MODE_CONFIG_KEY = "crawl4ai_cache_mode"  # Added
DEFAULT_CRAWL4AI_CACHE_MODE = "bypass"  # Added
VALID_CRAWL4AI_CACHE_MODES = [
    "bypass",
    "enabled",
    "refresh",
    "only_refresh",
]  # Added

# --- External Web Content API ---
WEB_CONTENT_EXTRACTION_API_ENABLED_CONFIG_KEY = "web_content_extraction_api_enabled"
WEB_CONTENT_EXTRACTION_API_URL_CONFIG_KEY = "web_content_extraction_api_url"
WEB_CONTENT_EXTRACTION_API_MAX_RESULTS_CONFIG_KEY = (
    "web_content_extraction_api_max_results"
)
DEFAULT_WEB_CONTENT_EXTRACTION_API_ENABLED = False
DEFAULT_WEB_CONTENT_EXTRACTION_API_URL = "http://localhost:8080/search"
DEFAULT_WEB_CONTENT_EXTRACTION_API_MAX_RESULTS = 3
# Cache configuration for web content extraction API
WEB_CONTENT_EXTRACTION_API_CACHE_TTL_CONFIG_KEY = (
    "web_content_extraction_api_cache_ttl_minutes"
)
DEFAULT_WEB_CONTENT_EXTRACTION_API_CACHE_TTL = 15
# HTTP client optimization
HTTP_CLIENT_USE_HTTP2_CONFIG_KEY = "http_client_use_http2"
DEFAULT_HTTP_CLIENT_USE_HTTP2 = True

# --- Gemini Thinking Budget ---
GEMINI_USE_THINKING_BUDGET_CONFIG_KEY = "gemini_use_thinking_budget"
GEMINI_THINKING_BUDGET_VALUE_CONFIG_KEY = "gemini_thinking_budget_value"
GEMINI_DEFAULT_USE_THINKING_BUDGET = False
GEMINI_DEFAULT_THINKING_BUDGET_VALUE = 0
GEMINI_MIN_THINKING_BUDGET_VALUE = 0
GEMINI_MAX_THINKING_BUDGET_VALUE = 24576

# --- Grounding Model Thinking Budget Specific Constants ---
GROUNDING_MODEL_USE_THINKING_BUDGET_CONFIG_KEY = "grounding_model_use_thinking_budget"
GROUNDING_MODEL_THINKING_BUDGET_VALUE_CONFIG_KEY = (
    "grounding_model_thinking_budget_value"
)
GROUNDING_MODEL_DEFAULT_USE_THINKING_BUDGET = False
GROUNDING_MODEL_DEFAULT_THINKING_BUDGET_VALUE = (
    0  # Default to 0 (thinking disabled) for grounding
)

USER_GEMINI_THINKING_BUDGET_PREFS_FILENAME = (
    "config/user_gemini_thinking_budget_prefs.json"
)
USER_SYSTEM_PROMPTS_FILENAME = "config/user_system_prompts.json"
USER_MODEL_PREFS_FILENAME = "config/user_model_prefs.json"

# --- Output Sharing Settings ---
OUTPUT_SHARING_CONFIG_KEY = "output_sharing"
TEXTIS_ENABLED_CONFIG_KEY = "textis_enabled"  # Repurposed from NGROK_ENABLED_CONFIG_KEY

RATE_LIMIT_COOLDOWN_HOURS_CONFIG_KEY = "rate_limit_cooldown_hours"
URL_SHORTENER_ENABLED_CONFIG_KEY = "url_shortener_enabled"
URL_SHORTENER_SERVICE_CONFIG_KEY = "url_shortener_service"

# --- Rate Limiting ---
GLOBAL_RESET_FILE = "logs/last_reset_timestamp.txt"
DB_FOLDER = "data/ratelimit_dbs"

FALLBACK_VISION_MODEL_CONFIG_KEY = "fallback_vision_model"
# --- Model Selection ---
FALLBACK_MODEL_INCOMPLETE_STREAM_CONFIG_KEY = "fallback_model_incomplete_stream"
# This defines which models are *known* to the bot for autocomplete/validation
# The actual availability depends on the user's config.yaml

# --- ADDED: Fallback vision model if selected model cannot handle images ---

# --- ADDED: Fallback model for incomplete non-Gemini streams ---
# --- System prompt for the fallback model ---
FALLBACK_MODEL_SYSTEM_PROMPT_CONFIG_KEY = "fallback_model_system_prompt"
DEFAULT_FALLBACK_MODEL_SYSTEM_PROMPT = """
You are a concise assistant. The previous attempt to generate a response by another model was incomplete or failed.
Please provide a very brief, helpful, and directly relevant answer to the user's query based on the provided context.
If the query seems to imply a very long or complex answer that cannot be brief, it's okay to state that a concise answer isn't possible for that specific query.
Do not apologize for the previous model. Focus on fulfilling the user's request succinctly.
""".strip()

AVAILABLE_MODELS = {
    "google": [
        "gemini-2.5-flash",
        "gemini-2.0-flash-preview-image-generation",
        "imagen-4.0-ultra-generate-preview-06-06",
        # Add other Gemini models as needed/supported
    ],
    "openai": [
        "gpt-4.1",
        "o4-mini",
        "gpt-4o",
        # Add other OpenAI models as needed/supported
    ],
    "x-ai": [
        "grok-3"
        # Add other xAI models as needed/supported
    ],
    "mistral": [
        "mistral-medium-latest"
        # Add other Mistral models as needed/supported
    ],
    "anthropic": ["claude-3.7-sonnet", "claude-3.7-sonnet-thought", "claude-sonnet-4"],
    "deepseek": [
        "deepseek-ai/DeepSeek-V3-0324"
        # Add other DeepSeek models as needed/supported
    ],
}
# Keywords for Model Override
DEEP_SEARCH_KEYWORDS = ["deepsearch", "deepersearch"]
DEEP_SEARCH_MODEL_CONFIG_KEY = "deep_search_model"

# --- Gemini Safety Settings ---
GEMINI_SAFETY_SETTINGS_CONFIG_KEY = "gemini_safety_settings"
# Use google.genai.types (imported as google_types)

# --- Prompt Enhancer ---
PROMPT_ENHANCER_SYSTEM_PROMPT_CONFIG_KEY = "prompt_enhancer_system_prompt"

# --- History Persistence Configuration Keys ---
STAY_IN_CHAT_HISTORY_CONFIG_KEY = "stay_in_chat_history"
STAY_IN_HISTORY_USER_URLS_KEY = "user_provided_urls"
STAY_IN_HISTORY_SEARCH_RESULTS_KEY = "search_results"
STAY_IN_HISTORY_GOOGLE_LENS_KEY = "google_lens"

# --- Table Rendering Configuration ---
AUTO_RENDER_MARKDOWN_TABLES_CONFIG_KEY = "auto_render_markdown_tables"
DEFAULT_AUTO_RENDER_MARKDOWN_TABLES = True


# --- Custom Exceptions ---
class AllKeysFailedError(Exception):
    """Custom exception raised when all API keys for a service have failed."""

    def __init__(self, service_name, errors):
        self.service_name = service_name
        self.errors = errors  # List of errors encountered for each key
        super().__init__(
            f"All API keys failed for service '{service_name}'. Last error: {errors[-1] if errors else 'None'}"
        )
