"""Provider and model configuration logic."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import discord

from llmcord.core.config import ensure_list, get_config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderSettings:
    """Resolved provider/model settings."""

    provider: str
    model: str
    provider_slash_model: str
    base_url: str | None
    api_keys: list[str]
    model_parameters: dict[str, object] | None
    extra_headers: dict[str, str] | None
    extra_query: dict[str, object] | None
    extra_body: dict[str, object] | None
    actual_model: str


async def resolve_provider_settings(
    *,
    processing_msg: discord.Message,
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
    override_provider_slash_model: str | None,
    send_error_func: Callable[[discord.Message, str], Awaitable[None]] | None = None,
) -> ProviderSettings | None:
    """Resolve provider and model settings."""
    provider_slash_model = await get_provider_slash_model(
        override_provider_slash_model=override_provider_slash_model,
        curr_model_lock=curr_model_lock,
        curr_model_ref=curr_model_ref,
    )
    config = get_config()

    parsed_provider = parse_provider_slash_model(provider_slash_model)
    if parsed_provider is None:
        logger.exception(
            "Invalid model format: %s. Expected 'provider/model'.",
            provider_slash_model,
        )
        if send_error_func:
            await send_error_func(
                processing_msg,
                (
                    "❌ Invalid model configuration: "
                    f"'{provider_slash_model}'. Expected format: "
                    "'provider/model'.\n"
                    "Please contact an administrator."
                ),
            )
        return None

    provider, model = parsed_provider
    providers = config.get("providers", {})
    provider_config = providers.get(provider)
    if provider_config is None:
        logger.error("Provider '%s' not found in config.", provider)
        if send_error_func:
            await send_error_func(
                processing_msg,
                (
                    f"❌ Provider '{provider}' is not configured. "
                    "Please contact an administrator."
                ),
            )
        return None

    base_url = provider_config.get("base_url")
    api_keys = ensure_list(provider_config.get("api_key"))
    if not api_keys:
        api_keys = ["sk-no-key-required"]
    model_parameters = config["models"].get(provider_slash_model, None)
    override_model = None
    if model_parameters:
        override_model = (
            model_parameters.get("model")
            or model_parameters.get("model_name")
            or model_parameters.get("modelName")
        )

    if isinstance(override_model, str) and "/" in override_model:
        _, override_model = override_model.split("/", 1)

    actual_model = override_model or model
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {})
    extra_body = extra_body or None

    return ProviderSettings(
        provider=provider,
        model=model,
        provider_slash_model=provider_slash_model,
        base_url=base_url,
        api_keys=api_keys,
        model_parameters=model_parameters,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
        actual_model=actual_model,
    )


async def get_provider_slash_model(
    *,
    override_provider_slash_model: str | None,
    curr_model_lock: asyncio.Lock,
    curr_model_ref: list[str],
) -> str:
    """Get the provider/model string, handling overrides and locks."""
    if override_provider_slash_model:
        return override_provider_slash_model

    async with curr_model_lock:
        return curr_model_ref[0]


def parse_provider_slash_model(
    provider_slash_model: str,
) -> tuple[str, str] | None:
    """Parse 'provider/model' string."""
    parts = provider_slash_model.removesuffix(":vision").split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None
