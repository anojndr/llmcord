from __future__ import annotations

from llmcord.core.config import VISION_MODEL_TAGS
from llmcord.services.llm import build_litellm_model_name


def test_openrouter_model_name_prefix() -> None:
    assert (
        build_litellm_model_name("openrouter", "openrouter/free")
        == "openrouter/openrouter/free"
    )


def test_openrouter_free_is_vision_capable() -> None:
    assert "openrouter/free" in VISION_MODEL_TAGS
