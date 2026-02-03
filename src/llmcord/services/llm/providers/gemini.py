"""Gemini provider implementation."""

from typing import Any


def _get_safety_settings(model_parameters: dict | None) -> object | None:
    if not model_parameters:
        return None
    safety_settings = model_parameters.get("safety_settings")
    if safety_settings is None:
        safety_settings = model_parameters.get("safetySettings")
    return safety_settings


def _resolve_thinking_level(
    model: str,
    model_parameters: dict | None,
) -> str | None:
    thinking_level = (
        model_parameters.get("thinking_level") if model_parameters else None
    )
    if thinking_level:
        return thinking_level
    if "gemini-3-flash" in model:
        return "MINIMAL"
    if "gemini-3-pro" in model:
        return "LOW"
    return None


def _apply_thinking_config(
    kwargs: dict[str, Any],
    model: str,
    model_parameters: dict | None,
) -> None:
    thinking_level = _resolve_thinking_level(model, model_parameters)
    if not thinking_level:
        return
    thinking_map = {
        "MINIMAL": "minimal",
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
    }
    kwargs["reasoning_effort"] = thinking_map.get(
        thinking_level,
        thinking_level.lower(),
    )


def _resolve_code_execution_config(
    model_parameters: dict | None,
) -> dict[str, object] | None:
    if not model_parameters:
        return {}
    for key in ("enable_code_execution", "code_execution", "codeExecution"):
        if key not in model_parameters:
            continue
        value = model_parameters.get(key)
        if value is False:
            return None
        if value is True:
            return {}
        if isinstance(value, dict):
            return value
        return None
    return {}


def _merge_tools(
    existing_tools: list[dict[str, object]],
    new_tools: object | None,
) -> list[dict[str, object]]:
    if isinstance(new_tools, list):
        return [*existing_tools, *new_tools]
    if isinstance(new_tools, dict):
        return [*existing_tools, new_tools]
    return existing_tools


def _build_gemini_tools(
    *,
    kwargs: dict[str, Any],
    model_parameters: dict | None,
    enable_grounding: bool,
    is_preview: bool,
) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    if isinstance(kwargs.get("tools"), list):
        tools.extend(kwargs["tools"])

    tools = _merge_tools(
        tools,
        model_parameters.get("tools") if model_parameters else None,
    )

    if enable_grounding and not is_preview:
        if not any("googleSearch" in tool for tool in tools):
            tools.append({"googleSearch": {}})
        if not any("urlContext" in tool for tool in tools):
            tools.append({"urlContext": {}})

    code_execution_config = _resolve_code_execution_config(model_parameters)
    if code_execution_config is not None and not any(
        "codeExecution" in tool for tool in tools
    ):
        tools.append({"codeExecution": code_execution_config})

    return tools


def _apply_temperature_config(
    kwargs: dict[str, Any],
    model_parameters: dict | None,
    *,
    is_gemini_3: bool,
) -> None:
    if (
        model_parameters
        and not is_gemini_3
        and "temperature" in model_parameters
        and "temperature" not in kwargs
    ):
        kwargs["temperature"] = model_parameters["temperature"]


def configure_gemini_kwargs(
    kwargs: dict[str, Any],
    model: str,
    model_parameters: dict | None = None,
    *,
    enable_grounding: bool = False,
) -> None:
    """Configure Gemini-specific kwargs."""
    is_gemini_3 = "gemini-3" in model
    is_preview = "preview" in model

    if safety_settings := _get_safety_settings(model_parameters):
        kwargs["safety_settings"] = safety_settings

    _apply_thinking_config(kwargs, model, model_parameters)

    tools = _build_gemini_tools(
        kwargs=kwargs,
        model_parameters=model_parameters,
        enable_grounding=enable_grounding,
        is_preview=is_preview,
    )
    if tools:
        kwargs["tools"] = tools

    _apply_temperature_config(
        kwargs,
        model_parameters,
        is_gemini_3=is_gemini_3,
    )
