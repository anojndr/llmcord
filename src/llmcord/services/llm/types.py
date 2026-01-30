from dataclasses import dataclass

@dataclass(slots=True)
class LiteLLMOptions:
    """Optional configuration for building LiteLLM kwargs."""

    base_url: str | None = None
    extra_headers: dict | None = None
    stream: bool = False
    model_parameters: dict | None = None
    temperature: float | None = None
    enable_grounding: bool = False
