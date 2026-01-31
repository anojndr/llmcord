"""Tests for logic utility helpers."""

from __future__ import annotations

from llmcord.logic.utils import patched_get_scripts_list, script_url


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_script_url() -> None:
    """Construct script URL from key and value."""
    assert_true(
        condition=script_url("key", "value")
        == "https://abs.twimg.com/responsive-web/client-web/key.value.js",
        message="Expected script URL format",
    )


def test_patched_get_scripts_list_valid_json() -> None:
    """Parse scripts list from valid JSON-like text."""
    text = 'stuff e=>e+"."+{"key": "value"}[e]+"a.js" stuff'
    scripts = list(patched_get_scripts_list(text))
    assert_true(
        condition=scripts
        == ["https://abs.twimg.com/responsive-web/client-web/key.valuea.js"],
        message="Expected parsed script URL",
    )


def test_patched_get_scripts_list_invalid_json_fixed() -> None:
    """Handle unquoted keys by applying regex fix before parsing."""
    # Regex expects: ([,\{])(\s*)([\w]+_[\w_]+)(\s*):
    text = 'stuff e=>e+"."+{key_name: "value"}[e]+"a.js" stuff'
    scripts = list(patched_get_scripts_list(text))
    assert_true(
        condition=scripts
        == ["https://abs.twimg.com/responsive-web/client-web/key_name.valuea.js"],
        message="Expected parsed script URL with fixed JSON",
    )
