"""Tests for utility helpers."""

# ruff: noqa: S101

import utils


def test_script_url() -> None:
    """Build script URLs for Twitter assets."""
    assert (
        utils.script_url("main", "123")
        == "https://abs.twimg.com/responsive-web/client-web/main.123.js"
    )


def test_patched_get_scripts_list_parses_json() -> None:
    """Parse script URLs from JSON mapping text."""
    text = 'prefix e=>e+"."+{"ab_cd":"123"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))

    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/ab_cd.123a.js",
    ]


def test_patched_get_scripts_list_fixes_invalid_json() -> None:
    """Fix invalid JSON before extracting script URLs."""
    text = 'prefix e=>e+"."+{ab_cd:"456"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))

    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/ab_cd.456a.js",
    ]
