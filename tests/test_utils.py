from __future__ import annotations

from llmcord.logic import utils


def test_script_url() -> None:
    assert (
        utils.script_url("foo", "bar")
        == "https://abs.twimg.com/responsive-web/client-web/foo.bar.js"
    )


def test_patched_get_scripts_list_valid_json() -> None:
    text = 'prefix e=>e+"."+{"a":"1","b":"2"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))
    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/a.1a.js",
        "https://abs.twimg.com/responsive-web/client-web/b.2a.js",
    ]


def test_patched_get_scripts_list_invalid_json() -> None:
    text = 'prefix e=>e+"."+{a_b:"1"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))
    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/a_b.1a.js",
    ]
