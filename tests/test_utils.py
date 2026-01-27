import utils


def test_script_url() -> None:
    assert (
        utils.script_url("main", "123")
        == "https://abs.twimg.com/responsive-web/client-web/main.123.js"
    )


def test_patched_get_scripts_list_parses_json() -> None:
    text = 'prefix e=>e+"."+{"ab_cd":"123"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))

    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/ab_cd.123a.js",
    ]


def test_patched_get_scripts_list_fixes_invalid_json() -> None:
    text = 'prefix e=>e+"."+{ab_cd:"456"}[e]+"a.js" suffix'
    scripts = list(utils.patched_get_scripts_list(text))

    assert scripts == [
        "https://abs.twimg.com/responsive-web/client-web/ab_cd.456a.js",
    ]
