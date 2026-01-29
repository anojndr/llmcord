import pytest
from llmcord.logic.utils import script_url, patched_get_scripts_list

def test_script_url():
    assert script_url("key", "value") == "https://abs.twimg.com/responsive-web/client-web/key.value.js"

def test_patched_get_scripts_list_valid_json():
    text = 'stuff e=>e+"."+{"key": "value"}[e]+"a.js" stuff'
    scripts = list(patched_get_scripts_list(text))
    assert scripts == ["https://abs.twimg.com/responsive-web/client-web/key.valuea.js"]

def test_patched_get_scripts_list_invalid_json_fixed():
    # simulate the case where keys are not quoted, which standard json.loads fails on, but the regex fix handles
    # The regex expects: ([,\{])(\s*)([\w]+_[\w_]+)(\s*):
    text = 'stuff e=>e+"."+{key_name: "value"}[e]+"a.js" stuff'
    scripts = list(patched_get_scripts_list(text))
    assert scripts == ["https://abs.twimg.com/responsive-web/client-web/key_name.valuea.js"]
