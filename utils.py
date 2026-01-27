"""Utility functions and patches for llmcord."""
import json
import re
from collections.abc import Iterator

from twscrape import xclid


def script_url(k: str, v: str) -> str:
    """Generate Twitter script URL."""
    return f"https://abs.twimg.com/responsive-web/client-web/{k}.{v}.js"


def patched_get_scripts_list(text: str) -> Iterator[str]:
    """Patched function for twscrape to handle script parsing."""
    scripts = text.split('e=>e+"."+')[1].split('[e]+"a.js"')[0]

    try:
        for k, v in json.loads(scripts).items():
            yield script_url(k, f"{v}a")
    except json.decoder.JSONDecodeError:
        fixed_scripts = re.sub(
            r"([,\{])(\s*)([\w]+_[\w_]+)(\s*):",
            r'\1\2"\3"\4:',
            scripts,
        )
        for k, v in json.loads(fixed_scripts).items():
            yield script_url(k, f"{v}a")


# Apply the patch
xclid.get_scripts_list = patched_get_scripts_list
