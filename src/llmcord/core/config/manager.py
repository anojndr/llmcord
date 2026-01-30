"""Configuration manager for loading and caching config."""

import time
from pathlib import Path
from typing import Any

import yaml


class ConfigFileNotFoundError(FileNotFoundError):
    """Raised when a configuration file cannot be found."""

    def __init__(self, filename: str) -> None:
        """Initialize the error with the missing filename."""
        self.filename = filename
        super().__init__(filename)

    def __str__(self) -> str:
        """Return a readable error message."""
        return (
            f"Config file '{self.filename}' not found in current directory or "
            "/etc/secrets/"
        )


class ConfigFileEmptyError(ValueError):
    """Raised when a configuration file is empty or corrupted."""

    def __init__(self, path: Path) -> None:
        """Initialize the error with the invalid config path."""
        self.path = path
        super().__init__(path)

    def __str__(self) -> str:
        """Return a readable error message."""
        return f"Config file is empty or corrupted: {self.path}"


# Config caching - reload every 30 seconds at most
class _ConfigCacheState:
    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.mtime: float = 0
        self.check_time: float = 0


_CONFIG_STATE = _ConfigCacheState()
CONFIG_CACHE_TTL = 5  # Check file modification time every 5 seconds


def _resolve_config_path(filename: str) -> Path:
    candidates = [Path(filename), Path("/etc/secrets") / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ConfigFileNotFoundError(filename)


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file with caching.

    Only reloads if file has been modified (checked every `CONFIG_CACHE_TTL` seconds).
    """
    current_time = time.time()

    # Only check file mtime periodically to avoid stat() on every call
    if (
        current_time - _CONFIG_STATE.check_time > CONFIG_CACHE_TTL
        or not _CONFIG_STATE.cache
    ):
        _CONFIG_STATE.check_time = current_time

        filepath = _resolve_config_path(filename)
        file_mtime = filepath.stat().st_mtime

        # Only reload if file was modified
        if file_mtime != _CONFIG_STATE.mtime or not _CONFIG_STATE.cache:
            _CONFIG_STATE.mtime = file_mtime
            with filepath.open(encoding="utf-8") as file:
                loaded_config = yaml.safe_load(file)
                # Handle empty/corrupted YAML that returns None
                if loaded_config is None:
                    raise ConfigFileEmptyError(filepath)
                _CONFIG_STATE.cache = loaded_config

    return _CONFIG_STATE.cache


def clear_config_cache() -> None:
    """Clear the config cache to force a reload on next `get_config()` call."""
    _CONFIG_STATE.cache = {}
    _CONFIG_STATE.mtime = 0
    _CONFIG_STATE.check_time = 0


def ensure_list(value: str | list[str] | None) -> list[str]:
    """Convert a value to a list if it isn't one already.

    This is commonly needed for API keys which may be configured as either
    a single string or a list of strings.

    Args:
        value: A string, list, or None.

    Returns:
        - If value is None: empty list.
        - If value is a string: single-element list containing that string.
        - If value is already a list: return as-is.

    Examples:
        >>> ensure_list("key123")
        ['key123']
        >>> ensure_list(["key1", "key2"])
        ['key1', 'key2']
        >>> ensure_list(None)
        []

    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
