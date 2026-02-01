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


class ProfileConfigError(ValueError):
    """Raised when profile configuration is invalid."""


# Config caching - reload every 30 seconds at most
class _ConfigCacheState:
    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.mtime: float = 0
        self.check_time: float = 0


_CONFIG_STATE = _ConfigCacheState()
CONFIG_CACHE_TTL = 5  # Check file modification time every 5 seconds
PROFILE_NAMES = ("main", "test")
PROFILE_KEYS = {"bot_token", "port"}


def _normalize_profile_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize profile-aware config settings.

    If profile settings are present, copy the selected profile's port and bot
    token into top-level keys for backwards compatibility.
    """
    has_profiles = any(profile in config for profile in PROFILE_NAMES)
    has_profile_selector = "profile" in config

    if not has_profiles and not has_profile_selector:
        return config

    if not all(profile in config for profile in PROFILE_NAMES):
        message = "Both 'main' and 'test' profiles must be defined in config.yaml."
        raise ProfileConfigError(message)

    profile_name = config.get("profile") or "main"
    if profile_name not in PROFILE_NAMES:
        message = "Config 'profile' must be either 'main' or 'test'."
        raise ProfileConfigError(message)

    for profile in PROFILE_NAMES:
        profile_config = config.get(profile)
        if not isinstance(profile_config, dict):
            message = f"Profile '{profile}' must be a mapping of settings."
            raise TypeError(message)

        invalid_keys = set(profile_config) - PROFILE_KEYS
        if invalid_keys:
            invalid_keys_list = ", ".join(sorted(invalid_keys))
            message = (
                "Profiles may only define 'bot_token' and 'port'. "
                f"Invalid keys in '{profile}': {invalid_keys_list}."
            )
            raise ProfileConfigError(message)

        missing_keys = PROFILE_KEYS - set(profile_config)
        if missing_keys:
            missing_keys_list = ", ".join(sorted(missing_keys))
            message = f"Profile '{profile}' is missing: {missing_keys_list}."
            raise ProfileConfigError(message)

    selected_profile = config[profile_name]
    config["profile"] = profile_name
    config["bot_token"] = selected_profile["bot_token"]
    config["port"] = selected_profile["port"]
    return config


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
                _CONFIG_STATE.cache = _normalize_profile_config(loaded_config)

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
