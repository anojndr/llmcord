"""Tests for configuration helpers and caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from llmcord.config import (
    _CONFIG_STATE,
    ConfigFileEmptyError,
    ConfigFileNotFoundError,
    ProfileConfigError,
    _resolve_config_path,
    clear_config_cache,
    ensure_list,
    get_config,
)


def assert_true(*, condition: bool, message: str) -> None:
    """Raise an AssertionError when a condition is false."""
    if not condition:
        raise AssertionError(message)


def test_ensure_list() -> None:
    """Ensure inputs are normalized to list values."""
    assert_true(
        condition=ensure_list(None) == [],
        message="Expected empty list for None",
    )
    assert_true(
        condition=ensure_list("string") == ["string"],
        message="Expected string wrapped",
    )
    assert_true(
        condition=ensure_list(["list"]) == ["list"],
        message="Expected list preserved",
    )
    assert_true(
        condition=ensure_list(("tuple",)) == ["tuple"],
        message="Expected tuple converted",
    )


def test_resolve_config_path_found(tmp_path: Path) -> None:
    """Resolve existing config path to a Path object."""
    # Test valid path
    f = tmp_path / "config.yaml"
    f.touch()
    with patch("pathlib.Path.exists", return_value=True):
        assert_true(
            condition=_resolve_config_path(str(f)) == Path(str(f)),
            message="Expected resolved config path",
        )


def test_resolve_config_path_not_found() -> None:
    """Raise when configuration path does not exist."""
    with patch("pathlib.Path.exists", return_value=False), pytest.raises(
        ConfigFileNotFoundError,
    ):
        _resolve_config_path("nonexistent.yaml")


def test_get_config_caching() -> None:
    """Cache should serve results within TTL and refresh after expiry."""
    clear_config_cache()
    mock_data = "key: value"

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.open", mock_open(read_data=mock_data)),
    ):

        mock_stat.return_value.st_mtime = 100

        # First call should load
        config1 = get_config("test.yaml")
        assert_true(
            condition=config1 == {"key": "value"},
            message="Expected loaded config",
        )

        # Second call within TTL should hit cache even if mtime changes.
        mock_stat.return_value.st_mtime = 200
        config2 = get_config("test.yaml")
        assert_true(
            condition=config1 is config2,
            message="Expected cached config instance",
        )

        # Force TTL expiry
        _CONFIG_STATE.check_time = 0

        # Third call should reload because mtime changed and TTL expired
        config3 = get_config("test.yaml")
        assert_true(
            condition=config3 == {"key": "value"},
            message="Expected refreshed config",
        )


def test_config_file_empty_error() -> None:
    """Raise when configuration file is empty or invalid."""
    clear_config_cache()
    # Mock yaml.safe_load to return None (empty file)
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat"),
        patch("pathlib.Path.open", mock_open(read_data="")),
        patch("yaml.safe_load", return_value=None),
        pytest.raises(ConfigFileEmptyError),
    ):
        get_config("empty.yaml")


def test_get_config_profile_selection() -> None:
    """Load profile-specific port and bot token into top-level keys."""
    clear_config_cache()
    expected_token = "test-token"  # noqa: S105
    expected_port = 9001
    mock_data = """
profile: test
main:
  port: 8001
  bot_token: main-token
test:
  port: 9001
  bot_token: test-token
client_id: 123
"""

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.open", mock_open(read_data=mock_data)),
    ):
        mock_stat.return_value.st_mtime = 100
        config = get_config("test.yaml")

    assert_true(
        condition=config["profile"] == "test",
        message="Expected profile to be normalized",
    )
    assert_true(
        condition=config["bot_token"] == expected_token,
        message="Expected selected profile bot token",
    )
    assert_true(
        condition=config["port"] == expected_port,
        message="Expected selected profile port",
    )


def test_get_config_profile_validation() -> None:
    """Reject profile blocks with unexpected keys."""
    clear_config_cache()
    mock_data = """
profile: main
main:
  port: 8001
  bot_token: main-token
  extra: nope
test:
  port: 9001
  bot_token: test-token
"""

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.open", mock_open(read_data=mock_data)),
    ):
        mock_stat.return_value.st_mtime = 100
        with pytest.raises(
            ProfileConfigError,
            match="Profiles may only define 'bot_token' and 'port'",
        ):
            get_config("test.yaml")
