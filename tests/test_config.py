import pytest
import time
from pathlib import Path
from unittest.mock import patch, mock_open
from llmcord.config import (
    get_config,
    _resolve_config_path,
    ensure_list,
    ConfigFileNotFoundError,
    ConfigFileEmptyError,
    _CONFIG_STATE,
    clear_config_cache,
)

def test_ensure_list():
    assert ensure_list(None) == []
    assert ensure_list("string") == ["string"]
    assert ensure_list(["list"]) == ["list"]
    assert ensure_list(("tuple",)) == ["tuple"]

def test_resolve_config_path_found(tmp_path):
    # Test valid path
    f = tmp_path / "config.yaml"
    f.touch()
    with patch("pathlib.Path.exists", return_value=True):
        assert _resolve_config_path(str(f)) == Path(str(f))

def test_resolve_config_path_not_found():
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(ConfigFileNotFoundError):
            _resolve_config_path("nonexistent.yaml")

def test_get_config_caching():
    clear_config_cache()
    mock_data = 'key: value'
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat, \
         patch("pathlib.Path.open", mock_open(read_data=mock_data)):
        
        mock_stat.return_value.st_mtime = 100
        
        # First call should load
        config1 = get_config("test.yaml")
        assert config1 == {"key": "value"}
        
        # Second call within TTL should hit cache (even if mtime changes, we don't check yet)
        mock_stat.return_value.st_mtime = 200
        config2 = get_config("test.yaml")
        assert config1 is config2
        
        # Force TTL expiry
        _CONFIG_STATE.check_time = 0
        
        # Third call should reload because mtime changed and TTL expired
        config3 = get_config("test.yaml")
        assert config3 == {"key": "value"}

def test_config_file_empty_error():
    clear_config_cache()
    # Mock yaml.safe_load to return None (empty file)
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat"), \
         patch("pathlib.Path.open", mock_open(read_data="")), \
         patch("yaml.safe_load", return_value=None):
            
        with pytest.raises(ConfigFileEmptyError):
            get_config("empty.yaml")
