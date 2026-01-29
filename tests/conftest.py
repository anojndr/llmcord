import sys
from unittest.mock import MagicMock

# Mock tiktoken before any imports that use it
mock_tiktoken = MagicMock()
mock_encoding = MagicMock()
mock_tiktoken.get_encoding.return_value = mock_encoding
sys.modules["tiktoken"] = mock_tiktoken

import pytest
from unittest.mock import AsyncMock
import discord

# Now we can import modules that might use tiktoken
# from llmcord.logic import processor # avoid top level import if possible to be safe, but now mock_tiktoken is set

@pytest.fixture
def mock_config():
    return {
        "bot_token": "test_token",
        "client_id": 123456789,
        "models": {
            "gpt-4": {
                "api_key": "test_key",
                "base_url": "https://api.openai.com/v1",
                "system": "You are a helpful assistant."
            }
        },
        "status_message": "test status",
        "allowed_channel_ids": [],
        "allowed_role_ids": [],
        "max_text": 1000,
        "max_images": 5,
        "max_messages": 10,
        "channel_model_overrides": {},
        "web_search_decider_model": "gpt-4",
    }

@pytest.fixture
def mock_discord_bot():
    bot = MagicMock(spec=discord.ext.commands.Bot)
    bot.user.id = 123456789
    bot.tree = MagicMock()
    bot.tree.sync = AsyncMock()
    return bot

@pytest.fixture
def mock_message():
    message = AsyncMock(spec=discord.Message)
    message.author.id = 12345
    message.content = "Hello bot"
    message.channel.id = 98765
    message.reference = None
    message.attachments = []
    message.replies = []
    message.reply = AsyncMock()
    # Mock mentions
    message.mentions = []
    return message

@pytest.fixture
def mock_interaction(mock_message):
    interaction = AsyncMock(spec=discord.Interaction)
    interaction.user.id = 12345
    interaction.channel_id = 98765
    interaction.followup.send = AsyncMock()
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.channel = mock_message.channel
    return interaction

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch, mock_config):
    # Mock config
    monkeypatch.setattr("llmcord.config.get_config", lambda: mock_config)
    monkeypatch.setattr("llmcord.bot.config", mock_config)
    
    # Mock Database
    mock_db_instance = MagicMock()
    mock_db_instance.get_user_model.return_value = None
    mock_db_instance.set_user_model = MagicMock()
    mock_db_instance.get_user_search_decider_model.return_value = None
    mock_db_instance.set_user_search_decider_model = MagicMock()
    mock_db_instance.get_message_search_data.return_value = (None, None, None) # results, metadata, lens
    
    # We validly need to mock get_bad_keys_db in multiple places where it's imported
    # But patching 'llmcord.services.database.get_bad_keys_db' handles import source
    monkeypatch.setattr("llmcord.services.database.get_bad_keys_db", lambda: mock_db_instance)
    
    # For modules that already imported it:
    # We can rely on tests using mock_dependencies
