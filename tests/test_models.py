"""Tests for model defaults."""

# ruff: noqa: S101

import asyncio

import models


def test_msgnode_defaults() -> None:
    """Verify default values on a new message node."""
    node = models.MsgNode()

    assert node.text is None
    assert node.images == []
    assert node.raw_attachments == []
    assert node.role == "assistant"
    assert node.user_id is None
    assert node.thought_signature is None
    assert node.has_bad_attachments is False
    assert node.fetch_parent_failed is False
    assert node.parent_msg is None
    assert node.search_results is None
    assert node.tavily_metadata is None
    assert node.lens_results is None
    assert isinstance(node.lock, asyncio.Lock)
