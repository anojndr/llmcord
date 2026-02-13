from llmcord.logic.providers import _normalize_api_keys

EXPECTED_KEY_COUNT = 2


def test_normalize_api_keys_keeps_string() -> None:
    keys = _normalize_api_keys("single-key")
    assert keys == ["single-key"]


def test_normalize_api_keys_serializes_mapping() -> None:
    keys = _normalize_api_keys({"refresh": "token", "projectId": "my-project"})
    assert len(keys) == 1
    assert '"refresh":"token"' in keys[0]
    assert '"projectId":"my-project"' in keys[0]


def test_normalize_api_keys_serializes_mapping_items_in_list() -> None:
    keys = _normalize_api_keys(
        [
            {"refresh": "token-a", "projectId": "project-a"},
            {"refresh": "token-b", "projectId": "project-b"},
        ],
    )

    assert len(keys) == EXPECTED_KEY_COUNT
    assert '"refresh":"token-a"' in keys[0]
    assert '"refresh":"token-b"' in keys[1]
