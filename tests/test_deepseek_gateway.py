from scripts.deepseek_gateway import (
    apply_thinking_policy,
    request_cache_key,
    response_contains_valid_json,
)
from s2rag.cache import JsonFileCache


def test_gateway_disables_thinking_before_cache_keying():
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "extract entities"}],
        "thinking": {"type": "enabled"},
    }

    normalized = apply_thinking_policy(payload)

    assert normalized["thinking"] == {"type": "disabled"}
    assert payload["thinking"] == {"type": "enabled"}
    assert request_cache_key("https://api.deepseek.com/v1", normalized) != (
        request_cache_key("https://api.deepseek.com/v1", payload)
    )


def test_gateway_can_explicitly_preserve_thinking():
    payload = {
        "model": "deepseek-v4-flash",
        "thinking": {"type": "enabled"},
    }

    assert apply_thinking_policy(payload, allow_thinking=True) is payload


def test_gateway_cache_key_is_framework_scoped():
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "same prompt"}],
        "temperature": 0,
    }

    provenance = {
        "framework": "hipporag2",
        "framework_commit": "abc123",
        "prompt_sha256": "prompt",
        "document_sha256": "document",
    }
    key = request_cache_key(
        "https://api.deepseek.com/v1/",
        payload,
        provenance,
    )

    assert key == {
        "upstream": "https://api.deepseek.com/v1",
        "provenance": provenance,
        "request": payload,
    }
    assert JsonFileCache.digest("chat_completions", key) == JsonFileCache.digest(
        "chat_completions",
        request_cache_key("https://api.deepseek.com/v1", payload, provenance),
    )
    assert JsonFileCache.digest("chat_completions", key) != JsonFileCache.digest(
        "chat_completions",
        request_cache_key(
            "https://api.deepseek.com/v1",
            payload,
            {**provenance, "framework": "hgrag"},
        ),
    )


def test_gateway_rejects_invalid_structured_completion():
    request = {"response_format": {"type": "json_object"}}
    valid = {"choices": [{"message": {"content": '{"entities":[]}'}}]}
    invalid = {"choices": [{"message": {"content": '{"entities":['}}]}

    assert response_contains_valid_json(request, valid)
    assert not response_contains_valid_json(request, invalid)


def test_gateway_does_not_parse_unstructured_completion():
    request = {}
    response = {"choices": [{"message": {"content": "plain answer"}}]}

    assert response_contains_valid_json(request, response)
