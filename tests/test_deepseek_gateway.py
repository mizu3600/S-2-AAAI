from scripts.deepseek_gateway import request_cache_key
from s2rag.cache import JsonFileCache


def test_gateway_cache_key_is_shared_across_frameworks():
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "same prompt"}],
        "temperature": 0,
    }

    key = request_cache_key("https://api.deepseek.com/v1/", payload)

    assert key == {
        "upstream": "https://api.deepseek.com/v1",
        "request": payload,
    }
    assert "framework" not in key
    assert JsonFileCache.digest("chat_completions", key) == JsonFileCache.digest(
        "chat_completions",
        request_cache_key("https://api.deepseek.com/v1", payload),
    )
