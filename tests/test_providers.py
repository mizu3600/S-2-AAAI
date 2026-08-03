from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import httpx
import pytest

from s2rag.generation.generator import EvidenceGenerator
from s2rag.generation.prompt_builder import SYSTEM_PROMPT, build_prompt
from s2rag.providers import DeepSeekClient, ProviderError
from s2rag.settings import Settings


class StopRetry(RuntimeError):
    pass


def test_deepseek_stops_after_bounded_exponential_backoff(tmp_path):
    delays = []

    def record_delay(delay):
        delays.append(delay)

    client = DeepSeekClient(
        Settings(
            deepseek_api_key="test",
            deepseek_retry_initial_seconds=0.25,
            deepseek_retry_max_seconds=0.5,
            deepseek_max_attempts=4,
            deepseek_response_cache_dir=tmp_path,
        ),
        sleep=record_delay,
    )
    client._complete_once = lambda *args, **kwargs: (_ for _ in ()).throw(
        ProviderError("temporary")
    )

    with pytest.raises(ProviderError, match="temporary"):
        client.complete("system", "prompt")

    assert delays == [0.25, 0.5, 0.5]


def test_invalid_json_is_evicted_before_short_retry(tmp_path):
    delays = []
    responses = iter(('{"facts":[', '{"facts":[]}'))
    client = DeepSeekClient(
        Settings(
            deepseek_api_key="test",
            deepseek_json_retry_initial_seconds=0.25,
            deepseek_json_retry_max_seconds=0.5,
            deepseek_json_max_attempts=2,
            deepseek_response_cache_dir=tmp_path,
        ),
        sleep=delays.append,
    )
    client._complete_once = lambda *args, **kwargs: next(responses)

    assert client.complete_json("system", "prompt") == {"facts": []}
    assert delays == [0.25]


def test_json_response_is_repaired_before_retry(tmp_path):
    client = DeepSeekClient(
        Settings(
            deepseek_api_key="test",
            deepseek_json_max_attempts=1,
            deepseek_response_cache_dir=tmp_path,
        )
    )
    client._complete_once = (
        lambda *args, **kwargs: '```json\n{"facts": []}\n```\nextra text'
    )

    assert client.complete_json("system", "prompt") == {"facts": []}


def test_truncated_completion_is_reported_without_caching(tmp_path):
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": '{"facts": ['},
                    }
                ]
            },
        )

    client = DeepSeekClient(
        Settings(
            deepseek_api_key="test",
            deepseek_max_attempts=1,
            deepseek_response_cache_dir=tmp_path,
        ),
        http_client=httpx.Client(
            base_url="https://example.test",
            transport=httpx.MockTransport(respond),
        ),
    )

    with pytest.raises(ProviderError, match="truncated at max_tokens"):
        client.complete("system", "prompt")

    with pytest.raises(ProviderError, match="truncated at max_tokens"):
        client.complete("system", "prompt")

    assert len(requests) == 2


def test_generator_never_falls_back_when_client_raises():
    class FailingClient:
        def complete(self, system, prompt):
            raise StopRetry("stop")

    with pytest.raises(StopRetry):
        EvidenceGenerator(client=FailingClient()).generate("question", "context")


def test_generator_uses_unified_concise_prompt_without_system_message():
    calls = []

    class RecordingClient:
        def complete(self, system, prompt):
            calls.append((system, prompt))
            return "answer"

    answer = EvidenceGenerator(client=RecordingClient()).generate(
        "Who won?", "[e1] Ada won."
    )

    assert answer == "answer"
    assert calls == [
        (
            "",
            "Answer concisely using only the evidence.\n\n"
            "Question:\nWho won?\n\nEvidence:\n[e1] Ada won.",
        )
    ]
    assert SYSTEM_PROMPT == ""
    assert build_prompt("Q", "E").startswith(
        "Answer concisely using only the evidence."
    )


def test_empty_system_prompt_is_omitted_from_deepseek_messages(tmp_path):
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ada"}}]},
        )

    settings = Settings(
        deepseek_api_key="test",
        deepseek_response_cache_dir=tmp_path,
    )
    http_client = httpx.Client(
        base_url="https://example.test",
        transport=httpx.MockTransport(respond),
    )
    client = DeepSeekClient(settings, http_client=http_client)

    assert client.complete("", "Answer concisely.") == "Ada"
    assert requests[0]["messages"] == [
        {"role": "user", "content": "Answer concisely."}
    ]


def test_concurrent_identical_requests_are_coalesced(tmp_path):
    client = DeepSeekClient(
        Settings(
            deepseek_api_key="test",
            deepseek_response_cache_dir=tmp_path,
        )
    )
    call_count = 0
    count_lock = threading.Lock()

    def complete_once(*args, **kwargs):
        nonlocal call_count
        with count_lock:
            call_count += 1
        time.sleep(0.05)
        return "shared result"

    client._complete_once = complete_once
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: client.complete("system", "identical prompt"),
                range(8),
            )
        )

    assert results == ["shared result"] * 8
    assert call_count == 1
