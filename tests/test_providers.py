from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from s2rag.generation.generator import EvidenceGenerator
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


def test_generator_never_falls_back_when_client_raises():
    class FailingClient:
        def complete(self, system, prompt):
            raise StopRetry("stop")

    with pytest.raises(StopRetry):
        EvidenceGenerator(client=FailingClient()).generate("question", "context")


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
