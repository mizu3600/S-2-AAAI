import json
import logging
import threading
import time
from collections.abc import Callable

import httpx

from s2rag.cache import JsonFileCache
from s2rag.settings import Settings, get_settings


class ProviderError(RuntimeError):
    pass


class PermanentProviderError(ProviderError):
    pass


logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        http_client: httpx.Client | None = None,
        cache: JsonFileCache | None = None,
        cache_namespace: str = "s2rag",
    ):
        self.settings = settings or get_settings()
        self._sleep = sleep
        self._semaphore = threading.BoundedSemaphore(
            self.settings.deepseek_max_concurrency
        )
        self._request_locks_guard = threading.Lock()
        self._request_locks: dict[str, threading.Lock] = {}
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            base_url=self.settings.deepseek_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.settings.deepseek_api_key or ''}"
            },
            timeout=self.settings.request_timeout,
            limits=httpx.Limits(
                max_connections=self.settings.deepseek_pool_max_connections,
                max_keepalive_connections=(
                    self.settings.deepseek_pool_keepalive_connections
                ),
            ),
        )
        self._cache = cache or JsonFileCache(
            self.settings.deepseek_response_cache_dir
        )
        self.cache_namespace = cache_namespace

    def complete_json(
        self,
        system: str,
        prompt: str,
        *,
        cache_namespace: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        request_key = self._request_key(
            system,
            prompt,
            {"type": "json_object"},
            max_tokens,
        )
        namespace = cache_namespace or self.cache_namespace
        cached = self._cache.get(namespace, request_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "json"
            and isinstance(cached.get("value"), dict)
        ):
            return cached["value"]
        delay = self.settings.deepseek_retry_initial_seconds
        for attempt in range(1, self.settings.deepseek_max_attempts + 1):
            text = self.complete(
                system,
                prompt,
                response_format={"type": "json_object"},
                cache_namespace=f"{namespace}.raw",
                max_tokens=max_tokens,
            )
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError("DeepSeek JSON response is not an object")
                self._cache.put(
                    namespace,
                    request_key,
                    {"kind": "json", "value": payload},
                )
                return payload
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt >= self.settings.deepseek_max_attempts:
                    raise ProviderError(
                        "DeepSeek returned invalid JSON after "
                        f"{self.settings.deepseek_max_attempts} attempts"
                    ) from exc
                logger.warning(
                    "DeepSeek returned invalid JSON; retrying in %.1fs: %s",
                    delay,
                    exc,
                )
                self._sleep(delay)
                delay = min(delay * 2, self.settings.deepseek_retry_max_seconds)
        raise AssertionError("unreachable")

    def complete(
        self,
        system: str,
        prompt: str,
        response_format: dict | None = None,
        *,
        cache_namespace: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        request_key = self._request_key(system, prompt, response_format, max_tokens)
        namespace = cache_namespace or self.cache_namespace
        cached = self._cache.get(namespace, request_key)
        if (
            isinstance(cached, dict)
            and cached.get("kind") == "text"
            and isinstance(cached.get("value"), str)
        ):
            return cached["value"]
        lock = self._request_lock(
            JsonFileCache.digest(namespace, request_key)
        )
        with lock:
            cached = self._cache.get(namespace, request_key)
            if (
                isinstance(cached, dict)
                and cached.get("kind") == "text"
                and isinstance(cached.get("value"), str)
            ):
                return cached["value"]
            delay = self.settings.deepseek_retry_initial_seconds
            for attempt in range(1, self.settings.deepseek_max_attempts + 1):
                try:
                    text = self._complete_once(
                        system,
                        prompt,
                        response_format,
                        max_tokens=max_tokens,
                    )
                    self._cache.put(
                        namespace,
                        request_key,
                        {"kind": "text", "value": text},
                    )
                    return text
                except PermanentProviderError:
                    raise
                except Exception as exc:
                    if attempt >= self.settings.deepseek_max_attempts:
                        if isinstance(exc, ProviderError):
                            raise
                        raise ProviderError(
                            "DeepSeek request failed after "
                            f"{self.settings.deepseek_max_attempts} attempts"
                        ) from exc
                    logger.warning(
                        "DeepSeek request failed; retrying in %.1fs: %s",
                        delay,
                        exc,
                    )
                    self._sleep(delay)
                    delay = min(delay * 2, self.settings.deepseek_retry_max_seconds)
        raise AssertionError("unreachable")

    def generation_config(self) -> dict:
        return {
            "model_id": self.settings.deepseek_model,
            "temperature": self.settings.deepseek_temperature,
            "max_tokens": self.settings.deepseek_max_tokens,
            "retry_policy": "bounded_exponential_backoff",
            "retry_initial_seconds": self.settings.deepseek_retry_initial_seconds,
            "retry_max_seconds": self.settings.deepseek_retry_max_seconds,
            "max_attempts": self.settings.deepseek_max_attempts,
            "max_concurrency": self.settings.deepseek_max_concurrency,
            "connection_pool": "persistent_httpx",
            "response_cache": "content_addressed_v1",
        }

    def _complete_once(
        self,
        system: str,
        prompt: str,
        response_format: dict | None,
        *,
        max_tokens: int | None = None,
    ) -> str:
        if not self.settings.deepseek_api_key:
            raise PermanentProviderError("DEEPSEEK_API_KEY is not configured")
        payload = {
            "model": self.settings.deepseek_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.deepseek_temperature,
            "max_tokens": max_tokens or self.settings.deepseek_max_tokens,
            "thinking": {"type": "disabled"},
        }
        if response_format:
            payload["response_format"] = response_format
        with self._semaphore:
            response = self._http_client.post(
                "chat/completions",
                json=payload,
            )
        if response.is_error:
            error_type = (
                PermanentProviderError
                if response.status_code in {400, 401, 402, 403, 404, 422}
                else ProviderError
            )
            raise error_type(
                f"DeepSeek request failed ({response.status_code}): {response.text[:300]}"
            )
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("DeepSeek returned an empty completion")
        return content

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _request_key(
        self,
        system: str,
        prompt: str,
        response_format: dict | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": self.settings.deepseek_model,
            "temperature": self.settings.deepseek_temperature,
            "max_tokens": max_tokens or self.settings.deepseek_max_tokens,
            "system": system,
            "prompt": prompt,
            "response_format": response_format,
        }

    def _request_lock(self, key: str) -> threading.Lock:
        with self._request_locks_guard:
            return self._request_locks.setdefault(key, threading.Lock())
