from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from s2rag.cache import JsonFileCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One process-wide DeepSeek pool, cache and limiter for all benchmark runners"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--upstream", default="https://api.deepseek.com/v1")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--requests-per-minute", type=int, default=0)
    parser.add_argument("--tokens-per-minute", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--invalid-json-max-attempts", type=int, default=3)
    parser.add_argument("--invalid-json-retry-delay", type=float, default=0.25)
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help=(
            "Forward the caller's DeepSeek thinking setting. By default the "
            "benchmark gateway disables thinking to prevent reasoning tokens "
            "from exhausting the completion budget before structured output."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/deepseek_gateway"))
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> FastAPI:
    api_key = os.environ.get("UPSTREAM_DEEPSEEK_API_KEY") or os.environ.get(
        "DEEPSEEK_API_KEY"
    )
    if not api_key:
        raise RuntimeError("UPSTREAM_DEEPSEEK_API_KEY is required")
    app = FastAPI(title="S2RAG DeepSeek benchmark gateway")
    semaphore = asyncio.Semaphore(args.max_concurrency)
    cache = JsonFileCache(args.cache_dir)
    cache_lock = asyncio.Lock()
    request_locks: dict[str, asyncio.Lock] = {}
    limiter = SlidingWindowLimiter(args.requests_per_minute)
    token_limiter = WeightedSlidingWindowLimiter(args.tokens_per_minute)
    stats: Counter[str] = Counter()
    framework_requests: Counter[str] = Counter()
    framework_stats: dict[str, Counter[str]] = {}
    client = httpx.AsyncClient(
        base_url=args.upstream.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180,
        limits=httpx.Limits(
            max_connections=args.max_concurrency,
            max_keepalive_connections=args.max_concurrency,
        ),
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "max_concurrency": args.max_concurrency,
            "requests_per_minute": args.requests_per_minute,
            "tokens_per_minute": args.tokens_per_minute,
            "thinking_policy": (
                "caller_controlled" if args.allow_thinking else "forced_disabled"
            ),
            "inflight_request_keys": sum(
                request_lock.locked() for request_lock in request_locks.values()
            ),
            "stats": dict(stats),
            "framework_requests": dict(framework_requests),
            "framework_stats": {
                name: dict(values) for name, values in framework_stats.items()
            },
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def completions(
        request: Request,
        x_s2rag_framework: str | None = Header(default=None),
        x_s2rag_framework_commit: str | None = Header(default=None),
        x_s2rag_prompt_sha256: str | None = Header(default=None),
        x_s2rag_document_sha256: str | None = Header(default=None),
    ):
        payload = apply_thinking_policy(
            await request.json(),
            allow_thinking=args.allow_thinking,
        )
        framework = x_s2rag_framework or downstream_identity(request)
        provenance = {
            "framework": framework,
            "framework_commit": x_s2rag_framework_commit or "unspecified",
            "prompt_sha256": x_s2rag_prompt_sha256 or "implicit_in_request",
            "document_sha256": x_s2rag_document_sha256 or "implicit_in_request",
        }
        per_framework = framework_stats.setdefault(framework, Counter())
        framework_requests[framework] += 1
        per_framework["requests"] += 1
        key = request_cache_key(args.upstream, payload, provenance)
        cached = cache.get("chat_completions", key)
        if isinstance(cached, dict):
            if response_contains_valid_json(payload, cached):
                stats["cache_hits"] += 1
                per_framework["cache_hits"] += 1
                return JSONResponse(cached)
            cache.delete("chat_completions", key)
            stats["invalid_cache_evictions"] += 1
            per_framework["invalid_cache_evictions"] += 1
        stats["cache_misses"] += 1
        per_framework["cache_misses"] += 1

        request_digest = JsonFileCache.digest("chat_completions", key)
        request_lock = request_locks.setdefault(request_digest, asyncio.Lock())
        if request_lock.locked():
            stats["coalesced_waits"] += 1
        async with request_lock:
            cached = cache.get("chat_completions", key)
            if isinstance(cached, dict):
                if response_contains_valid_json(payload, cached):
                    stats["cache_hits_after_wait"] += 1
                    per_framework["cache_hits_after_wait"] += 1
                    return JSONResponse(cached)
                cache.delete("chat_completions", key)
                stats["invalid_cache_evictions"] += 1
                per_framework["invalid_cache_evictions"] += 1

            delay = 1.0
            last_response = None
            invalid_json_attempts = 0
            for attempt in range(1, args.max_attempts + 1):
                fast_retry = False
                await limiter.acquire()
                await token_limiter.acquire(estimate_request_tokens(payload))
                async with semaphore:
                    stats["inflight"] += 1
                    stats["peak_inflight"] = max(
                        stats["peak_inflight"], stats["inflight"]
                    )
                    stats["upstream_requests"] += 1
                    per_framework["upstream_requests"] += 1
                    try:
                        response = await client.post("chat/completions", json=payload)
                    except httpx.HTTPError as exc:
                        stats["upstream_transport_errors"] += 1
                        per_framework["upstream_transport_errors"] += 1
                        if attempt >= args.max_attempts:
                            return JSONResponse(
                                {
                                    "error": {
                                        "message": str(exc),
                                        "type": type(exc).__name__,
                                    }
                                },
                                status_code=502,
                            )
                    else:
                        last_response = response
                        stats[f"upstream_status_{response.status_code}"] += 1
                        per_framework[f"upstream_status_{response.status_code}"] += 1
                        if response.status_code == 429:
                            stats["upstream_429s"] += 1
                            per_framework["upstream_429s"] += 1
                        if response.status_code < 400:
                            result = response.json()
                            usage = result.get("usage") or {}
                            stats["prompt_tokens"] += int(
                                usage.get("prompt_tokens") or 0
                            )
                            stats["completion_tokens"] += int(
                                usage.get("completion_tokens") or 0
                            )
                            if response_contains_valid_json(payload, result):
                                async with cache_lock:
                                    cache.put("chat_completions", key, result)
                                return JSONResponse(result)
                            stats["upstream_invalid_json"] += 1
                            per_framework["upstream_invalid_json"] += 1
                            invalid_json_attempts += 1
                            if invalid_json_attempts >= args.invalid_json_max_attempts:
                                return JSONResponse(result)
                            fast_retry = True
                        if response.status_code in {400, 401, 402, 403, 404, 422}:
                            return JSONResponse(
                                _response_payload(response),
                                status_code=response.status_code,
                            )
                        if attempt >= args.max_attempts:
                            return JSONResponse(
                                _response_payload(response),
                                status_code=response.status_code,
                            )
                    finally:
                        stats["inflight"] -= 1
                stats["retries"] += 1
                per_framework["retries"] += 1
                await asyncio.sleep(
                    args.invalid_json_retry_delay if fast_retry else delay
                )
                if not fast_retry:
                    delay = min(delay * 2, 60.0)
            return JSONResponse(
                (
                    _response_payload(last_response)
                    if last_response
                    else {"error": "unreachable"}
                ),
                status_code=502,
            )

    @app.on_event("shutdown")
    async def shutdown():
        await client.aclose()

    return app


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = max(0, requests_per_minute)
        self.events: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        if not self.limit:
            return
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0] >= 60:
                    self.events.popleft()
                if len(self.events) < self.limit:
                    self.events.append(now)
                    return
                delay = max(0.01, 60 - (now - self.events[0]))
            await asyncio.sleep(delay)


class WeightedSlidingWindowLimiter:
    def __init__(self, units_per_minute: int):
        self.limit = max(0, units_per_minute)
        self.events: deque[tuple[float, int]] = deque()
        self.total = 0
        self.lock = asyncio.Lock()

    async def acquire(self, units: int) -> None:
        if not self.limit:
            return
        units = min(max(1, units), self.limit)
        while True:
            async with self.lock:
                now = time.monotonic()
                while self.events and now - self.events[0][0] >= 60:
                    _, expired = self.events.popleft()
                    self.total -= expired
                if self.total + units <= self.limit:
                    self.events.append((now, units))
                    self.total += units
                    return
                delay = max(0.01, 60 - (now - self.events[0][0]))
            await asyncio.sleep(delay)


def downstream_identity(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.casefold().startswith("bearer "):
        token = authorization[7:].strip()
        if token and token not in {"none", "dummy", "unused"}:
            return token[:80]
    return "unspecified"


def estimate_request_tokens(payload: dict) -> int:
    message_characters = sum(
        len(str(message.get("content") or ""))
        for message in payload.get("messages") or []
        if isinstance(message, dict)
    )
    prompt_estimate = max(1, (message_characters + 3) // 4)
    return prompt_estimate + int(payload.get("max_tokens") or 0)


def apply_thinking_policy(payload: dict, *, allow_thinking: bool = False) -> dict:
    if allow_thinking:
        return payload
    normalized = dict(payload)
    normalized["thinking"] = {"type": "disabled"}
    return normalized


def response_contains_valid_json(request_payload: dict, response_payload: dict) -> bool:
    response_format = request_payload.get("response_format")
    if response_format != {"type": "json_object"}:
        return True
    try:
        content = response_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def request_cache_key(
    upstream: str,
    payload: dict,
    provenance: dict | None = None,
) -> dict:
    """Keep official-framework extraction caches independent and auditable."""
    return {
        "upstream": upstream.rstrip("/"),
        "provenance": provenance or {
            "framework": "unspecified",
            "framework_commit": "unspecified",
            "prompt_sha256": "implicit_in_request",
            "document_sha256": "implicit_in_request",
        },
        "request": payload,
    }


def _response_payload(response: httpx.Response) -> dict:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {"error": payload}
    except (json.JSONDecodeError, ValueError):
        return {"error": {"message": response.text[:1000]}}


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
