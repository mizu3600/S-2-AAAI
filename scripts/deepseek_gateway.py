from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import deque
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
    parser.add_argument("--max-attempts", type=int, default=8)
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
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def completions(
        request: Request,
        authorization: str | None = Header(default=None),
        x_s2rag_framework: str | None = Header(default=None),
    ):
        payload = await request.json()
        namespace = x_s2rag_framework or _bearer_namespace(authorization)
        key = {
            "framework": namespace,
            "upstream": args.upstream,
            "request": payload,
        }
        cached = cache.get("chat_completions", key)
        if isinstance(cached, dict):
            return JSONResponse(cached)

        request_digest = JsonFileCache.digest("chat_completions", key)
        request_lock = request_locks.setdefault(request_digest, asyncio.Lock())
        async with request_lock:
            cached = cache.get("chat_completions", key)
            if isinstance(cached, dict):
                return JSONResponse(cached)

            delay = 1.0
            last_response = None
            for attempt in range(1, args.max_attempts + 1):
                await limiter.acquire()
                async with semaphore:
                    try:
                        response = await client.post("chat/completions", json=payload)
                    except httpx.HTTPError as exc:
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
                        if response.status_code < 400:
                            result = response.json()
                            async with cache_lock:
                                cache.put("chat_completions", key, result)
                            return JSONResponse(result)
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
                await asyncio.sleep(delay)
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


def _bearer_namespace(authorization: str | None) -> str:
    if not authorization:
        return "anonymous"
    prefix = "bearer "
    value = authorization.strip()
    return value[len(prefix):] if value.casefold().startswith(prefix) else value


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
