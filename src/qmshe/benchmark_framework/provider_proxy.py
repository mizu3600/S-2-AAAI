from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass
class ProxyStats:
    llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    api_cost_usd: float = 0.0
    upstream_errors: int = 0


def create_provider_proxy() -> FastAPI:
    app = FastAPI(title="Benchmark Provider Proxy")
    app.state.stats = ProxyStats()
    app.state.lock = asyncio.Lock()

    @app.post("/reset")
    async def reset():
        async with app.state.lock:
            app.state.stats = ProxyStats()
        return {"status": "reset"}

    @app.get("/stats")
    async def stats():
        return asdict(app.state.stats)

    @app.post("/{path:path}")
    async def forward(path: str, request: Request):
        payload = await request.json()
        is_embedding = path.endswith("embeddings")
        base_url = "https://api.siliconflow.cn/v1" if is_embedding else "https://api.deepseek.com"
        api_key = os.environ["SILICONFLOW_API_KEY" if is_embedding else "DEEPSEEK_API_KEY"]
        upstream_path = "/embeddings" if is_embedding else "/chat/completions"
        async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
            response = await client.post(
                upstream_path,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": response.text}}
        async with app.state.lock:
            usage = body.get("usage") or {}
            if is_embedding:
                app.state.stats.embedding_calls += 1
                tokens = usage.get("total_tokens", usage.get("prompt_tokens", 0))
                app.state.stats.embedding_tokens += tokens or 0
            else:
                app.state.stats.llm_calls += 1
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                hit = usage.get("prompt_cache_hit_tokens", 0)
                miss = usage.get("prompt_cache_miss_tokens", max(prompt - hit, 0))
                app.state.stats.prompt_tokens += prompt
                app.state.stats.completion_tokens += completion
                app.state.stats.api_cost_usd += (
                    hit * 0.0028 + miss * 0.14 + completion * 0.28
                ) / 1_000_000
            if response.status_code >= 400:
                app.state.stats.upstream_errors += 1
        return JSONResponse(body, status_code=response.status_code)

    return app


app = create_provider_proxy()
