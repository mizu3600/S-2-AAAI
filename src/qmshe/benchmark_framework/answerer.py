from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx

from qmshe.benchmark_framework.schemas import CanonicalExample, StandardTrace


class SharedDeepSeekAnswerer:
    """One deterministic generator for comparing retrieval contexts fairly."""

    def __init__(self, model: str = "deepseek-chat", concurrency: int = 12):
        self.model = model
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            base_url="https://api.deepseek.com",
            headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
            timeout=180,
        )

    async def enrich(self, example: CanonicalExample, trace: StandardTrace) -> StandardTrace:
        documents = {document.document_id: document for document in example.documents}
        selected = [documents[item] for item in trace.document_ranking[:5] if item in documents]
        context = "\n\n".join(
            f"[{document.document_id}] {document.title}\n{document.text}" for document in selected
        )
        prompt = (
            "Answer the question using only the evidence. Return strict JSON with keys answer and "
            "citations. citations must be a list of the supporting document IDs shown in brackets.\n"
            f"Question: {example.question}\nEvidence:\n{context}"
        )
        async with self.semaphore:
            started = time.perf_counter()
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
            )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"].get("content") or "{}"
        parsed = _json_object(content)
        trace.answer = str(parsed.get("answer", ""))
        valid = set(trace.document_ranking)
        trace.citations = [str(item) for item in parsed.get("citations", []) if str(item) in valid]
        trace.citation_level = "document"
        usage = payload.get("usage") or {}
        trace.usage.llm_calls += 1
        trace.usage.prompt_tokens = (trace.usage.prompt_tokens or 0) + usage.get("prompt_tokens", 0)
        trace.usage.completion_tokens = (trace.usage.completion_tokens or 0) + usage.get(
            "completion_tokens", 0
        )
        cache_hit = usage.get("prompt_cache_hit_tokens", 0)
        cache_miss = usage.get(
            "prompt_cache_miss_tokens", max(usage.get("prompt_tokens", 0) - cache_hit, 0)
        )
        generation_cost = (
            cache_hit * 0.0028 + cache_miss * 0.14 + usage.get("completion_tokens", 0) * 0.28
        ) / 1_000_000
        trace.usage.api_cost_usd = (trace.usage.api_cost_usd or 0.0) + generation_cost
        trace.usage.token_count_mode = (
            "generation_measured_index_unavailable"
            if "unavailable" in trace.usage.token_count_mode
            else "measured"
        )
        generation = time.perf_counter() - started
        trace.timing.generation_seconds = generation
        trace.timing.total_seconds = (trace.timing.total_seconds or 0) + generation
        trace.metadata["answer_generator"] = self.model
        return trace

    async def close(self) -> None:
        await self.client.aclose()


def _json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
