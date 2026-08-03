from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import re
from pathlib import Path

import httpx
import numpy as np
from openai import AsyncOpenAI

from common import PersistentLLMCache, git_commit, sha256_json


DOCUMENT_MARKER = "S2RAG-DOCUMENT-ID"
_ACTIVE_CLIENTS: contextvars.ContextVar["AlignedClients"] = contextvars.ContextVar(
    "s2rag_official_aligned_clients"
)


def set_active_clients(clients: "AlignedClients"):
    return _ACTIVE_CLIENTS.set(clients)


def reset_active_clients(token) -> None:
    _ACTIVE_CLIENTS.reset(token)


async def active_embed(texts: list[str]) -> np.ndarray:
    return await _ACTIVE_CLIENTS.get().embed(texts)


async def active_complete(*args, **kwargs) -> str:
    return await _ACTIVE_CLIENTS.get().complete(*args, **kwargs)


def document_text(document: dict) -> str:
    return (
        f"[{DOCUMENT_MARKER}: {document['document_id']}]\n"
        f"Title: {document.get('title') or document['document_id']}\n"
        f"{document.get('text') or ''}"
    )


def document_hash(documents: list[dict]) -> str:
    return sha256_json(documents)


def prompt_hash(official_repo: Path | None, relative_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = official_repo / relative if official_repo else None
        if path is None or not path.exists():
            continue
        candidates = (
            sorted(item for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            digest.update(str(candidate.relative_to(official_repo)).encode("utf-8"))
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


def rank_marked_context(context: str, documents: list[dict]) -> list[str]:
    valid = {str(item["document_id"]) for item in documents}
    seen: set[str] = set()
    ranked = []
    pattern = rf"\[{re.escape(DOCUMENT_MARKER)}:\s*([^\]\n]+)\]"
    for match in re.finditer(pattern, context):
        document_id = match.group(1).strip()
        if document_id in valid and document_id not in seen:
            seen.add(document_id)
            ranked.append(document_id)
    if ranked:
        return ranked

    positions = []
    for document in documents:
        document_id = str(document["document_id"])
        text = str(document.get("text") or "")
        position = context.find(text)
        if position >= 0:
            positions.append((position, document_id))
    return [document_id for _, document_id in sorted(positions)]


class AlignedClients:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        framework: str,
        official_repo: Path | None,
        documents: list[dict],
        prompt_sha256: str,
    ) -> None:
        self.args = args
        self.framework = framework
        self.commit = git_commit(official_repo)
        headers = {
            "X-S2RAG-Framework": framework,
            "X-S2RAG-Framework-Commit": self.commit,
            "X-S2RAG-Prompt-SHA256": prompt_sha256,
            "X-S2RAG-Document-SHA256": document_hash(documents),
        }
        self.llm = AsyncOpenAI(
            api_key=framework,
            base_url=args.llm_base_url,
            default_headers=headers,
        )
        self.embedding = AsyncOpenAI(
            api_key="local-tei",
            base_url=args.embedding_base_url,
        )
        self.cache = PersistentLLMCache(args.llm_cache)
        self.semaphore = asyncio.Semaphore(max(1, args.global_llm_concurrency))
        self.cache_hits = 0
        self.cache_misses = 0

    async def embed(self, texts: list[str]) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), 64):
            response = await self.embedding.embeddings.create(
                model=self.args.embedding_model,
                input=texts[start : start + 64],
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            batches.extend(item.embedding for item in ordered)
        return np.asarray(batches, dtype=np.float32)

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        for key in (
            "hashing_kv",
            "keyword_extraction",
            "stream",
            "_priority",
            "cache_context",
            "role",
        ):
            kwargs.pop(key, None)
        response_format = kwargs.pop("response_format", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": self.args.llm_model,
            "messages": messages,
            "temperature": 0,
            "extra_body": {"thinking": {"type": "disabled"}},
            **kwargs,
        }
        if keyword_extraction:
            response_format = {"type": "json_object"}
        if response_format:
            request["response_format"] = response_format

        async def request_completion() -> str:
            async with self.semaphore:
                response = await self.llm.chat.completions.create(**request)
            return response.choices[0].message.content or ""

        return await self.cache.get_or_compute(
            {
                "framework": self.framework,
                "commit": self.commit,
                "base_url": self.args.llm_base_url,
                **request,
            },
            request_completion,
        )

    async def chat_json(self, messages: list[dict], *, max_tokens: int = 1000) -> dict:
        response = await self.llm.chat.completions.create(
            model=self.args.llm_model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = response.choices[0].message.content or "{}"
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("entity extraction response is not a JSON object")
        return payload

    async def close(self) -> None:
        await self.cache.close()
        await self.llm.close()
        await self.embedding.close()


async def rerank(
    endpoint: str,
    query: str,
    documents: list[str],
) -> tuple[list[int], list[float]]:
    if not documents:
        return [], []
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            endpoint,
            json={"query": query, "texts": documents, "truncate": True},
        )
        response.raise_for_status()
    payload = response.json()
    items = payload if isinstance(payload, list) else payload.get("results", [])
    ordered = sorted(items, key=lambda item: float(item["score"]), reverse=True)
    return (
        [int(item["index"]) for item in ordered],
        [float(item["score"]) for item in ordered],
    )


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denominator = np.maximum(query_norm * matrix_norm, 1e-12)
    return matrix @ query / denominator
