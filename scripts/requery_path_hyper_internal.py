from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pathrag", "hypergraphrag"), required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--work-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=60)
    return parser.parse_args()


@asynccontextmanager
async def capture_ranked_text_units(framework: str, module, function_name: str):
    original = getattr(module, function_name)
    capture = type("Capture", (), {"items": [], "framework": framework})()

    async def wrapper(*args, **kwargs):
        result = await original(*args, **kwargs)
        capture.items = [dict(item) for item in result]
        return result

    setattr(module, function_name, wrapper)
    try:
        yield capture
    finally:
        setattr(module, function_name, original)


def load_framework(name: str, repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    if name == "pathrag":
        from PathRAG import PathRAG, QueryParam
        from PathRAG import operate
        from PathRAG.utils import EmbeddingFunc

        return PathRAG, QueryParam, EmbeddingFunc, operate
    from hypergraphrag import HyperGraphRAG, QueryParam, operate
    from hypergraphrag.utils import EmbeddingFunc

    return HyperGraphRAG, QueryParam, EmbeddingFunc, operate


def find_work_dir(parent: Path, framework: str, example_id: str) -> Path | None:
    for root in sorted(parent.glob(f"{framework}_*")):
        candidate = root / example_id
        if candidate.is_dir():
            return candidate
    return None


def map_items(example: dict, items: list[dict]) -> tuple[list[str], list[str], list[str]]:
    texts = list(
        dict.fromkeys(str(item.get("content", "")) for item in items if item.get("content"))
    )
    documents, facts = [], []
    for text in texts:
        for document in example["documents"]:
            if document["document_id"] not in documents and (
                document["text"] in text
                or text in document["text"]
                or f"{document['title']}: {document['text']}" in text
            ):
                documents.append(document["document_id"])
        for fact in example["facts"]:
            if fact["fact_id"] not in facts and (
                fact["sentence"] in text or text in fact["sentence"]
            ):
                facts.append(fact["fact_id"])
    return texts, documents, facts


async def main() -> None:
    args = parse_args()
    logging.disable(logging.INFO)
    rag_class, query_class, embedding_class, operate = load_framework(
        args.framework, args.official_repo
    )
    llm_client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
    )
    embedding_client = AsyncOpenAI(
        api_key=os.environ["SILICONFLOW_API_KEY"], base_url="https://api.siliconflow.cn/v1"
    )
    usage = {}

    async def embed(texts: list[str]) -> np.ndarray:
        response = await embedding_client.embeddings.create(
            model="BAAI/bge-m3", input=texts, encoding_format="float"
        )
        current = usage.setdefault("current", {})
        current["embedding_calls"] = current.get("embedding_calls", 0) + 1
        if response.usage:
            current["embedding_tokens"] = current.get("embedding_tokens", 0) + (
                response.usage.total_tokens or 0
            )
        return np.asarray(
            [item.embedding for item in sorted(response.data, key=lambda item: item.index)],
            dtype=np.float32,
        )

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        for key in ("hashing_kv", "response_format", "stream"):
            kwargs.pop(key, None)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0,
            **kwargs,
        }
        if keyword_extraction:
            request["response_format"] = {"type": "json_object"}
        response = await llm_client.chat.completions.create(**request)
        current = usage.setdefault("current", {})
        current["llm_calls"] = current.get("llm_calls", 0) + 1
        if response.usage:
            current["prompt_tokens"] = current.get("prompt_tokens", 0) + (
                response.usage.prompt_tokens or 0
            )
            current["completion_tokens"] = current.get("completion_tokens", 0) + (
                response.usage.completion_tokens or 0
            )
        return response.choices[0].message.content or ""

    embedding = embedding_class(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed,
        concurrent_limit=8,
    )
    examples = json.loads(args.input.read_text(encoding="utf-8"))[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        {
            json.loads(line)["example_id"]
            for line in args.output.read_text().splitlines()
            if line.strip()
        }
        if args.output.exists()
        else set()
    )
    with args.output.open("a", encoding="utf-8") as output:
        for number, example in enumerate(examples, 1):
            if example["example_id"] in completed:
                continue
            work_dir = find_work_dir(args.work_parent, args.framework, example["example_id"])
            usage["current"] = {}
            record = {
                "system": f"official:{args.framework}",
                "example_id": example["example_id"],
                "status": "success",
                "text_unit_ranking": [],
                "document_ranking": [],
                "fact_ranking": [],
                "usage": {},
                "retrieval_seconds": None,
                "ranking_origin": "official_internal_text_unit_function",
                "error": None,
            }
            started = time.perf_counter()
            try:
                if work_dir is None:
                    raise FileNotFoundError("existing official index not found")
                rag = rag_class(
                    working_dir=str(work_dir),
                    embedding_func=embedding,
                    llm_model_func=complete,
                    llm_model_name="deepseek-chat",
                    entity_extract_max_gleaning=0,
                    enable_llm_cache=True,
                )
                query = query_class(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=args.top_k,
                    max_token_for_text_unit=12000,
                    max_token_for_global_context=8000,
                    max_token_for_local_context=8000,
                )
                async with capture_ranked_text_units(
                    args.framework, operate, "_find_most_related_text_unit_from_entities"
                ) as local_capture:
                    async with capture_ranked_text_units(
                        args.framework, operate, "_find_related_text_unit_from_relationships"
                    ) as global_capture:
                        await rag.aquery(example["question"], query)
                texts, documents, facts = map_items(
                    example, [*global_capture.items, *local_capture.items]
                )
                record.update(
                    {
                        "text_unit_ranking": texts,
                        "document_ranking": documents,
                        "fact_ranking": facts,
                    }
                )
            except Exception as error:
                record["status"] = "error"
                record["error"] = f"{type(error).__name__}: {error}"
            record["retrieval_seconds"] = time.perf_counter() - started
            record["usage"] = usage["current"]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{args.framework} {number}/{len(examples)} {record['status']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
