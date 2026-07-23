from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--embedding-base-url", default="https://api.siliconflow.cn/v1")
    return parser.parse_args()


def document_text(document: dict) -> str:
    return f"{document['title']}: {document['text']}"


def chunk_ranking(result: dict, documents: list[dict]) -> list[str]:
    valid = {document["document_id"] for document in documents}
    ranking = []
    data = result.get("data") or {}
    for chunk in data.get("chunks") or []:
        candidates = (
            chunk.get("file_path"),
            chunk.get("document_id"),
            chunk.get("full_doc_id"),
        )
        document_id = next((item for item in candidates if item in valid), None)
        if document_id is None:
            content = chunk.get("content", "")
            document_id = next(
                (
                    document["document_id"]
                    for document in documents
                    if document_text(document) in content or document["text"] in content
                ),
                None,
            )
        if document_id is not None and document_id not in ranking:
            ranking.append(document_id)
    return ranking


async def main() -> None:
    args = parse_args()
    llm_client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=args.llm_base_url)
    embedding_client = AsyncOpenAI(
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url=args.embedding_base_url,
    )

    async def embed(texts: list[str]) -> np.ndarray:
        response = await embedding_client.embeddings.create(
            model=args.embedding_model, input=texts, encoding_format="float"
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        for key in (
            "hashing_kv",
            "keyword_extraction",
            "response_format",
            "stream",
            "_priority",
            "cache_context",
            "role",
        ):
            kwargs.pop(key, None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": args.llm_model,
            "messages": messages,
            "temperature": 0,
            **kwargs,
        }
        if keyword_extraction:
            request["response_format"] = {"type": "json_object"}
        response = await llm_client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    embedding = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        model_name=args.embedding_model,
        func=embed,
    )
    examples = json.loads(args.input.read_text(encoding="utf-8"))[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            json.loads(line)["example_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    with args.output.open("a", encoding="utf-8") as output:
        for index, example in enumerate(examples, 1):
            if example["example_id"] in completed:
                continue
            started = time.monotonic()
            record = {
                "example_id": example["example_id"],
                "framework": "lightrag",
                "ranking": [],
                "elapsed_seconds": None,
                "error": None,
            }
            rag = None
            try:
                rag = LightRAG(
                    working_dir=str(args.work_dir / example["example_id"]),
                    llm_model_func=complete,
                    embedding_func=embedding,
                    entity_extract_max_gleaning=0,
                    llm_model_max_async=16,
                    embedding_func_max_async=8,
                    top_k=60,
                    chunk_top_k=10,
                )
                await rag.initialize_storages()
                await rag.ainsert(
                    [document_text(document) for document in example["documents"]],
                    ids=[document["document_id"] for document in example["documents"]],
                    file_paths=[document["document_id"] for document in example["documents"]],
                )
                result = await rag.aquery_data(
                    example["question"],
                    QueryParam(
                        mode="hybrid",
                        top_k=60,
                        chunk_top_k=10,
                        max_total_tokens=28000,
                    ),
                )
                record["ranking"] = chunk_ranking(result, example["documents"])
                record["query_status"] = result.get("status")
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            finally:
                if rag is not None:
                    await rag.finalize_storages()
            record["elapsed_seconds"] = time.monotonic() - started
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[{index}/{len(examples)}] {example['example_id']} "
                f"docs={len(record['ranking'])} error={record['error']}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
