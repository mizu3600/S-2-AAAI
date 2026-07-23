from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("pathrag", "hypergraphrag"), required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--max-gleaning", type=int, default=0)
    parser.add_argument("--llm-concurrency", type=int, default=16)
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com")
    parser.add_argument("--embedding-base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY")
    return parser.parse_args()


def load_framework(name: str, repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    if name == "pathrag":
        from PathRAG import PathRAG, QueryParam
        from PathRAG.utils import EmbeddingFunc

        return PathRAG, QueryParam, EmbeddingFunc
    from hypergraphrag import HyperGraphRAG, QueryParam
    from hypergraphrag.utils import EmbeddingFunc

    return HyperGraphRAG, QueryParam, EmbeddingFunc


def document_text(document: dict) -> str:
    return f"{document['title']}: {document['text']}"


def rank_documents(context: str, documents: list[dict]) -> list[str]:
    positions = []
    for document in documents:
        text = document_text(document)
        position = context.find(text)
        if position < 0:
            position = context.find(document["text"])
        if position >= 0:
            positions.append((position, document["document_id"]))
    return [document_id for _, document_id in sorted(positions)]


async def main() -> None:
    args = parse_args()
    llm_key = os.environ[args.llm_api_key_env]
    embedding_key = os.environ["SILICONFLOW_API_KEY"]
    rag_class, query_param_class, embedding_func_class = load_framework(
        args.framework, args.official_repo
    )
    llm_client = AsyncOpenAI(api_key=llm_key, base_url=args.llm_base_url)
    embedding_client = AsyncOpenAI(api_key=embedding_key, base_url=args.embedding_base_url)

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
        kwargs.pop("hashing_kv", None)
        kwargs.pop("response_format", None)
        kwargs.pop("stream", None)
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

    embedding = embedding_func_class(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed,
        concurrent_limit=8,
    )
    examples = json.loads(args.input.read_text(encoding="utf-8"))
    examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["example_id"])

    with args.output.open("a", encoding="utf-8") as output:
        for index, example in enumerate(examples, 1):
            if example["example_id"] in completed:
                continue
            started = time.monotonic()
            example_dir = args.work_dir / example["example_id"]
            record = {
                "example_id": example["example_id"],
                "framework": args.framework,
                "ranking": [],
                "context_sha256": None,
                "elapsed_seconds": None,
                "error": None,
            }
            try:
                rag = rag_class(
                    working_dir=str(example_dir),
                    embedding_func=embedding,
                    llm_model_func=complete,
                    llm_model_name=args.llm_model,
                    llm_model_max_async=args.llm_concurrency,
                    embedding_func_max_async=8,
                    entity_extract_max_gleaning=args.max_gleaning,
                    enable_llm_cache=True,
                )
                await rag.ainsert([document_text(document) for document in example["documents"]])
                query_param = query_param_class(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=args.top_k,
                    max_token_for_text_unit=12000,
                    max_token_for_global_context=8000,
                    max_token_for_local_context=8000,
                )
                context = await rag.aquery(example["question"], query_param)
                if not isinstance(context, str):
                    context = str(context)
                record["ranking"] = rank_documents(context, example["documents"])
                record["context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()
            except Exception as error:  # keep checkpoints for long official runs
                record["error"] = f"{type(error).__name__}: {error}"
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
