from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from openai import AsyncOpenAI


PROMPT_PROTOCOL = "unified_concise_deepseek_v1"
PROMPT_TEMPLATE = (
    "Answer concisely using only the evidence.\n\n"
    "Question:\n{question}\n\nEvidence:\n{context}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank native candidates and generate one aligned answer."
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument(
        "--reranker-url",
        default="http://127.0.0.1:18081/rerank",
    )
    parser.add_argument("--context-k", type=int, default=12)
    parser.add_argument("--context-token-budget", type=int, default=4096)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} is not a JSON array")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def document_text(document: dict) -> str:
    return (
        f"[S2RAG-DOCUMENT-ID: {document['document_id']}]\n"
        f"Title: {document.get('title') or document['document_id']}\n"
        f"{document.get('text') or ''}"
    )


async def rerank(
    client,
    endpoint: str,
    query: str,
    documents: list[str],
) -> tuple[list[int], list[float]]:
    if not documents:
        return [], []
    response = await client.post(
        endpoint,
        json={"query": query, "texts": documents, "truncate": True},
    )
    response.raise_for_status()
    payload = response.json()
    results = payload if isinstance(payload, list) else payload.get("results", [])
    ordered = sorted(results, key=lambda item: float(item["score"]), reverse=True)
    return (
        [int(item["index"]) for item in ordered],
        [float(item["score"]) for item in ordered],
    )


def fit_context(documents: list[str], token_budget: int) -> tuple[str, int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
    except (ImportError, ValueError):
        encoding = None
    selected = []
    used = 0
    for document in documents:
        if encoding is None:
            token_count = max(1, (len(document) + 3) // 4)
            remaining = token_budget - used
            if remaining <= 0:
                break
            if token_count > remaining:
                document = document[: remaining * 4]
                token_count = remaining
        else:
            tokens = encoding.encode(document)
            remaining = token_budget - used
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                document = encoding.decode(tokens)
            token_count = len(tokens)
        selected.append(document)
        used += token_count
    return "\n\n".join(selected), used


async def process_row(
    *,
    args: argparse.Namespace,
    row: dict,
    example: dict,
    http_client,
    llm_client,
) -> dict:
    started = time.perf_counter()
    documents_by_id = {
        str(document["document_id"]): document
        for document in example["documents"]
    }
    native_ranking = list(
        dict.fromkeys(str(item) for item in row.get("ranking") or [])
    )
    candidates = [
        document_id
        for document_id in native_ranking
        if document_id in documents_by_id
    ]
    candidate_texts = [
        document_text(documents_by_id[document_id])
        for document_id in candidates
    ]
    order, scores = await rerank(
        http_client,
        args.reranker_url,
        example["question"],
        candidate_texts,
    )
    reranked_ids = [candidates[index] for index in order][: args.context_k]
    reranked_texts = [candidate_texts[index] for index in order][: args.context_k]
    context, context_tokens = fit_context(
        reranked_texts,
        args.context_token_budget,
    )
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    prompt = PROMPT_TEMPLATE.format(
        question=example["question"],
        context=context,
    )
    answer = ""
    error = row.get("error")
    status = row.get("status", "success")
    if status == "success" and context:
        try:
            response = await llm_client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=args.max_answer_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            answer = response.choices[0].message.content or ""
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
    elif status == "success":
        status = "unscorable"
        error = "native retrieval returned no mappable candidates"

    result = dict(row)
    result.update(
        {
            "status": status,
            "error": error,
            "native_ranking": native_ranking,
            "ranking": reranked_ids,
            "reranker_scores": scores[: len(reranked_ids)],
            "answer": answer,
            "generation_protocol": PROMPT_PROTOCOL,
            "generation_trace": {
                "model_id": args.model,
                "temperature": 0,
                "max_tokens": args.max_answer_tokens,
                "prompt_sha256": hashlib.sha256(
                    PROMPT_TEMPLATE.encode("utf-8")
                ).hexdigest(),
                "context_sha256": context_sha256,
                "context_tokens": context_tokens,
                "context_token_budget": args.context_token_budget,
                "context_k": args.context_k,
                "candidate_policy": "native_candidates_only",
            },
            "shared_model_trace": {
                "embedding_model": "BAAI/bge-m3",
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "answer_model": args.model,
                "generation_protocol": PROMPT_PROTOCOL,
            },
            "shared_generation_seconds": time.perf_counter() - started,
        }
    )
    return result


async def main() -> None:
    args = parse_args()
    if min(
        args.expected_count,
        args.context_k,
        args.context_token_budget,
        args.max_answer_tokens,
        args.concurrency,
        args.checkpoint_every,
    ) < 1:
        raise ValueError("all numeric limits must be positive")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    retrieval_rows = read_rows(args.retrieval)
    if len(suite) != args.expected_count or len(retrieval_rows) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} suite/retrieval rows, got "
            f"{len(suite)}/{len(retrieval_rows)}"
        )
    examples = {str(row["example_id"]): row for row in suite}
    retrieval = {str(row["example_id"]): row for row in retrieval_rows}
    if set(examples) != set(retrieval):
        raise ValueError("suite and retrieval example IDs do not match")

    partial = args.output.with_suffix(args.output.suffix + ".partial")
    existing = read_rows(partial) if args.resume and partial.exists() else []
    if not args.resume:
        partial.unlink(missing_ok=True)
    completed = {str(row["example_id"]) for row in existing}

    import httpx

    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=180) as http_client:
        llm_client = AsyncOpenAI(
            api_key="shared_generation",
            base_url=args.llm_base_url,
            default_headers={
                "X-S2RAG-Framework": "shared_generation",
                "X-S2RAG-Prompt-SHA256": hashlib.sha256(
                    PROMPT_TEMPLATE.encode("utf-8")
                ).hexdigest(),
            },
        )

        async def bounded(example_id: str):
            async with semaphore:
                return await process_row(
                    args=args,
                    row=retrieval[example_id],
                    example=examples[example_id],
                    http_client=http_client,
                    llm_client=llm_client,
                )

        tasks = [
            asyncio.create_task(bounded(example_id))
            for example_id in examples
            if example_id not in completed
        ]
        with partial.open("a", encoding="utf-8") as output:
            unsynced = 0
            for task in asyncio.as_completed(tasks):
                row = await task
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                unsynced += 1
                if unsynced >= args.checkpoint_every:
                    os.fsync(output.fileno())
                    unsynced = 0
                print(
                    f"{row['example_id']} candidates={len(row['native_ranking'])} "
                    f"reranked={len(row['ranking'])} status={row['status']}",
                    flush=True,
                )
            os.fsync(output.fileno())
        await llm_client.close()

    rows = read_rows(partial)
    if len(rows) != args.expected_count:
        raise ValueError(f"generated {len(rows)} rows; expected {args.expected_count}")
    order = {example_id: index for index, example_id in enumerate(examples)}
    rows.sort(key=lambda row: order[str(row["example_id"])])
    atomic_write_jsonl(args.output, rows)
    partial.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
