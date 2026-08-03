from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import (
    add_strict_suite_args,
    artifact_is_current,
    artifact_key,
    base_record,
    finalize,
    git_commit,
    load_resume_rows,
    load_strict_rows,
    mark_artifact,
)
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from runtime import (
    AlignedClients,
    active_complete,
    active_embed,
    document_text,
    prompt_hash,
    rank_marked_context,
    reset_active_clients,
    set_active_clients,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--example-concurrency", type=int, default=2)
    parser.add_argument("--global-llm-concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--llm-cache", type=Path, required=True)
    parser.add_argument("--cleanup-completed-workdirs", action="store_true")
    add_strict_suite_args(parser)
    return parser.parse_args()


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


def index_ready(path: Path) -> bool:
    return (
        (path / "vdb_entities.json").is_file()
        and (path / "vdb_entities.json").stat().st_size > 100
        and (path / "kv_store_text_chunks.json").is_file()
    )


async def process_example(
    args_values: dict,
    index: int,
    example: dict,
) -> tuple[int, dict, str]:
    args = argparse.Namespace(**args_values)
    per_worker_llm_concurrency = max(
        1,
        args.global_llm_concurrency // args.example_concurrency,
    )
    official_prompt_hash = prompt_hash(args.official_repo, ("lightrag/prompt.py",))
    clients = AlignedClients(
        args,
        framework="lightrag",
        official_repo=args.official_repo,
        documents=example["documents"],
        prompt_sha256=official_prompt_hash,
    )
    clients_token = set_active_clients(clients)

    embedding = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        model_name=args.embedding_model,
        func=active_embed,
    )
    example_dir = args.work_dir / example["example_id"]
    key = artifact_key(
        framework="lightrag",
        commit=git_commit(args.official_repo),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        prompt_sha256=official_prompt_hash,
        documents=example["documents"],
        parameters={"mode": "hybrid", "top_k": 60, "chunk_top_k": 10},
    )
    cache_hit = artifact_is_current(example_dir, key) and index_ready(example_dir)
    if example_dir.exists() and not cache_hit:
        shutil.rmtree(example_dir)
    record = base_record(example["example_id"], "lightrag", args)
    index_started = time.perf_counter()
    rag = None
    try:
        rag = LightRAG(
            working_dir=str(example_dir),
            llm_model_func=active_complete,
            embedding_func=embedding,
            entity_extract_max_gleaning=0,
            llm_model_max_async=per_worker_llm_concurrency,
            embedding_func_max_async=8,
            top_k=60,
            chunk_top_k=10,
        )
        await rag.initialize_storages()
        if not cache_hit:
            await rag.ainsert(
                [document_text(document) for document in example["documents"]],
                ids=[document["document_id"] for document in example["documents"]],
                file_paths=[document["document_id"] for document in example["documents"]],
            )
            if not index_ready(example_dir):
                raise RuntimeError(
                    "LightRAG indexing completed without persisted entities/text chunks"
                )
            mark_artifact(example_dir, key)
        record["index_seconds"] = time.perf_counter() - index_started
        record["index_cache_hit"] = cache_hit
        record["artifact_key"] = key
        retrieval_started = time.perf_counter()
        result = await rag.aquery_data(
            example["question"],
            QueryParam(
                mode="hybrid",
                top_k=60,
                chunk_top_k=10,
                max_total_tokens=28000,
            ),
        )
        record["retrieval_seconds"] = time.perf_counter() - retrieval_started
        record["ranking"] = chunk_ranking(result, example["documents"])
        if not record["ranking"]:
            record["ranking"] = rank_marked_context(
                json.dumps(result, ensure_ascii=False),
                example["documents"],
            )
        record["query_status"] = result.get("status")
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        if record["index_seconds"] is None:
            record["index_seconds"] = time.perf_counter() - index_started
    finally:
        if rag is not None:
            await rag.finalize_storages()
        await clients.close()
        reset_active_clients(clients_token)
    return index, record, str(example_dir)


def process_example_sync(
    args_values: dict,
    index: int,
    example: dict,
) -> tuple[int, dict, str]:
    return asyncio.run(process_example(args_values, index, example))


def main() -> None:
    args = parse_args()
    if (
        args.example_concurrency <= 0
        or args.global_llm_concurrency <= 0
        or args.checkpoint_every <= 0
    ):
        raise ValueError("concurrency and checkpoint values must be positive")
    examples = load_strict_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    system = "official:lightrag"
    partial, existing = load_resume_rows(args.output, args=args, system=system)
    completed = {row["example_id"] for row in existing}
    successful = {
        row["example_id"] for row in existing if row.get("status") == "success"
    }
    if args.cleanup_completed_workdirs:
        for example_id in successful:
            shutil.rmtree(args.work_dir / example_id, ignore_errors=True)

    work = [
        (index, example)
        for index, example in enumerate(examples, 1)
        if example["example_id"] not in completed
    ]
    written_since_checkpoint = 0
    with ProcessPoolExecutor(
        max_workers=args.example_concurrency,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(process_example_sync, vars(args), index, example): (
                index,
                example,
            )
            for index, example in work
        }
        with partial.open("a", encoding="utf-8") as output:
            for future in as_completed(futures):
                expected_index, example = futures[future]
                try:
                    index, record, example_dir_text = future.result()
                except Exception as error:
                    index = expected_index
                    record = base_record(example["example_id"], "lightrag", args)
                    record["status"] = "error"
                    record["error"] = f"{type(error).__name__}: {error}"
                    example_dir_text = str(args.work_dir / example["example_id"])
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                written_since_checkpoint += 1
                if written_since_checkpoint >= args.checkpoint_every:
                    os.fsync(output.fileno())
                    written_since_checkpoint = 0
                if args.cleanup_completed_workdirs and record["status"] == "success":
                    shutil.rmtree(Path(example_dir_text), ignore_errors=True)
                print(
                    f"[{index}/{len(examples)}] {record['example_id']} "
                    f"docs={len(record['ranking'])} error={record['error']}",
                    flush=True,
                )
            os.fsync(output.fileno())
        finalize(
            output=args.output,
            partial=partial,
            args=args,
            system=system,
            expected_ids={example["example_id"] for example in examples},
        )


if __name__ == "__main__":
    main()
