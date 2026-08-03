from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
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


def load_framework(name: str, repo: Path):
    sys.path.insert(0, str(repo.resolve()))
    if name == "pathrag":
        from PathRAG import PathRAG, QueryParam
        from PathRAG.utils import EmbeddingFunc

        return PathRAG, QueryParam, EmbeddingFunc
    from hypergraphrag import HyperGraphRAG, QueryParam
    from hypergraphrag.utils import EmbeddingFunc

    return HyperGraphRAG, QueryParam, EmbeddingFunc


def rank_documents(context: str, documents: list[dict]) -> list[str]:
    return rank_marked_context(context, documents)


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
    rag_class, query_param_class, embedding_func_class = load_framework(
        args.framework, args.official_repo
    )
    prompt_paths = (
        ("PathRAG/prompt.py",)
        if args.framework == "pathrag"
        else ("hypergraphrag/prompt.py",)
    )
    official_prompt_hash = prompt_hash(args.official_repo, prompt_paths)
    clients = AlignedClients(
        args,
        framework=args.framework,
        official_repo=args.official_repo,
        documents=example["documents"],
        prompt_sha256=official_prompt_hash,
    )
    clients_token = set_active_clients(clients)

    embedding = embedding_func_class(
        embedding_dim=1024,
        max_token_size=8192,
        func=active_embed,
        concurrent_limit=8,
    )
    index_started = time.perf_counter()
    example_dir = args.work_dir / example["example_id"]
    key = artifact_key(
        framework=args.framework,
        commit=git_commit(args.official_repo),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        prompt_sha256=official_prompt_hash,
        documents=example["documents"],
        parameters={
            "mode": "hybrid",
            "top_k": args.top_k,
            "max_gleaning": args.max_gleaning,
        },
    )
    cache_hit = artifact_is_current(example_dir, key)
    if example_dir.exists() and not cache_hit:
        shutil.rmtree(example_dir)
    record = base_record(example["example_id"], args.framework, args)
    record["context_sha256"] = None
    try:
        rag = rag_class(
            working_dir=str(example_dir),
            embedding_func=embedding,
            llm_model_func=active_complete,
            llm_model_name=args.llm_model,
            llm_model_max_async=min(args.llm_concurrency, per_worker_llm_concurrency),
            embedding_func_max_async=8,
            entity_extract_max_gleaning=args.max_gleaning,
            enable_llm_cache=True,
        )
        if not cache_hit:
            await rag.ainsert(
                [document_text(document) for document in example["documents"]]
            )
            mark_artifact(example_dir, key)
        record["index_seconds"] = time.perf_counter() - index_started
        record["index_cache_hit"] = cache_hit
        record["artifact_key"] = key
        query_param = query_param_class(
            mode="hybrid",
            only_need_context=True,
            top_k=args.top_k,
            max_token_for_text_unit=12000,
            max_token_for_global_context=8000,
            max_token_for_local_context=8000,
        )
        retrieval_started = time.perf_counter()
        context = await rag.aquery(example["question"], query_param)
        record["retrieval_seconds"] = time.perf_counter() - retrieval_started
        if not isinstance(context, str):
            context = str(context)
        record["ranking"] = rank_documents(context, example["documents"])
        record["context_sha256"] = hashlib.sha256(context.encode("utf-8")).hexdigest()
    except Exception as error:  # keep checkpoints for long official runs
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        if record["index_seconds"] is None:
            record["index_seconds"] = time.perf_counter() - index_started
    finally:
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
        or args.llm_concurrency <= 0
        or args.checkpoint_every <= 0
    ):
        raise ValueError("concurrency and checkpoint values must be positive")
    examples = load_strict_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    framework = args.framework
    system = f"official:{framework}"
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
                    record = base_record(example["example_id"], framework, args)
                    record["context_sha256"] = None
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
