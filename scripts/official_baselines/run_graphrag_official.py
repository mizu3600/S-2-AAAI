from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import multiprocessing
import os
import re
import signal
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from common import (
    add_strict_suite_args,
    artifact_is_current,
    artifact_key,
    base_record,
    finalize,
    git_commit,
    is_retryable_error,
    load_resume_rows,
    load_strict_rows,
    mark_artifact,
)
from graphrag.api.index import build_index
from graphrag.cli.initialize import initialize_project_at
from graphrag.config.load_config import load_config
from runtime import document_text, prompt_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:18080/v1",
    )
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--example-concurrency", type=int, default=2)
    parser.add_argument("--internal-concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-example-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    add_strict_suite_args(parser)
    return parser.parse_args()


def configure(
    root: Path,
    embedding_base_url: str,
    cache_dir: Path,
    internal_concurrency: int,
    llm_model: str,
    llm_base_url: str,
    llm_api_key_env: str,
) -> None:
    initialize_project_at(
        root,
        force=True,
        model=llm_model,
        embedding_model="BAAI/bge-m3",
    )
    path = root / "settings.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "api_key: ${GRAPHRAG_API_KEY}",
        (
            f"api_key: ${{{llm_api_key_env}}}\n"
            f"    api_base: {llm_base_url}\n"
            "    call_args:\n"
            "      timeout: 60"
        ),
        1,
    )
    text = text.replace(
        "    retry:\n      type: exponential_backoff",
        (
            "    retry:\n"
            "      type: exponential_backoff\n"
            "      max_retries: 2\n"
            "      base_delay: 2.0\n"
            "      max_delay: 15.0"
        ),
        1,
    )
    text = text.replace(
        "api_key: ${GRAPHRAG_API_KEY}",
        f"api_key: local-tei\n    api_base: {embedding_base_url}",
        1,
    )
    text = text.replace("max_gleanings: 1", "max_gleanings: 0", 1)
    text = text.replace(
        "  db_uri: output/lancedb", "  db_uri: output/lancedb\n  vector_size: 1024"
    )
    text = text.replace(
        '    base_dir: "cache"',
        f'    base_dir: "{cache_dir.as_posix()}"',
        1,
    )
    text = f"concurrent_requests: {internal_concurrency}\n" + text
    path.write_text(text, encoding="utf-8")


def enable_deepseek_json_object_compat(llm_base_url: str) -> None:
    import litellm

    if getattr(litellm.acompletion, "_qmshe_deepseek_json_compat", False):
        return

    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    def normalize_response_format(kwargs: dict) -> None:
        response_format = kwargs.get("response_format")
        if (
            isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            kwargs["response_format"] = {"type": "json_object"}

    def completion(*args, **kwargs):
        normalize_response_format(kwargs)
        return original_completion(*args, **kwargs)

    async def acompletion(*args, **kwargs):
        normalize_response_format(kwargs)
        return await original_acompletion(*args, **kwargs)

    completion._qmshe_deepseek_json_compat = True
    acompletion._qmshe_deepseek_json_compat = True
    litellm.completion = completion
    litellm.acompletion = acompletion


def load_outputs(root: Path) -> dict[str, pd.DataFrame | None]:
    output = root / "output"
    required = (
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    )
    result = {name: pd.read_parquet(output / f"{name}.parquet") for name in required}
    covariates = output / "covariates.parquet"
    result["covariates"] = pd.read_parquet(covariates) if covariates.exists() else None
    return result


def source_ranking(context: dict, documents: list[dict]) -> list[str]:
    sources = context.get("sources")
    if sources is None:
        return []
    rows = sources.to_dict("records") if hasattr(sources, "to_dict") else sources
    ranking = []
    for row in rows:
        content = str(row.get("text") or row.get("content") or "")
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


def build_native_context(
    *,
    config,
    frames: dict[str, pd.DataFrame | None],
    community_level: int,
    query: str,
) -> dict:
    """Use GraphRAG's official context builder without native answer generation."""
    query_api = importlib.import_module("graphrag.api.query")
    description_embedding_store = query_api.get_embedding_store(
        config=config.vector_store,
        embedding_name=query_api.entity_description_embedding,
    )
    communities = frames["communities"]
    entities = query_api.read_indexer_entities(
        frames["entities"],
        communities,
        community_level,
    )
    covariates = (
        query_api.read_indexer_covariates(frames["covariates"])
        if frames["covariates"] is not None
        else []
    )
    search_engine = query_api.get_local_search_engine(
        config=config,
        reports=query_api.read_indexer_reports(
            frames["community_reports"],
            communities,
            community_level,
        ),
        text_units=query_api.read_indexer_text_units(frames["text_units"]),
        entities=entities,
        relationships=query_api.read_indexer_relationships(frames["relationships"]),
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type="Single Paragraph",
        system_prompt=query_api.load_search_prompt(config.local_search.prompt),
        callbacks=[],
    )
    result = search_engine.context_builder.build_context(
        query=query,
        **search_engine.context_builder_params,
    )
    return result.context_records


async def main() -> None:
    args = parse_args()
    if (
        min(
            args.example_concurrency,
            args.internal_concurrency,
            args.checkpoint_every,
            args.max_example_attempts,
        )
        < 1
    ):
        raise ValueError("concurrency, checkpoint and attempt values must be positive")
    work_root = args.work_dir.resolve()
    signal.signal(signal.SIGINT, terminate_worker_pool)
    signal.signal(signal.SIGTERM, terminate_worker_pool)
    examples = load_strict_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    system = "official:graphrag"
    partial, existing = load_resume_rows(args.output, args=args, system=system)
    completed = {row["example_id"] for row in existing}
    pending = [
        (index, example)
        for index, example in enumerate(examples, 1)
        if example["example_id"] not in completed
    ]

    with partial.open("a", encoding="utf-8") as output:
        unsynced = 0
        loop = asyncio.get_running_loop()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.example_concurrency,
            mp_context=context,
        ) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    run_example_sync,
                    index,
                    example,
                    args,
                    work_root,
                )
                for index, example in pending
            ]
            for future in asyncio.as_completed(futures):
                index, record = await future
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[{index}/{len(examples)}] {record['example_id']} "
                    f"docs={len(record['ranking'])} error={record['error']}",
                    flush=True,
                )
                unsynced += 1
                output.flush()
                if unsynced < args.checkpoint_every:
                    continue
                os.fsync(output.fileno())
                unsynced = 0
        if unsynced:
            os.fsync(output.fileno())
    finalize(
        output=args.output,
        partial=partial,
        args=args,
        system=system,
        expected_ids={example["example_id"] for example in examples},
    )


def terminate_worker_pool(signum: int, _frame: object) -> None:
    children = multiprocessing.active_children()
    if children:
        print(
            f"signal={signum}; terminating {len(children)} GraphRAG workers",
            flush=True,
        )
    for child in children:
        child.terminate()
    raise SystemExit(128 + signum)


def run_example_sync(
    index: int,
    example: dict,
    args: argparse.Namespace,
    work_root: Path,
) -> tuple[int, dict]:
    return asyncio.run(
        run_example(
            index=index,
            example=example,
            args=args,
            work_root=work_root,
            cache_dir=work_root / example["example_id"] / "cache",
        )
    )


async def run_example(
    *,
    index: int,
    example: dict,
    args: argparse.Namespace,
    work_root: Path,
    cache_dir: Path,
) -> tuple[int, dict]:
    for attempt in range(1, args.max_example_attempts + 1):
        record = base_record(example["example_id"], "graphrag", args)
        index_started = time.perf_counter()
        try:
            root = work_root / example["example_id"]
            official_prompt_hash = prompt_hash(
                args.official_repo,
                ("graphrag/prompts",),
            )
            key = artifact_key(
                framework="graphrag",
                commit=git_commit(args.official_repo),
                llm_model=args.llm_model,
                embedding_model="BAAI/bge-m3",
                prompt_sha256=official_prompt_hash,
                documents=example["documents"],
                parameters={
                    "method": "standard",
                    "max_gleanings": 0,
                    "community_level": 2,
                },
            )
            cache_hit = artifact_is_current(root, key)
            if root.exists() and not cache_hit:
                shutil.rmtree(root)
            enable_deepseek_json_object_compat(args.llm_base_url)
            if not cache_hit:
                configure(
                    root,
                    args.embedding_base_url,
                    cache_dir,
                    args.internal_concurrency,
                    args.llm_model,
                    args.llm_base_url,
                    args.llm_api_key_env,
                )
            config = load_config(root)
            if not cache_hit:
                documents = pd.DataFrame(
                    [
                        {
                            "id": document["document_id"],
                            "title": document["title"],
                            "text": document_text(document),
                            "creation_date": "2026-07-22",
                        }
                        for document in example["documents"]
                    ]
                )
                results = await build_index(
                    config=config,
                    method="standard",
                    input_documents=documents,
                )
                errors = [
                    f"{item.workflow}: {item.error}"
                    for item in results
                    if item.error is not None
                ]
                if errors:
                    raise RuntimeError("; ".join(errors))
                mark_artifact(root, key)
            frames = load_outputs(root)
            record["index_seconds"] = time.perf_counter() - index_started
            record["index_cache_hit"] = cache_hit
            record["artifact_key"] = key
            retrieval_started = time.perf_counter()
            context = build_native_context(
                config=config,
                frames=frames,
                community_level=2,
                query=example["question"],
            )
            record["retrieval_seconds"] = time.perf_counter() - retrieval_started
            record["ranking"] = source_ranking(context, example["documents"])
            record["context_tables"] = sorted(context)
            return index, record
        except Exception as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            if record["index_seconds"] is None:
                record["index_seconds"] = time.perf_counter() - index_started
            if (
                is_retryable_error(record["error"])
                and attempt < args.max_example_attempts
            ):
                delay = retry_delay(
                    record["error"], args.retry_backoff_seconds * attempt
                )
                print(
                    f"[{index}/retry] {example['example_id']} attempt={attempt} "
                    f"sleep={delay:.1f}s error={record['error']}",
                    flush=True,
                )
                await asyncio.sleep(delay)
                continue
            return index, record
    raise AssertionError("example retry loop exited unexpectedly")


def retry_delay(error: str, fallback: float) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error, flags=re.IGNORECASE)
    if match is None:
        return fallback
    return max(fallback, float(match.group(1)) + 2.0)


if __name__ == "__main__":
    asyncio.run(main())
