from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

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
    cosine_scores,
    document_text,
    prompt_hash,
    rank_marked_context,
    reset_active_clients,
    set_active_clients,
)


FRAMEWORKS = ("hipporag2", "cograg", "hgrag", "hyperrag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official HippoRAG2, Cog-RAG, HGRAG or Hyper-RAG retrieval."
    )
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:18080/v1",
    )
    parser.add_argument("--global-llm-concurrency", type=int, default=16)
    parser.add_argument("--example-concurrency", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--llm-cache", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hgrag-device", default="cpu")
    parser.add_argument("--cner-max-characters", type=int, default=24000)
    parser.add_argument("--cleanup-completed-workdirs", action="store_true")
    add_strict_suite_args(parser)
    return parser.parse_args()


def framework_prompt_paths(framework: str) -> tuple[str, ...]:
    if framework == "hipporag2":
        return ("src/hipporag/prompts", "src/hipporag/information_extraction")
    if framework == "cograg":
        return ("cograg/prompt.py", "cograg/operate.py")
    if framework == "hyperrag":
        return ("hyperrag/prompt.py", "hyperrag/operate.py")
    return ("src/prompts.py",)


async def run_example(
    args: argparse.Namespace,
    index: int,
    example: dict,
) -> tuple[int, dict, Path]:
    framework = args.framework
    example_dir = args.work_dir / example["example_id"]
    official_prompt_hash = prompt_hash(
        args.official_repo,
        framework_prompt_paths(framework),
    )
    parameters = {
        "top_k": args.top_k,
        "complete_documents": True,
        "corpus_scope": "per_question_candidate_passages",
    }
    if framework == "hgrag":
        parameters.update(
            {
                "beta": 0.5,
                "step": 1,
                "multihot": True,
                "struct_topk1": 5,
                "struct_topk2": 10,
                "cner_max_characters": args.cner_max_characters,
            }
        )
    key = artifact_key(
        framework=framework,
        commit=git_commit(args.official_repo),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        prompt_sha256=official_prompt_hash,
        documents=example["documents"],
        parameters=parameters,
    )
    cache_hit = artifact_is_current(example_dir, key)
    if example_dir.exists() and not cache_hit:
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)

    clients = AlignedClients(
        args,
        framework=framework,
        official_repo=args.official_repo,
        documents=example["documents"],
        prompt_sha256=official_prompt_hash,
    )
    clients_token = set_active_clients(clients)
    record = base_record(example["example_id"], framework, args)
    record["artifact_key"] = key
    record["index_cache_hit"] = cache_hit
    index_started = time.perf_counter()
    try:
        if framework == "hipporag2":
            ranking, scores = await run_hipporag2(
                args,
                example,
                example_dir,
                cache_hit,
            )
        elif framework in {"cograg", "hyperrag"}:
            ranking, scores = await run_hyper_family(
                args,
                example,
                example_dir,
                cache_hit,
                clients,
            )
        else:
            ranking, scores = await run_hgrag(
                args,
                example,
                example_dir,
                cache_hit,
                clients,
            )
        if not cache_hit:
            mark_artifact(example_dir, key)
        record["index_seconds"] = time.perf_counter() - index_started
        record["retrieval_seconds"] = scores.pop("retrieval_seconds")
        record["ranking"] = ranking
        record["ranking_scores"] = scores.pop("values")
        record.update(scores)
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        record["index_seconds"] = time.perf_counter() - index_started
    finally:
        await clients.close()
        reset_active_clients(clients_token)
    return index, record, example_dir


async def run_hipporag2(
    args: argparse.Namespace,
    example: dict,
    example_dir: Path,
    cache_hit: bool,
) -> tuple[list[str], dict]:
    sys.path.insert(0, str((args.official_repo / "src").resolve()))
    from hipporag import HippoRAG
    from hipporag.utils.misc_utils import Chunk

    previous_openai_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "hipporag2"
    try:
        rag = HippoRAG(
            save_dir=str(example_dir),
            llm_model_name=args.llm_model,
            llm_base_url=args.llm_base_url,
            embedding_model_name="text-embedding-bge-m3",
            embedding_base_url=args.embedding_base_url,
        )
        chunks = [
            Chunk(
                content=document_text(document),
                source_id=str(document["document_id"]),
                metadata={"document_id": str(document["document_id"])},
            )
            for document in example["documents"]
        ]
        if not cache_hit:
            await asyncio.to_thread(rag.index, chunks)
        retrieval_started = time.perf_counter()
        solutions = await asyncio.to_thread(
            rag.retrieve,
            [example["question"]],
            args.top_k,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
    finally:
        if previous_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_openai_key

    solution = solutions[0]
    ranking = rank_marked_context("\n".join(solution.docs), example["documents"])
    if not ranking and solution.doc_metadata:
        valid = {str(item["document_id"]) for item in example["documents"]}
        ranking = [
            str(item.get("document_id") or item.get("source_id"))
            for item in solution.doc_metadata
            if str(item.get("document_id") or item.get("source_id")) in valid
        ]
    values = (
        [float(value) for value in solution.doc_scores[: len(ranking)]]
        if solution.doc_scores is not None
        else []
    )
    return ranking, {
        "values": values,
        "retrieval_seconds": retrieval_seconds,
        "native_api": "HippoRAG.index+HippoRAG.retrieve",
    }


async def run_hyper_family(
    args: argparse.Namespace,
    example: dict,
    example_dir: Path,
    cache_hit: bool,
    clients: AlignedClients,
) -> tuple[list[str], dict]:
    return await asyncio.to_thread(
        run_hyper_family_sync,
        vars(args),
        example,
        str(example_dir),
        cache_hit,
    )


def run_hyper_family_sync(
    args_values: dict,
    example: dict,
    example_dir_text: str,
    cache_hit: bool,
) -> tuple[list[str], dict]:
    args = argparse.Namespace(**args_values)
    example_dir = Path(example_dir_text)
    official_prompt_hash = prompt_hash(
        args.official_repo,
        framework_prompt_paths(args.framework),
    )
    clients = AlignedClients(
        args,
        framework=args.framework,
        official_repo=args.official_repo,
        documents=example["documents"],
        prompt_sha256=official_prompt_hash,
    )
    clients_token = set_active_clients(clients)
    sys.path.insert(0, str(args.official_repo.resolve()))
    try:
        if args.framework == "cograg":
            from cograg import CogRAG as RAG
            from cograg import QueryParam
            from cograg.utils import EmbeddingFunc

            query_mode = "cog"
            native_api = "CogRAG.insert+CogRAG.query"
        else:
            from hyperrag import HyperRAG as RAG
            from hyperrag.base import QueryParam
            from hyperrag.utils import EmbeddingFunc

            query_mode = "hyper-query"
            native_api = "HyperRAG.insert+HyperRAG.query"

        embedding = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=active_embed,
        )
        rag = RAG(
            working_dir=str(example_dir),
            embedding_func=embedding,
            llm_model_func=active_complete,
            llm_model_name=args.llm_model,
            llm_model_max_async=max(1, args.global_llm_concurrency),
            embedding_func_max_async=8,
            entity_extract_max_gleaning=0,
            enable_llm_cache=True,
        )
        if not cache_hit:
            rag.insert(
                [document_text(document) for document in example["documents"]]
            )
        query_param = QueryParam(
            mode=query_mode,
            only_need_context=True,
            top_k=args.top_k,
            max_token_for_text_unit=12000,
            max_token_for_entity_context=8000,
            max_token_for_relation_context=8000,
        )
        retrieval_started = time.perf_counter()
        context = rag.query(example["question"], query_param)
        retrieval_seconds = time.perf_counter() - retrieval_started
        context = (
            context
            if isinstance(context, str)
            else json.dumps(context, ensure_ascii=False)
        )
        ranking = rank_marked_context(context, example["documents"])
        return ranking, {
            "values": [],
            "retrieval_seconds": retrieval_seconds,
            "native_context_sha256": _sha256_text(context),
            "native_api": native_api,
        }
    finally:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(clients.close())
        reset_active_clients(clients_token)


async def run_hgrag(
    args: argparse.Namespace,
    example: dict,
    example_dir: Path,
    cache_hit: bool,
    clients: AlignedClients,
) -> tuple[list[str], dict]:
    sys.path.insert(0, str(args.official_repo.resolve()))
    from src.modules.hgraph import HG
    from src.prompts import add_prompt

    entity_csv = example_dir / "entity_document.csv"
    if not cache_hit:
        rows = []
        for did, document in enumerate(example["documents"]):
            entities: list[str] = []
            for chunk in _complete_document_batches(
                document_text(document),
                args.cner_max_characters,
            ):
                payload = await clients.chat_json(
                    add_prompt("CNER", chunk),
                    max_tokens=1000,
                )
                entities.extend(_entities(payload))
            for entity in dict.fromkeys(item.strip() for item in entities if item.strip()):
                rows.append((entity, did))
        if not rows:
            raise ValueError("official HGRAG CNER extracted no entities")
        with entity_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("ent", "did"))
            writer.writerows(rows)
    if not entity_csv.exists():
        raise FileNotFoundError(f"missing cached HGRAG entity table: {entity_csv}")

    graph = HG(str(entity_csv), device=args.hgrag_device)
    query_payload = await clients.chat_json(
        add_prompt("QNER", example["question"]),
        max_tokens=500,
    )
    query_entities = _entities(query_payload)
    if not query_entities:
        raise ValueError("official HGRAG QNER extracted no query entities")

    graph_entities = [graph.id2ent[index] for index in range(graph.ent_num)]
    entity_vectors = await clients.embed(graph_entities)
    query_entity_vectors = await clients.embed(query_entities)
    entity_ids: list[int] = []
    entity_similarities: list[float] = []
    for query_vector in query_entity_vectors:
        similarities = cosine_scores(query_vector, entity_vectors)
        best = int(np.argmax(similarities))
        entity_ids.append(best)
        entity_similarities.append(float(similarities[best]))

    documents = [document_text(item) for item in example["documents"]]
    document_vectors = await clients.embed(documents)
    question_vector = (await clients.embed([example["question"]]))[0]
    document_similarities = cosine_scores(question_vector, document_vectors).tolist()

    retrieval_started = time.perf_counter()
    dids, values = graph.hg_diffusion(
        entity_ids,
        entity_similarities,
        document_similarities,
        beta=0.5,
        step=1,
        multihot=True,
    )
    topk2 = min(args.top_k, len(example["documents"]))
    topk1 = min(5, max(1, topk2 - 1))
    if topk2 > topk1:
        enhanced = graph.struct_enhance(
            {"query": [dids, values]},
            topk1=topk1,
            topk2=topk2,
        )["query"]
        dids, values = enhanced
    else:
        dids, values = dids[:topk2], values[:topk2]
    retrieval_seconds = time.perf_counter() - retrieval_started
    ranking = [
        str(example["documents"][int(did)]["document_id"])
        for did in dids
    ]
    return ranking, {
        "values": [float(value) for value in values],
        "retrieval_seconds": retrieval_seconds,
        "native_api": "HGRAG.HG.hg_diffusion+HGRAG.HG.struct_enhance",
        "query_entities": query_entities,
    }


def _complete_document_batches(text: str, max_characters: int) -> list[str]:
    if len(text) <= max_characters:
        return [text]
    batches = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_characters)
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary <= start:
                boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        batches.append(text[start:end])
        start = end
    return batches


def _entities(payload: dict) -> list[str]:
    values = payload.get("entities", payload.get("named_entities", []))
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item).strip()]


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def main() -> None:
    args = parse_args()
    if min(
        args.example_concurrency,
        args.global_llm_concurrency,
        args.checkpoint_every,
        args.top_k,
        args.cner_max_characters,
    ) < 1:
        raise ValueError("concurrency, checkpoint, top-k and CNER limits must be positive")
    examples = load_strict_rows(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    system = f"official:{args.framework}"
    partial, existing = load_resume_rows(args.output, args=args, system=system)
    completed = {row["example_id"] for row in existing}
    successful = {
        row["example_id"] for row in existing if row.get("status") == "success"
    }
    if args.cleanup_completed_workdirs:
        for example_id in successful:
            shutil.rmtree(args.work_dir / example_id, ignore_errors=True)

    semaphore = asyncio.Semaphore(args.example_concurrency)

    async def bounded(index: int, example: dict):
        async with semaphore:
            return await run_example(args, index, example)

    tasks = [
        asyncio.create_task(bounded(index, example))
        for index, example in enumerate(examples, 1)
        if example["example_id"] not in completed
    ]
    written_since_checkpoint = 0
    with partial.open("a", encoding="utf-8") as output:
        for task in asyncio.as_completed(tasks):
            index, record, example_dir = await task
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            written_since_checkpoint += 1
            if written_since_checkpoint >= args.checkpoint_every:
                os.fsync(output.fileno())
                written_since_checkpoint = 0
            if args.cleanup_completed_workdirs and record["status"] == "success":
                shutil.rmtree(example_dir, ignore_errors=True)
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
    asyncio.run(main())
