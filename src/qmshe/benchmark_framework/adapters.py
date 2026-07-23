from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from qmshe.benchmark_framework.dataset import induce_fact_ranking
from qmshe.benchmark_framework.schemas import (
    CanonicalExample,
    StandardTrace,
    TimingTrace,
    UsageTrace,
)


def load_internal_text_unit_traces(
    path: str | Path,
    examples: dict[str, CanonicalExample],
) -> list[StandardTrace]:
    """Load a fresh PathRAG/HyperGraphRAG native text-unit capture.

    These records are produced without modifying either upstream repository.  The
    adapter only translates upstream text units to the benchmark's canonical IDs.
    """
    source = Path(path)
    if not source.exists():
        return []
    output = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("example_id") not in examples:
            continue
        example = examples[row["example_id"]]
        documents = list(dict.fromkeys(row.get("document_ranking", [])))
        facts = list(dict.fromkeys(row.get("fact_ranking", [])))
        # Keep a complete canonical ranking even when an upstream text unit only
        # maps to a passage and not to an exact supporting sentence.
        induced = induce_fact_ranking(example, documents)
        facts.extend(fact_id for fact_id in induced if fact_id not in facts)
        raw_usage = row.get("usage") or {}
        prompt = raw_usage.get("prompt_tokens")
        completion = raw_usage.get("completion_tokens")
        # Query-time DeepSeek usage is measured. Index-build usage belongs to the
        # pre-existing official index and is intentionally reported unavailable.
        cost = None
        if prompt is not None or completion is not None:
            cost = (prompt or 0) * 0.14 / 1_000_000 + (completion or 0) * 0.28 / 1_000_000
        output.append(
            StandardTrace(
                system=row["system"],
                example_id=example.example_id,
                status=row.get("status", "success"),
                document_ranking=documents,
                fact_ranking=facts,
                induced_path=documents[:2],
                ranking_origin=row.get("ranking_origin", "official_internal_text_unit_function"),
                path_origin="induced_from_native_text_unit_ranking",
                usage=UsageTrace(
                    llm_calls=raw_usage.get("llm_calls", 0),
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    embedding_calls=raw_usage.get("embedding_calls", 0),
                    embedding_tokens=raw_usage.get("embedding_tokens"),
                    api_cost_usd=cost,
                    token_count_mode="query_measured_index_unavailable",
                ),
                timing=TimingTrace(
                    retrieval_seconds=row.get("retrieval_seconds"),
                    total_seconds=row.get("retrieval_seconds"),
                    timing_scope="query_only_existing_index",
                ),
                error=row.get("error"),
                metadata={
                    "fact_ranking_origin": "native_text_unit_then_document_induced",
                    "native_text_unit_count": len(row.get("text_unit_ranking", [])),
                },
            )
        )
    return output


def load_official_traces(
    path: str | Path, examples: dict[str, CanonicalExample]
) -> list[StandardTrace]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    output = []
    structured = {"graphrag", "lightrag"}
    for row in records:
        if row["example_id"] not in examples:
            continue
        example = examples[row["example_id"]]
        ranking = list(dict.fromkeys(row.get("ranking", [])))
        framework = row["framework"]
        output.append(
            StandardTrace(
                system=f"official:{framework}",
                example_id=example.example_id,
                status="success" if row.get("error") is None else "error",
                document_ranking=ranking,
                fact_ranking=induce_fact_ranking(example, ranking),
                induced_path=ranking[:2],
                ranking_origin=(
                    "official_internal_structured"
                    if framework in structured
                    else "legacy_rendered_context_order"
                ),
                path_origin="induced_from_document_ranking",
                usage=UsageTrace(
                    prompt_tokens=None,
                    completion_tokens=None,
                    embedding_tokens=None,
                    api_cost_usd=None,
                    token_count_mode="unavailable_legacy_run",
                ),
                timing=TimingTrace(
                    total_seconds=row.get("elapsed_seconds"), timing_scope="index_plus_query"
                ),
                error=row.get("error"),
                metadata={"fact_ranking_origin": "document_induced"},
            )
        )
    return output


def load_qmsxe_passage_traces(
    path: str | Path,
    examples: dict[str, CanonicalExample],
    seed: int = 42,
) -> list[StandardTrace]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    output = []
    for row in records:
        if row.get("seed") != seed or row["example_id"] not in examples:
            continue
        example = examples[row["example_id"]]
        ranking = list(dict.fromkeys(row["ranking"]))
        output.append(
            StandardTrace(
                system=row["framework"].replace("qmsxe:", "qmsxe:"),
                example_id=example.example_id,
                status="success",
                document_ranking=ranking,
                fact_ranking=induce_fact_ranking(example, ranking),
                induced_path=ranking[:2],
                ranking_origin="qmsxe_internal_fact_to_document",
                path_origin="induced_from_document_ranking",
                usage=UsageTrace(token_count_mode="local_models_no_api"),
                timing=TimingTrace(timing_scope="unavailable_legacy_run"),
                metadata={"seed": seed, "fact_ranking_origin": "document_induced"},
            )
        )
    return output


def load_qmsxe_parity_traces(
    path: str | Path,
    examples: dict[str, CanonicalExample],
    seed: int = 42,
) -> list[StandardTrace]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    output = []
    suffix = f"seed_{seed}"
    for row in records:
        if row["example_id"] not in examples or not row["system"].endswith(suffix):
            continue
        documents = list(dict.fromkeys(row.get("canonical_document_ranking", [])))
        valid_fact_ids = {fact.fact_id for fact in examples[row["example_id"]].facts}
        facts = [
            fact_id
            for fact_id in dict.fromkeys(row.get("canonical_fact_ranking", []))
            if fact_id in valid_fact_ids
        ]
        if not documents or not facts:
            continue
        output.append(
            StandardTrace(
                system=row["system"]
                .replace("trained_parity:", "qmsxe:")
                .removesuffix(f":seed_{seed}"),
                example_id=row["example_id"],
                status="success",
                document_ranking=documents,
                fact_ranking=facts,
                induced_path=documents[:2],
                ranking_origin="qmsxe_internal_fact_scores",
                path_origin="induced_from_document_ranking",
                usage=UsageTrace(token_count_mode="local_models_no_api"),
                timing=TimingTrace(
                    total_seconds=row.get("elapsed_seconds"), timing_scope="index_plus_query"
                ),
                metadata={"seed": seed, "fact_budget": row.get("fact_budget")},
            )
        )
    return output


def bm25_trace(example: CanonicalExample) -> StandardTrace:
    started = time.perf_counter()
    tokenized = [document.text.casefold().split() for document in example.documents]
    scores = BM25Okapi(tokenized).get_scores(example.question.casefold().split())
    ranking = [
        example.documents[index].document_id
        for index in np.argsort(-np.asarray(scores), kind="stable")
    ]
    elapsed = time.perf_counter() - started
    return StandardTrace(
        system="baseline:bm25",
        example_id=example.example_id,
        status="success",
        document_ranking=ranking,
        fact_ranking=induce_fact_ranking(example, ranking),
        induced_path=ranking[:2],
        ranking_origin="native_bm25_document_scores",
        path_origin="induced_from_document_ranking",
        usage=UsageTrace(token_count_mode="no_model_tokens"),
        timing=TimingTrace(retrieval_seconds=elapsed, total_seconds=elapsed),
        metadata={"fact_ranking_origin": "document_induced"},
    )


class SiliconFlowDenseAdapter:
    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
    ):
        self.model = model
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {os.environ['SILICONFLOW_API_KEY']}"},
            timeout=120,
        )

    def rank(self, example: CanonicalExample) -> StandardTrace:
        started = time.perf_counter()
        texts = [example.question, *(f"{item.title}: {item.text}" for item in example.documents)]
        response = self.client.post("/embeddings", json={"model": self.model, "input": texts})
        response.raise_for_status()
        payload = response.json()
        vectors = np.asarray(
            [item["embedding"] for item in sorted(payload["data"], key=lambda row: row["index"])],
            dtype=np.float32,
        )
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        scores = vectors[1:] @ vectors[0]
        ranking = [
            example.documents[index].document_id for index in np.argsort(-scores, kind="stable")
        ]
        elapsed = time.perf_counter() - started
        usage = payload.get("usage") or {}
        embedding_tokens = usage.get("total_tokens", usage.get("prompt_tokens"))
        return StandardTrace(
            system="baseline:dense_bge_m3",
            example_id=example.example_id,
            status="success",
            document_ranking=ranking,
            fact_ranking=induce_fact_ranking(example, ranking),
            induced_path=ranking[:2],
            ranking_origin="native_dense_document_scores",
            path_origin="induced_from_document_ranking",
            usage=UsageTrace(
                embedding_calls=1,
                embedding_tokens=embedding_tokens,
                token_count_mode="measured" if embedding_tokens is not None else "provider_omitted",
            ),
            timing=TimingTrace(retrieval_seconds=elapsed, total_seconds=elapsed),
            metadata={"embedding_model": self.model, "fact_ranking_origin": "document_induced"},
        )

    def close(self) -> None:
        self.client.close()
