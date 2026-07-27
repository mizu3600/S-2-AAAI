from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import torch

from s2rag.embedding.graph_encoder import GraphSpectralSemanticEncoder
from s2rag.embedding.text_encoder import LocalBGEEncoder, encode_documents, encode_queries
from s2rag.generation.generator import EvidenceGenerator
from s2rag.graph.ordinary import build_reified_fact_graph
from s2rag.ingest.schemas import Corpus, EvidenceHyperedge
from s2rag.observability import RuntimeMetrics
from s2rag.retrieval.ann_retriever import ExactVectorIndex, SearchHit
from s2rag.retrieval.candidates import (
    ScoredFact,
    rank_scored_facts,
    rerank_scored_facts,
)
from s2rag.retrieval.context_builder import build_context
from s2rag.retrieval.evidence_verifier import verify_candidates
from s2rag.retrieval.graph_reranker import graph_rerank
from s2rag.retrieval.local_reranker import LocalBGEReranker
from s2rag.retrieval.seed_retriever import BM25Retriever, reciprocal_rank_fusion


def verbalize_fact(fact: EvidenceHyperedge, entity_names: dict[str, str]) -> str:
    arguments = ", ".join(
        f"{argument.role}={entity_names.get(argument.entity_id, argument.entity_id)}"
        for argument in fact.arguments
    )
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in fact.qualifiers.items() if value is not None
    )
    return f"{fact.predicate}: {arguments}" + (f"; {qualifiers}" if qualifiers else "")


@dataclass
class QueryResult:
    answer: str
    citations: list[dict]
    retrieved_nodes: list[str]
    retrieved_facts: list[str]
    band_weights: dict[str, float]
    evidence_path: list[str]
    rejected_candidates: dict[str, str]
    scores: list[dict]


@dataclass(frozen=True)
class FactCandidateResult:
    facts: list[ScoredFact]
    node_ids: list[str]
    band_weights: dict[str, float]
    channel_candidate_counts: dict[str, int]


class S2RAGPipeline:
    """The sole production pipeline: Reified-Fact graph, hybrid retrieval and generation."""

    def __init__(
        self,
        corpus: Corpus,
        text_encoder=None,
        reranker=None,
        seed: int = 42,
        use_local_reranker: bool = True,
        generator=None,
    ):
        if not corpus.entities or not corpus.evidence_hyperedges:
            raise ValueError("corpus must contain entities and evidence facts")
        self.corpus = corpus
        self.text_encoder = text_encoder or LocalBGEEncoder()
        self.reranker = reranker or (LocalBGEReranker() if use_local_reranker else None)
        self.seed = seed
        self.generator = generator or EvidenceGenerator()
        self.metrics = RuntimeMetrics()
        self._build()

    def _build(self) -> None:
        self.artifacts = build_reified_fact_graph(self.corpus)
        self.node_ids = self.artifacts.node_ids
        self.node_texts = self.artifacts.node_texts
        raw_np = encode_documents(self.text_encoder, self.node_texts)
        self.raw_features = torch.tensor(raw_np, dtype=torch.float32)
        self.propagation = _scipy_to_torch_sparse(self.artifacts.propagation)

        self.model = GraphSpectralSemanticEncoder(self.raw_features.shape[1])
        self.model.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.propagation)

        self.graph_index = ExactVectorIndex(self.node_ids, self.node_bands["full"].numpy())
        self.band_indices = {
            name: ExactVectorIndex(self.node_ids, self.node_bands[name].numpy())
            for name in ("raw", "low", "mid", "high")
        }
        self.raw_index = ExactVectorIndex(self.node_ids, raw_np)

        names = {entity.entity_id: entity.canonical_name for entity in self.corpus.entities}
        self.fact_text_by_id = {
            fact.hyperedge_id: verbalize_fact(fact, names)
            for fact in self.corpus.evidence_hyperedges
        }
        self.bm25 = BM25Retriever(list(self.fact_text_by_id), list(self.fact_text_by_id.values()))

    def query(
        self,
        question: str,
        top_k: int = 12,
        return_debug: bool = False,
        candidate_count: int | None = None,
        generate_answer: bool = True,
    ) -> QueryResult:
        started = perf_counter()
        retrieval_count = candidate_count or max(30, top_k * 3)
        if retrieval_count < top_k:
            raise ValueError("candidate_count must be at least top_k")
        candidate_result = self.retrieve_fact_candidates(
            question,
            per_channel_k=retrieval_count,
            candidate_count=retrieval_count,
        )
        scored_facts = candidate_result.facts
        if self.reranker is not None:
            scored_facts = rerank_scored_facts(
                question,
                scored_facts,
                self.fact_text_by_id,
                self.reranker,
            )
        verification = verify_candidates(
            [fact.fact_id for fact in scored_facts],
            self.corpus,
        )
        fact_ids = verification.accepted_ids[:top_k]
        context, citations = build_context(self.corpus, fact_ids, limit=top_k)
        answer = self.generator.generate(question, context) if generate_answer else ""

        result = QueryResult(
            answer=answer,
            citations=citations,
            retrieved_nodes=candidate_result.node_ids[:top_k],
            retrieved_facts=fact_ids,
            band_weights=candidate_result.band_weights,
            evidence_path=self._best_path(candidate_result.node_ids[:8]),
            rejected_candidates=verification.rejected if return_debug else {},
            scores=[
                {
                    "object_id": fact.fact_id,
                    "score": fact.score,
                    "source": fact.source,
                }
                for fact in scored_facts[:top_k]
            ]
            if return_debug
            else [],
        )
        self.metrics.observe(result, (perf_counter() - started) * 1000)
        return result

    def retrieve_fact_candidates(
        self,
        question: str,
        *,
        per_channel_k: int,
        candidate_count: int,
        channels: tuple[str, ...] = (
            "graph_full",
            "graph_multi",
            "dense",
            "bm25",
        ),
        apply_graph_rerank: bool = True,
    ) -> FactCandidateResult:
        if per_channel_k <= 0 or candidate_count <= 0:
            raise ValueError("candidate budgets must be positive")
        query_np = encode_queries(self.text_encoder, [question])[0]
        query_tensor = torch.tensor(query_np, dtype=torch.float32)
        with torch.no_grad():
            query_parts, gate = self.model.encode_query_parts(
                query_tensor,
                self.raw_features,
                self.node_bands,
                top_m=64,
                temperature=0.05,
            )
            query_vector = torch.cat(
                [
                    gate[index] * query_parts[name]
                    for index, name in enumerate(("raw", "low", "mid", "high"))
                ]
            )

        full_hits = self.graph_index.search(query_vector.numpy(), per_channel_k, "graph-full")
        band_hits = {
            name: self.band_indices[name].search(
                query_parts[name].numpy(), per_channel_k, f"graph-{name}"
            )
            for name in ("raw", "low", "mid", "high")
        }
        multi_hits = _weighted_band_fusion(band_hits, gate, per_channel_k)
        raw_hits = self.raw_index.search(query_np, per_channel_k, "raw-text")
        lexical_hits = self.bm25.search(question, per_channel_k)

        channel_hits = {
            "graph_full": full_hits,
            "graph_multi": multi_hits,
            "dense": raw_hits,
            "bm25": lexical_hits,
        }
        unknown_channels = set(channels) - set(channel_hits)
        if unknown_channels:
            raise ValueError(f"unknown retrieval channels: {sorted(unknown_channels)}")
        fused = reciprocal_rank_fusion([channel_hits[name] for name in channels])
        reranked = (
            graph_rerank(fused, self.artifacts.graph)
            if apply_graph_rerank
            else fused
        )
        node_ids = [hit.object_id for hit in reranked if hit.object_id in self.artifacts.graph]
        fact_scores: dict[str, tuple[float, int, str]] = {}
        for hit in reranked:
            fact_ids = []
            if hit.object_id in self.artifacts.fact_by_node:
                fact_ids.append(self.artifacts.fact_by_node[hit.object_id])
            fact_ids.extend(sorted(self.artifacts.facts_by_entity.get(hit.object_id, set())))
            for fact_id in fact_ids:
                candidate = (hit.score, hit.rank, "s2rag_first_stage")
                previous = fact_scores.get(fact_id)
                if (
                    previous is None
                    or candidate[0] > previous[0]
                    or (candidate[0] == previous[0] and candidate[1] < previous[1])
                ):
                    fact_scores[fact_id] = candidate
        return FactCandidateResult(
            facts=rank_scored_facts(fact_scores, candidate_count),
            node_ids=node_ids,
            band_weights={
                name: float(value)
                for name, value in zip(("raw", "low", "mid", "high"), gate, strict=True)
            },
            channel_candidate_counts={
                name: len(channel_hits[name]) for name in channels
            },
        )

    def _facts_from_candidates(self, candidate_ids: list[str]) -> list[str]:
        ranked: list[str] = []
        for candidate_id in candidate_ids:
            if candidate_id in self.artifacts.fact_by_node:
                ranked.append(self.artifacts.fact_by_node[candidate_id])
            ranked.extend(sorted(self.artifacts.facts_by_entity.get(candidate_id, set())))
        return list(dict.fromkeys(ranked))

    def _best_path(self, node_ids: list[str]) -> list[str]:
        for left_index, left in enumerate(node_ids):
            for right in node_ids[left_index + 1 :]:
                try:
                    path = nx.shortest_path(self.artifacts.graph, left, right)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(path) > 1:
                    return path
        return node_ids[:1]


def _weighted_band_fusion(
    band_hits: dict[str, list[SearchHit]], gate: torch.Tensor, top_k: int
) -> list[SearchHit]:
    totals: dict[str, float] = {}
    for band_index, name in enumerate(("raw", "low", "mid", "high")):
        weight = float(gate[band_index])
        for hit in band_hits[name]:
            totals[hit.object_id] = totals.get(hit.object_id, 0.0) + weight / (60 + hit.rank)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SearchHit(object_id, score, rank + 1, "graph-multi")
        for rank, (object_id, score) in enumerate(ranked)
    ]


def _scipy_to_torch_sparse(matrix, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, dtype=dtype, check_invariants=True
    ).coalesce()


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")


def load_corpus(path: str | Path) -> Corpus:
    return Corpus.model_validate_json(Path(path).read_text(encoding="utf-8"))
