from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import torch

from qmshe.embedding.graph_encoder import GraphSpectralSemanticEncoder
from qmshe.embedding.text_encoder import TextEncoder, encode_documents, encode_queries
from qmshe.generation.generator import EvidenceGenerator
from qmshe.graph.ordinary import build_reified_fact_graph
from qmshe.ingest.schemas import Corpus, EvidenceHyperedge
from qmshe.observability import RuntimeMetrics
from qmshe.providers import ProviderError, SiliconFlowClient
from qmshe.retrieval.ann_retriever import ExactVectorIndex, SearchHit
from qmshe.retrieval.context_builder import build_context
from qmshe.retrieval.evidence_verifier import verify_candidates
from qmshe.retrieval.graph_reranker import graph_rerank
from qmshe.retrieval.seed_retriever import BM25Retriever, reciprocal_rank_fusion


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


class QMSHERAGPipeline:
    """The sole production pipeline: Reified-Fact graph, hybrid retrieval and generation."""

    def __init__(
        self,
        corpus: Corpus,
        text_encoder: TextEncoder | None = None,
        reranker=None,
        seed: int = 42,
        enable_remote_reranker: bool = True,
    ):
        if not corpus.entities or not corpus.evidence_hyperedges:
            raise ValueError("corpus must contain entities and evidence facts")
        self.corpus = corpus
        self.text_encoder = text_encoder or TextEncoder()
        self.reranker = reranker
        self.seed = seed
        self.enable_remote_reranker = enable_remote_reranker
        self.generator = EvidenceGenerator()
        self.metrics = RuntimeMetrics()
        self._build()

    def _build(self) -> None:
        self.artifacts = build_reified_fact_graph(self.corpus)
        self.node_ids = self.artifacts.node_ids
        self.node_texts = self.artifacts.node_texts
        raw_np = encode_documents(self.text_encoder, self.node_texts)
        self.raw_features = torch.tensor(raw_np, dtype=torch.float32)
        self.propagation = _scipy_to_torch_sparse(self.artifacts.propagation)

        torch.manual_seed(self.seed)
        self.model = GraphSpectralSemanticEncoder(
            self.raw_features.shape[1], raw_dim=64, band_dim=32
        )
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
        self.bm25 = BM25Retriever(
            list(self.fact_text_by_id), list(self.fact_text_by_id.values())
        )

    def query(
        self,
        question: str,
        top_k: int = 12,
        return_debug: bool = False,
        candidate_count: int | None = None,
        generate_answer: bool = True,
    ) -> QueryResult:
        started = perf_counter()
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

        retrieval_count = candidate_count or max(30, top_k * 3)
        if retrieval_count < top_k:
            raise ValueError("candidate_count must be at least top_k")

        full_hits = self.graph_index.search(
            query_vector.numpy(), retrieval_count, "graph-full"
        )
        band_hits = {
            name: self.band_indices[name].search(
                query_parts[name].numpy(), retrieval_count, f"graph-{name}"
            )
            for name in ("raw", "low", "mid", "high")
        }
        multi_hits = _weighted_band_fusion(band_hits, gate, retrieval_count)
        raw_hits = self.raw_index.search(query_np, retrieval_count, "raw-text")
        lexical_hits = self.bm25.search(question, retrieval_count)

        fused = reciprocal_rank_fusion([full_hits, multi_hits, raw_hits, lexical_hits])
        reranked = graph_rerank(fused[:50], self.artifacts.graph)
        node_ids = [hit.object_id for hit in reranked if hit.object_id in self.artifacts.graph]

        fact_candidates = self._facts_from_candidates([hit.object_id for hit in reranked])
        fact_candidates = self._remote_rerank(question, fact_candidates)
        verification = verify_candidates(fact_candidates, self.corpus)
        fact_ids = verification.accepted_ids[:top_k]
        context, citations = build_context(self.corpus, fact_ids, limit=top_k)
        answer = self.generator.generate(question, context) if generate_answer else ""

        result = QueryResult(
            answer=answer,
            citations=citations,
            retrieved_nodes=node_ids[:top_k],
            retrieved_facts=fact_ids,
            band_weights={
                name: float(value)
                for name, value in zip(("raw", "low", "mid", "high"), gate, strict=True)
            },
            evidence_path=self._best_path(node_ids[:8]),
            rejected_candidates=verification.rejected if return_debug else {},
            scores=[
                {"object_id": hit.object_id, "score": hit.score, "source": hit.source}
                for hit in reranked[:top_k]
            ]
            if return_debug
            else [],
        )
        self.metrics.observe(result, (perf_counter() - started) * 1000)
        return result

    def _facts_from_candidates(self, candidate_ids: list[str]) -> list[str]:
        ranked: list[str] = []
        for candidate_id in candidate_ids:
            if candidate_id in self.artifacts.fact_by_node:
                ranked.append(self.artifacts.fact_by_node[candidate_id])
            ranked.extend(sorted(self.artifacts.facts_by_entity.get(candidate_id, set())))
        return list(dict.fromkeys(ranked))

    def _remote_rerank(self, question: str, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        documents = [self.fact_text_by_id[item] for item in fact_ids]
        if self.reranker is not None:
            return [fact_ids[index] for index in self.reranker.rank(question, documents)]
        if not self.enable_remote_reranker:
            return fact_ids
        try:
            results = SiliconFlowClient().rerank(
                question, documents, top_n=len(documents)
            )
        except ProviderError:
            return fact_ids
        return [fact_ids[item["index"]] for item in results]

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
