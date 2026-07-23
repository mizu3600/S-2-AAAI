from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import torch

from qmshe.embedding.chebyshev import scipy_to_torch_sparse
from qmshe.embedding.graph_encoder import GraphSpectralSemanticEncoder
from qmshe.embedding.text_encoder import TextEncoder, encode_documents, encode_queries
from qmshe.generation.generator import EvidenceGenerator
from qmshe.graph.ordinary import GraphProfile, build_ordinary_graph
from qmshe.graph.ordinary_incremental import plan_graph_incremental_update
from qmshe.ingest.schemas import Corpus
from qmshe.observability import RuntimeMetrics
from qmshe.pipeline import verbalize_fact
from qmshe.providers import ProviderError, SiliconFlowClient
from qmshe.retrieval.ann_retriever import ExactVectorIndex
from qmshe.retrieval.ann_retriever import SearchHit
from qmshe.retrieval.context_builder import build_context
from qmshe.retrieval.evidence_verifier import verify_candidates
from qmshe.retrieval.graph_reranker import graph_rerank
from qmshe.retrieval.seed_retriever import BM25Retriever, reciprocal_rank_fusion
from qmshe.scaling.cache import VersionedQueryCache
from qmshe.scaling.versioning import ArtifactVersion


@dataclass
class GraphQueryResult:
    mode: str
    profile: str
    index_strategy: str
    answer: str
    citations: list[dict]
    retrieved_nodes: list[str]
    retrieved_facts: list[str]
    band_weights: dict[str, float]
    evidence_path: list[str]
    rejected_candidates: dict[str, str]
    scores: list[dict]


class QMSGEGraphPipeline:
    """Independent ordinary-graph branch of QMSxE-RAG.

    It shares source Corpus and providers with QMSHE, but constructs no incidence matrix and uses
    no hypergraph Laplacian. The spectral operator is the normalized ordinary adjacency.
    """

    def __init__(
        self,
        corpus: Corpus,
        text_encoder: TextEncoder | None = None,
        profile: GraphProfile | str = GraphProfile.REIFIED_FACT,
        index_strategy: str = "hybrid",
        reranker=None,
        seed: int = 42,
        enable_remote_reranker: bool = True,
        use_graph_rerank: bool = False,
        use_bm25: bool = True,
        dense_only: bool = False,
    ):
        if not corpus.entities or not corpus.evidence_hyperedges:
            raise ValueError("corpus must contain entities and evidence facts")
        self.corpus = corpus
        self.profile = GraphProfile(profile)
        if index_strategy not in {"single", "multi", "hybrid"}:
            raise ValueError("index_strategy must be single, multi or hybrid")
        self.index_strategy = index_strategy
        self.text_encoder = text_encoder or TextEncoder()
        self.reranker = reranker
        self.seed = seed
        self.enable_remote_reranker = enable_remote_reranker
        self.use_graph_rerank = use_graph_rerank
        self.use_bm25 = use_bm25
        self.dense_only = dense_only
        self.generator = EvidenceGenerator()
        self.version = ArtifactVersion.create()
        self.query_cache = VersionedQueryCache(max_items=1024)
        self.metrics = RuntimeMetrics()
        self._build()

    def _build(self) -> None:
        self.artifacts = build_ordinary_graph(self.corpus, self.profile)
        self.node_ids = self.artifacts.node_ids
        self.node_texts = self.artifacts.node_texts
        raw_np = encode_documents(self.text_encoder, self.node_texts)
        self.raw_features = torch.tensor(raw_np, dtype=torch.float32)
        self.propagation = scipy_to_torch_sparse(self.artifacts.propagation)
        torch.manual_seed(self.seed)
        self.model = GraphSpectralSemanticEncoder(
            self.raw_features.shape[1], raw_dim=64, band_dim=32
        )
        self.model.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.propagation)
        self._rebuild_retrievers()

    def _rebuild_retrievers(self) -> None:
        self.graph_index = ExactVectorIndex(self.node_ids, self.node_bands["full"].numpy())
        self.band_indices = {
            name: ExactVectorIndex(self.node_ids, self.node_bands[name].numpy())
            for name in ("raw", "low", "mid", "high")
        }
        self.raw_index = ExactVectorIndex(self.node_ids, self.raw_features.numpy())
        names = {entity.entity_id: entity.canonical_name for entity in self.corpus.entities}
        self.fact_text_by_id = {
            fact.hyperedge_id: verbalize_fact(fact, names)
            for fact in self.corpus.evidence_hyperedges
        }
        self.bm25 = BM25Retriever(list(self.fact_text_by_id), list(self.fact_text_by_id.values()))

    def train_stage_a(
        self,
        training_pairs: list[tuple[str, set[str]]],
        epochs: int = 10,
        learning_rate: float = 2e-4,
        gradient_clip: float = 1.0,
    ) -> list[float]:
        node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        prepared: list[tuple[torch.Tensor, list[int]]] = []
        for question, positive_ids in training_pairs:
            expanded = set(positive_ids)
            if self.profile is GraphProfile.ENTITY_RELATION:
                for entity_id, fact_ids in self.artifacts.facts_by_entity.items():
                    if fact_ids & positive_ids:
                        expanded.add(entity_id)
            positive_indices = [node_index[item] for item in expanded if item in node_index]
            if positive_indices:
                query = torch.tensor(
                    encode_queries(self.text_encoder, [question])[0], dtype=torch.float32
                )
                prepared.append((query, positive_indices))
        if not prepared:
            raise ValueError("no training positives exist in the ordinary graph index")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        history: list[float] = []
        self.model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for query, positives in prepared:
                optimizer.zero_grad(set_to_none=True)
                bands = self.model.encode_nodes(self.raw_features, self.propagation)
                query_vector, _ = self.model.encode_query(query, self.raw_features, bands)
                scores = bands["full"] @ query_vector
                loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(
                    scores[torch.tensor(positives)], dim=0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                epoch_loss += float(loss.detach())
            history.append(epoch_loss / len(prepared))
        self.model.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.propagation)
        self._rebuild_retrievers()
        self.query_cache.clear()
        return history

    def load_stage_a_checkpoint(self, checkpoint: str | Path | dict) -> None:
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        if payload.get("mode") != "graph":
            raise ValueError("Stage A checkpoint is not an ordinary-graph checkpoint")
        if payload.get("profile") != self.profile.value:
            raise ValueError("Stage A checkpoint graph profile does not match")
        if payload.get("input_dim") != self.raw_features.shape[1]:
            raise ValueError("Stage A checkpoint encoder dimension does not match")
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.propagation)
        self._rebuild_retrievers()
        self.query_cache.clear()

    def query(
        self,
        question: str,
        top_k: int = 12,
        return_debug: bool = True,
        candidate_count: int | None = None,
    ) -> GraphQueryResult:
        started = perf_counter()
        cache_key = self.query_cache.key(
            question,
            self.version,
            {
                "mode": "graph",
                "profile": self.profile.value,
                "index_strategy": self.index_strategy,
                "top_k": top_k,
                "debug": return_debug,
                "candidate_count": candidate_count,
                "use_graph_rerank": self.use_graph_rerank,
                "use_bm25": self.use_bm25,
                "dense_only": self.dense_only,
            },
        )
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            self.metrics.observe(cached, (perf_counter() - started) * 1000, cache_hit=True)
            return cached
        query_np = encode_queries(self.text_encoder, [question])[0]
        query_tensor = torch.tensor(query_np, dtype=torch.float32)
        with torch.no_grad():
            query_parts, gate = self.model.encode_query_parts(
                query_tensor, self.raw_features, self.node_bands, top_m=64, temperature=0.05
            )
            query_vector = torch.cat(
                [
                    gate[index] * query_parts[name]
                    for index, name in enumerate(("raw", "low", "mid", "high"))
                ]
            )
        candidate_count = candidate_count or max(30, top_k * 3)
        if candidate_count < top_k:
            raise ValueError("candidate_count must be at least top_k")
        retrieval_lists = []
        if self.index_strategy in {"single", "hybrid"}:
            retrieval_lists.append(
                self.graph_index.search(query_vector.numpy(), candidate_count, "graph-single-index")
            )
        if self.index_strategy in {"multi", "hybrid"}:
            band_hits = {
                name: self.band_indices[name].search(
                    query_parts[name].numpy(), candidate_count, f"graph-{name}"
                )
                for name in ("raw", "low", "mid", "high")
            }
            retrieval_lists.append(_weighted_band_fusion(band_hits, gate, candidate_count))
        raw_hits = self.raw_index.search(query_np, candidate_count, "raw-text")
        lexical_hits = self.bm25.search(question, candidate_count)
        if self.dense_only:
            retrieval_lists = [raw_hits]
        else:
            retrieval_lists.append(raw_hits)
            if self.use_bm25:
                retrieval_lists.append(lexical_hits)
        fused = reciprocal_rank_fusion(retrieval_lists)
        reranked = (
            graph_rerank(fused[:50], self.artifacts.graph) if self.use_graph_rerank else fused[:50]
        )

        node_ids = [hit.object_id for hit in reranked if hit.object_id in self.artifacts.graph]
        fact_candidates = self._facts_from_candidates([hit.object_id for hit in reranked])
        fact_candidates = self._remote_rerank(question, fact_candidates)
        verification = verify_candidates(fact_candidates, self.corpus)
        fact_ids = verification.accepted_ids[:top_k]
        context, citations = build_context(self.corpus, fact_ids, limit=top_k)
        answer = self.generator.generate(question, context)
        result = GraphQueryResult(
            mode="graph",
            profile=self.profile.value,
            index_strategy=self.index_strategy,
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
        self.query_cache.put(cache_key, result)
        self.metrics.observe(result, (perf_counter() - started) * 1000)
        return result

    def _facts_from_candidates(self, candidate_ids: list[str]) -> list[str]:
        ranked: list[str] = []
        for candidate_id in candidate_ids:
            if candidate_id in self.artifacts.fact_by_node:
                ranked.append(self.artifacts.fact_by_node[candidate_id])
            ranked.extend(sorted(self.artifacts.facts_by_entity.get(candidate_id, set())))
            if candidate_id in self.fact_text_by_id:
                ranked.append(candidate_id)
        return list(dict.fromkeys(ranked))

    def _remote_rerank(self, question: str, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        if self.reranker is not None:
            order = self.reranker.rank(question, [self.fact_text_by_id[item] for item in fact_ids])
            return [fact_ids[index] for index in order]
        if not self.enable_remote_reranker:
            return fact_ids
        try:
            results = SiliconFlowClient().rerank(
                question, [self.fact_text_by_id[item] for item in fact_ids], top_n=len(fact_ids)
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

    def incremental_update(self, updated: Corpus) -> dict:
        new_artifacts = build_ordinary_graph(updated, self.profile)
        plan = plan_graph_incremental_update(self.artifacts.graph, new_artifacts.graph)
        new_texts = new_artifacts.node_texts
        new_raw = torch.tensor(encode_documents(self.text_encoder, new_texts), dtype=torch.float32)
        new_propagation = scipy_to_torch_sparse(new_artifacts.propagation)
        with torch.no_grad():
            recalculated = self.model.encode_nodes(new_raw, new_propagation)
        update_kind = "full_rebuild" if plan.requires_full_rebuild else "local_two_hop"
        if not plan.requires_full_rebuild:
            old_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
            new_index = {node_id: index for index, node_id in enumerate(new_artifacts.node_ids)}
            affected = set(plan.affected_nodes)
            for band_name in ("raw", "low", "mid", "high", "full"):
                for node_id in set(old_index) & set(new_index) - affected:
                    recalculated[band_name][new_index[node_id]] = self.node_bands[band_name][
                        old_index[node_id]
                    ]
        self.corpus = updated
        self.artifacts = new_artifacts
        self.node_ids = new_artifacts.node_ids
        self.node_texts = new_texts
        self.raw_features = new_raw
        self.propagation = new_propagation
        self.node_bands = recalculated
        self.version = ArtifactVersion.create()
        self._rebuild_retrievers()
        self.query_cache.clear()
        return {**plan.__dict__, "update_kind": update_kind, "profile": self.profile.value}


def _weighted_band_fusion(
    band_hits: dict[str, list[SearchHit]],
    gate: torch.Tensor,
    top_k: int,
) -> list[SearchHit]:
    totals: dict[str, float] = {}
    names = ("raw", "low", "mid", "high")
    for band_index, name in enumerate(names):
        weight = float(gate[band_index])
        for hit in band_hits[name]:
            # Rank term makes scores comparable across separately normalized vector spaces.
            totals[hit.object_id] = totals.get(hit.object_id, 0.0) + weight / (60 + hit.rank)
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SearchHit(object_id, score, rank + 1, "graph-multi-index")
        for rank, (object_id, score) in enumerate(ranked)
    ]
