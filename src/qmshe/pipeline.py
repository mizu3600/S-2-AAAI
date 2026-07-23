import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import networkx as nx
import torch

from qmshe.embedding.chebyshev import scipy_to_torch_sparse
from qmshe.embedding.joint_encoder import JointSpectralSemanticEncoder
from qmshe.embedding.query_gate import QueryRelationGate
from qmshe.embedding.text_encoder import TextEncoder, encode_documents, encode_queries
from qmshe.generation.generator import EvidenceGenerator
from qmshe.graph.evidence_graph import build_evidence_graph, evidence_paths
from qmshe.graph.incidence import build_incidence, build_role_incidence
from qmshe.graph.laplacian import build_joint_spectral_laplacian
from qmshe.graph.semantic_graph import build_mutual_knn_graph
from qmshe.ingest.schemas import Corpus, EvidenceHyperedge, SemanticHyperedge
from qmshe.observability import RuntimeMetrics
from qmshe.providers import ProviderError, SiliconFlowClient
from qmshe.retrieval.ann_retriever import ExactVectorIndex
from qmshe.retrieval.context_builder import build_context
from qmshe.retrieval.evidence_verifier import verify_candidates
from qmshe.retrieval.graph_reranker import graph_rerank
from qmshe.retrieval.seed_retriever import BM25Retriever, reciprocal_rank_fusion
from qmshe.scaling.cache import VersionedQueryCache
from qmshe.scaling.incremental_index import approximate_band_vectors, plan_incremental_update
from qmshe.scaling.versioning import ArtifactVersion


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
    retrieved_entities: list[str]
    retrieved_hyperedges: list[str]
    band_weights: dict[str, float]
    relation_weights: dict[str, float]
    evidence_path: list[str]
    rejected_candidates: dict[str, str]
    scores: list[dict]


class QMSHEPipeline:
    def __init__(
        self,
        corpus: Corpus,
        text_encoder: TextEncoder | None = None,
        reranker=None,
        seed: int = 42,
        enable_remote_reranker: bool = True,
        semantic_weight: float = 0.25,
        role_semantic_weight: float = 0.10,
        use_graph_rerank: bool = False,
        use_bm25: bool = True,
        dense_only: bool = False,
    ):
        if not corpus.entities or not corpus.evidence_hyperedges:
            raise ValueError("corpus must contain entities and evidence hyperedges")
        self.corpus = corpus
        self.text_encoder = text_encoder or TextEncoder()
        self.reranker = reranker
        self.seed = seed
        self.enable_remote_reranker = enable_remote_reranker
        self.semantic_weight = semantic_weight
        self.role_semantic_weight = role_semantic_weight
        self.use_graph_rerank = use_graph_rerank
        self.use_bm25 = use_bm25
        self.dense_only = dense_only
        self.generator = EvidenceGenerator()
        self.evidence_graph = build_evidence_graph(corpus)
        self.version = ArtifactVersion.create()
        self.query_cache = VersionedQueryCache(max_items=1024)
        self.metrics = RuntimeMetrics()
        self.use_role_aware_query = True
        self._build()

    def _build(self) -> None:
        names = {entity.entity_id: entity.canonical_name for entity in self.corpus.entities}
        entity_texts = [
            f"{entity.canonical_name}. {entity.description}" for entity in self.corpus.entities
        ]
        fact_texts = [verbalize_fact(fact, names) for fact in self.corpus.evidence_hyperedges]
        self.object_ids = [entity.entity_id for entity in self.corpus.entities] + [
            fact.hyperedge_id for fact in self.corpus.evidence_hyperedges
        ]
        self.object_texts = entity_texts + fact_texts
        features_np = encode_documents(self.text_encoder, self.object_texts)
        features = torch.tensor(features_np, dtype=torch.float32)
        incidence = build_incidence(self.corpus.entities, self.corpus.evidence_hyperedges)
        fact_features = features_np[len(self.corpus.entities) :]
        semantic = build_mutual_knn_graph(fact_features, k=15, threshold=0.72, max_degree=40)
        if not self.corpus.semantic_hyperedges:
            self.corpus.semantic_hyperedges = [
                SemanticHyperedge(
                    semantic_edge_id=f"sem_{left}_{right}",
                    member_ids=[
                        self.corpus.evidence_hyperedges[left].hyperedge_id,
                        self.corpus.evidence_hyperedges[right].hyperedge_id,
                    ],
                    topic="embedding_similarity",
                    confidence=score,
                )
                for (left, right), score in semantic.similarities.items()
            ]
        self.laplacian_scipy = build_joint_spectral_laplacian(
            incidence.matrix, semantic.adjacency, semantic_weight=self.semantic_weight
        )
        laplacian = scipy_to_torch_sparse(self.laplacian_scipy)
        self.laplacian = laplacian
        torch.manual_seed(self.seed)
        self.model = JointSpectralSemanticEncoder(
            features.shape[1], raw_dim=64, band_dim=32, order=5
        )
        self.model.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(features, laplacian)
        role_incidence = build_role_incidence(self.corpus.entities, self.corpus.evidence_hyperedges)
        self.role_names = sorted(role_incidence)
        self.role_node_bands = {}
        self.role_laplacians = {}
        for role in self.role_names:
            role_laplacian = scipy_to_torch_sparse(
                build_joint_spectral_laplacian(
                    role_incidence[role],
                    semantic.adjacency,
                    semantic_weight=self.role_semantic_weight,
                )
            )
            self.role_laplacians[role] = role_laplacian
            with torch.no_grad():
                self.role_node_bands[role] = self.model.encode_nodes(features, role_laplacian)
        self.relation_gate = QueryRelationGate(features.shape[1], max(len(self.role_names), 1))
        self.relation_gate.eval()
        self.raw_features = features
        self.qmshe_index = ExactVectorIndex(self.object_ids, self.node_bands["full"].numpy())
        self.raw_index = ExactVectorIndex(self.object_ids, features_np)
        fact_ids = [fact.hyperedge_id for fact in self.corpus.evidence_hyperedges]
        self.bm25 = BM25Retriever(fact_ids, fact_texts)
        self.text_by_id = dict(zip(self.object_ids, self.object_texts, strict=True))

    def train_stage_a(
        self,
        training_pairs: list[tuple[str, set[str]]],
        epochs: int = 10,
        learning_rate: float = 2e-4,
        gradient_clip: float = 1.0,
        bridge_by_question: dict[str, set[str]] | None = None,
        bridge_loss_weight: float = 0.5,
        use_hard_negatives: bool = True,
        use_role_aware: bool = True,
    ) -> list[float]:
        if not training_pairs:
            raise ValueError("training pairs are empty")
        object_index = {object_id: index for index, object_id in enumerate(self.object_ids)}
        prepared = []
        bridge_by_question = bridge_by_question or {}
        generator = random.Random(42)
        for question, positive_ids in training_pairs:
            indices = [object_index[item] for item in positive_ids if item in object_index]
            if indices:
                query = torch.tensor(
                    encode_queries(self.text_encoder, [question])[0], dtype=torch.float32
                )
                excluded = set(indices)
                semantic_order = torch.argsort(self.raw_features @ query, descending=True).tolist()
                semantic = [index for index in semantic_order if index not in excluded][:4]
                structural_ids = set()
                for positive_id in positive_ids:
                    if positive_id in self.evidence_graph:
                        structural_ids.update(
                            nx.single_source_shortest_path_length(
                                self.evidence_graph, positive_id, cutoff=2
                            )
                        )
                structural = [
                    object_index[item]
                    for item in structural_ids
                    if item in object_index and object_index[item] not in excluded
                ][:4]
                negative_pool = [
                    index for index in range(len(self.object_ids)) if index not in excluded
                ]
                if use_hard_negatives:
                    selected = list(dict.fromkeys([*semantic, *structural]))
                    remaining = [index for index in negative_pool if index not in selected]
                    selected.extend(generator.sample(remaining, min(8, len(remaining))))
                else:
                    selected = generator.sample(negative_pool, min(16, len(negative_pool)))
                bridge_indices = [
                    object_index[item]
                    for item in bridge_by_question.get(question, set())
                    if item in object_index
                ]
                prepared.append((query, indices, selected, bridge_indices))
        if not prepared:
            raise ValueError("no training positives exist in the index")
        optimizer = torch.optim.AdamW(
            [*self.model.parameters(), *self.relation_gate.parameters()], lr=learning_rate
        )
        history = []
        self.model.train()
        self.relation_gate.train()
        for _ in range(epochs):
            total = 0.0
            for query_tensor, positive_indices, negative_indices, bridge_indices in prepared:
                optimizer.zero_grad(set_to_none=True)
                bands = self.model.encode_nodes(self.raw_features, self.laplacian)
                if use_role_aware:
                    relation_weights = self.relation_gate(query_tensor)
                    role_bands = {
                        role: self.model.encode_nodes(self.raw_features, laplacian)
                        for role, laplacian in self.role_laplacians.items()
                    }
                    conditioned = {"raw": bands["raw"]}
                    for band_name in ("low", "mid", "high"):
                        stacked = torch.stack(
                            [role_bands[role][band_name] for role in self.role_names], dim=0
                        )
                        conditioned[band_name] = torch.einsum(
                            "r,rnd->nd", relation_weights, stacked
                        )
                else:
                    conditioned = bands
                query_vector, _ = self.model.encode_query(
                    query_tensor, self.raw_features, conditioned, top_m=64, temperature=0.05
                )
                conditioned_full = torch.cat(
                    [
                        conditioned["raw"],
                        conditioned["low"],
                        conditioned["mid"],
                        conditioned["high"],
                    ],
                    dim=-1,
                )
                scores = conditioned_full @ query_vector
                positive_tensor = torch.tensor(positive_indices)
                positives = scores[positive_tensor]
                candidate_indices = torch.tensor([*positive_indices, *negative_indices])
                candidate_scores = scores[candidate_indices]
                loss = torch.logsumexp(candidate_scores, dim=0) - torch.logsumexp(positives, dim=0)
                if negative_indices:
                    negatives = scores[torch.tensor(negative_indices)]
                    loss = loss + 0.2 * torch.relu(0.2 - positives.max() + negatives.max())
                if bridge_indices and bridge_loss_weight > 0:
                    bridges = scores[torch.tensor(bridge_indices)]
                    loss = loss + bridge_loss_weight * (
                        torch.logsumexp(scores, dim=0) - torch.logsumexp(bridges, dim=0)
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.model.parameters(), *self.relation_gate.parameters()], gradient_clip
                )
                optimizer.step()
                total += float(loss.detach())
            history.append(total / len(prepared))
        self.model.eval()
        self.relation_gate.eval()
        self.use_role_aware_query = use_role_aware
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.laplacian)
            self.role_node_bands = {
                role: self.model.encode_nodes(self.raw_features, laplacian)
                for role, laplacian in self.role_laplacians.items()
            }
        self.qmshe_index = ExactVectorIndex(self.object_ids, self.node_bands["full"].numpy())
        self.query_cache.clear()
        return history

    def load_stage_a_checkpoint(self, checkpoint: str | Path | dict) -> None:
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        if payload.get("mode") != "hypergraph":
            raise ValueError("Stage A checkpoint is not a hypergraph checkpoint")
        if payload.get("input_dim") != self.raw_features.shape[1]:
            raise ValueError("Stage A checkpoint encoder dimension does not match")
        checkpoint_roles = payload.get("roles", [])
        if checkpoint_roles != self.role_names:
            raise ValueError(
                f"Stage A checkpoint roles {checkpoint_roles} do not match {self.role_names}"
            )
        self.model.load_state_dict(payload["model"])
        self.relation_gate.load_state_dict(payload["relation_gate"])
        self.model.eval()
        self.relation_gate.eval()
        with torch.no_grad():
            self.node_bands = self.model.encode_nodes(self.raw_features, self.laplacian)
            self.role_node_bands = {
                role: self.model.encode_nodes(self.raw_features, laplacian)
                for role, laplacian in self.role_laplacians.items()
            }
        self.qmshe_index = ExactVectorIndex(self.object_ids, self.node_bands["full"].numpy())
        self.query_cache.clear()

    def query(
        self,
        question: str,
        top_k: int = 12,
        return_debug: bool = True,
        candidate_count: int | None = None,
    ) -> QueryResult:
        started = perf_counter()
        cache_key = self.query_cache.key(
            question,
            self.version,
            {
                "top_k": top_k,
                "return_debug": return_debug,
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
        relation_weights, query_node_bands = self._relation_conditioned_bands(query_tensor)
        with torch.no_grad():
            query_vector, gate = self.model.encode_query(
                query_tensor, self.raw_features, query_node_bands, top_m=64, temperature=0.05
            )
        retrieval_count = candidate_count or max(30, top_k * 3)
        if retrieval_count < top_k:
            raise ValueError("candidate_count must be at least top_k")
        qmshe_hits = self.qmshe_index.search(query_vector.numpy(), retrieval_count, "qmshe")
        relation_full = torch.cat(
            [
                self.node_bands["raw"],
                query_node_bands["low"],
                query_node_bands["mid"],
                query_node_bands["high"],
            ],
            dim=-1,
        ).numpy()
        relation_hits = ExactVectorIndex(self.object_ids, relation_full).search(
            query_vector.numpy(), retrieval_count, "relation-aware"
        )
        raw_hits = self.raw_index.search(query_np, retrieval_count, "raw")
        bm25_hits = self.bm25.search(question, retrieval_count)
        retrieval_lists = [raw_hits] if self.dense_only else [qmshe_hits, relation_hits, raw_hits]
        if self.use_bm25 and not self.dense_only:
            retrieval_lists.append(bm25_hits)
        fused = reciprocal_rank_fusion(retrieval_lists)
        reranked = (
            graph_rerank(fused[:50], self.evidence_graph) if self.use_graph_rerank else fused[:50]
        )
        reranked = self._remote_rerank(question, reranked)
        candidate_ids = [hit.object_id for hit in reranked]
        verification = verify_candidates(candidate_ids, self.corpus)
        fact_ids = verification.accepted_ids[:top_k]
        entity_ids = [object_id for object_id in candidate_ids if object_id.startswith("ent_")][:20]
        context, citations = build_context(self.corpus, fact_ids, limit=top_k)
        answer = self.generator.generate(question, context)
        paths = evidence_paths(self.evidence_graph, fact_ids[:8])
        gate_names = ["raw", "low", "mid", "high"]
        result = QueryResult(
            answer=answer,
            citations=citations,
            retrieved_entities=entity_ids,
            retrieved_hyperedges=fact_ids,
            band_weights={name: float(value) for name, value in zip(gate_names, gate, strict=True)},
            relation_weights={
                name: float(value)
                for name, value in zip(self.role_names, relation_weights, strict=True)
            },
            evidence_path=paths[0] if paths else (fact_ids[:1] or []),
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

    def _relation_conditioned_bands(self, query_tensor: torch.Tensor):
        if not self.role_names:
            return torch.ones(1), self.node_bands
        if not self.use_role_aware_query:
            return torch.full((len(self.role_names),), 1.0 / len(self.role_names)), self.node_bands
        with torch.no_grad():
            weights = self.relation_gate(query_tensor)
        combined = {"raw": self.node_bands["raw"]}
        for band in ("low", "mid", "high"):
            stacked = torch.stack(
                [self.role_node_bands[role][band] for role in self.role_names], dim=0
            )
            combined[band] = torch.einsum("r,rnd->nd", weights, stacked)
        return weights, combined

    def incremental_update(self, updated: Corpus) -> dict:
        plan = plan_incremental_update(self.corpus, updated)
        new_entity_set = set(plan.new_entity_ids)
        new_fact_set = set(plan.new_hyperedge_ids)
        if not new_entity_set and not new_fact_set:
            return {**plan.__dict__, "indexed_objects": 0}
        names = {entity.entity_id: entity.canonical_name for entity in updated.entities}
        new_entities = [entity for entity in updated.entities if entity.entity_id in new_entity_set]
        new_facts = [
            fact for fact in updated.evidence_hyperedges if fact.hyperedge_id in new_fact_set
        ]
        new_ids = [entity.entity_id for entity in new_entities] + [
            fact.hyperedge_id for fact in new_facts
        ]
        new_texts = [f"{entity.canonical_name}. {entity.description}" for entity in new_entities]
        new_texts.extend(verbalize_fact(fact, names) for fact in new_facts)
        raw_np = encode_documents(self.text_encoder, new_texts)
        existing_bands = torch.stack(
            [self.node_bands["low"], self.node_bands["mid"], self.node_bands["high"]], dim=1
        ).numpy()
        approximate = approximate_band_vectors(
            raw_np, self.raw_features.numpy(), existing_bands, neighbor_count=8
        )
        raw_tensor = torch.tensor(raw_np, dtype=torch.float32)
        with torch.no_grad():
            raw_projected = self.model.raw_projection(raw_tensor)
        for band_index, band_name in enumerate(("low", "mid", "high")):
            self.node_bands[band_name] = torch.cat(
                [self.node_bands[band_name], torch.tensor(approximate[:, band_index, :])], dim=0
            )
            for role in self.role_names:
                self.role_node_bands[role][band_name] = torch.cat(
                    [
                        self.role_node_bands[role][band_name],
                        torch.tensor(approximate[:, band_index, :]),
                    ],
                    dim=0,
                )
        self.node_bands["raw"] = torch.cat([self.node_bands["raw"], raw_projected], dim=0)
        new_full = torch.cat(
            [raw_projected, *(torch.tensor(approximate[:, index, :]) for index in range(3))], dim=-1
        )
        self.node_bands["full"] = torch.cat([self.node_bands["full"], new_full], dim=0)
        self.raw_features = torch.cat([self.raw_features, raw_tensor], dim=0)
        self.object_ids.extend(new_ids)
        self.object_texts.extend(new_texts)
        self.text_by_id.update(zip(new_ids, new_texts, strict=True))
        self.corpus = updated
        self.evidence_graph = build_evidence_graph(updated)
        self.qmshe_index = ExactVectorIndex(self.object_ids, self.node_bands["full"].numpy())
        self.raw_index = ExactVectorIndex(self.object_ids, self.raw_features.numpy())
        fact_ids = [fact.hyperedge_id for fact in updated.evidence_hyperedges]
        fact_texts = [verbalize_fact(fact, names) for fact in updated.evidence_hyperedges]
        self.bm25 = BM25Retriever(fact_ids, fact_texts)
        self.query_cache.clear()
        # Role operators changed; a full rebuild will calibrate them exactly.
        return {**plan.__dict__, "indexed_objects": len(new_ids)}

    def ablation_search(self, question: str, variant: str, top_k: int = 20) -> list[str]:
        supported = {"ours-low", "ours-fixed", "ours-no-high", "ours-no-raw", "ours-full"}
        if variant not in supported:
            raise ValueError(f"{variant} requires a separately trained/rebuilt checkpoint")
        query_np = self.text_encoder.encode([question])[0]
        query_tensor = torch.tensor(query_np, dtype=torch.float32)
        _, query_node_bands = self._relation_conditioned_bands(query_tensor)
        with torch.no_grad():
            query_vector, gate = self.model.encode_query(
                query_tensor, self.raw_features, query_node_bands, top_m=64, temperature=0.05
            )
        nodes = self.node_bands["full"].clone()
        query = query_vector.clone()
        raw_end = self.model.raw_projection.out_features
        band = self.model.filter_bank.projections[0].out_features
        slices = {
            "raw": slice(0, raw_end),
            "low": slice(raw_end, raw_end + band),
            "mid": slice(raw_end + band, raw_end + 2 * band),
            "high": slice(raw_end + 2 * band, raw_end + 3 * band),
        }
        if variant == "ours-low":
            disabled = {"raw", "mid", "high"}
        elif variant == "ours-no-high":
            disabled = {"high"}
        elif variant == "ours-no-raw":
            disabled = {"raw"}
        else:
            disabled = set()
        for name in disabled:
            query[slices[name]] = 0
            nodes[:, slices[name]] = 0
        if variant == "ours-fixed":
            for index, name in enumerate(("raw", "low", "mid", "high")):
                query[slices[name]] *= 0.25 / max(float(gate[index]), 1e-8)
        index = ExactVectorIndex(self.object_ids, nodes.numpy())
        return [
            hit.object_id
            for hit in index.search(query.numpy(), top_k * 3, variant)
            if hit.object_id.startswith("fact_")
        ][:top_k]

    def _remote_rerank(self, question: str, hits):
        fact_hits = [hit for hit in hits if hit.object_id.startswith("fact_")][:50]
        if not fact_hits:
            return hits
        if self.reranker is not None:
            order = self.reranker.rank(
                question, [self.text_by_id[hit.object_id] for hit in fact_hits]
            )
            local_order = [fact_hits[index] for index in order]
            remaining = [
                hit for hit in hits if hit.object_id not in {x.object_id for x in local_order}
            ]
            return local_order + remaining
        if not self.enable_remote_reranker:
            return hits
        try:
            client = SiliconFlowClient()
            results = client.rerank(
                question,
                [self.text_by_id[hit.object_id] for hit in fact_hits],
                top_n=len(fact_hits),
            )
        except ProviderError:
            return hits
        remote_order = [fact_hits[item["index"]] for item in results]
        remaining = [
            hit for hit in hits if hit.object_id not in {x.object_id for x in remote_order}
        ]
        return remote_order + remaining


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(corpus.model_dump_json(indent=2), encoding="utf-8")


def load_corpus(path: str | Path) -> Corpus:
    return Corpus.model_validate_json(Path(path).read_text(encoding="utf-8"))
