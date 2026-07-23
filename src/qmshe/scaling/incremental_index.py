from dataclasses import dataclass

import numpy as np

from qmshe.graph.incremental import approximate_new_spectral_embedding, needs_full_rebuild
from qmshe.ingest.schemas import Corpus


@dataclass(frozen=True)
class IncrementalUpdatePlan:
    new_entity_ids: list[str]
    new_hyperedge_ids: list[str]
    spectral_status: str
    requires_full_rebuild: bool
    reasons: list[str]


def plan_incremental_update(old: Corpus, updated: Corpus) -> IncrementalUpdatePlan:
    old_entities = {entity.entity_id for entity in old.entities}
    old_edges = {edge.hyperedge_id for edge in old.evidence_hyperedges}
    new_entities = [entity.entity_id for entity in updated.entities if entity.entity_id not in old_entities]
    new_edges = [edge.hyperedge_id for edge in updated.evidence_hyperedges if edge.hyperedge_id not in old_edges]
    rebuild = needs_full_rebuild(
        len(old.entities), len(new_entities), len(old.evidence_hyperedges), len(new_edges)
    )
    reasons = []
    if len(new_entities) / max(len(old.entities), 1) >= 0.03:
        reasons.append("new node ratio reached 3%")
    if len(new_edges) / max(len(old.evidence_hyperedges), 1) >= 0.05:
        reasons.append("new hyperedge ratio reached 5%")
    return IncrementalUpdatePlan(
        new_entity_ids=new_entities, new_hyperedge_ids=new_edges,
        spectral_status="approximate" if new_entities or new_edges else "exact",
        requires_full_rebuild=rebuild, reasons=reasons,
    )


def approximate_band_vectors(
    raw_vectors: np.ndarray,
    existing_raw: np.ndarray,
    existing_bands: np.ndarray,
    neighbor_count: int = 8,
) -> np.ndarray:
    output = []
    existing_norm = existing_raw / np.maximum(np.linalg.norm(existing_raw, axis=1, keepdims=True), 1e-12)
    for raw in raw_vectors:
        normalized = raw / max(float(np.linalg.norm(raw)), 1e-12)
        similarity = existing_norm @ normalized
        indices = np.argsort(-similarity)[: min(neighbor_count, len(similarity))]
        weights = np.maximum(similarity[indices], 0)
        output.append(approximate_new_spectral_embedding(raw, existing_bands[indices], weights))
    return np.asarray(output, dtype=np.float32)

