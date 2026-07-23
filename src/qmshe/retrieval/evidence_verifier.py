from dataclasses import dataclass

from qmshe.ingest.schemas import Corpus


@dataclass(frozen=True)
class VerificationResult:
    accepted_ids: list[str]
    rejected: dict[str, str]


def verify_candidates(candidate_ids: list[str], corpus: Corpus) -> VerificationResult:
    evidence = {fact.hyperedge_id: fact for fact in corpus.evidence_hyperedges}
    chunks = {chunk.chunk_id for chunk in corpus.chunks}
    accepted, rejected = [], {}
    semantic_ids = {edge.semantic_edge_id for edge in corpus.semantic_hyperedges}
    for object_id in candidate_ids:
        if object_id in semantic_ids:
            rejected[object_id] = "semantic hyperedges are retrieval-only"
        elif object_id in evidence:
            if not set(evidence[object_id].evidence_chunk_ids) <= chunks:
                rejected[object_id] = "missing source chunk"
            else:
                accepted.append(object_id)
        else:
            rejected[object_id] = "not an evidence hyperedge"
    return VerificationResult(accepted, rejected)

