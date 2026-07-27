from dataclasses import dataclass

from s2rag.ingest.schemas import Corpus


@dataclass(frozen=True)
class VerificationResult:
    accepted_ids: list[str]
    rejected: dict[str, str]


def verify_candidates(candidate_ids: list[str], corpus: Corpus) -> VerificationResult:
    evidence = {fact.hyperedge_id: fact for fact in corpus.evidence_hyperedges}
    chunks = {chunk.chunk_id for chunk in corpus.chunks}
    accepted, rejected = [], {}
    for object_id in candidate_ids:
        if object_id in evidence:
            if not set(evidence[object_id].evidence_chunk_ids) <= chunks:
                rejected[object_id] = "missing source chunk"
            else:
                accepted.append(object_id)
        else:
            rejected[object_id] = "not an evidence hyperedge"
    return VerificationResult(accepted, rejected)
