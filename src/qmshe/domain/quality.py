import random
from dataclasses import dataclass

from qmshe.ingest.schemas import Corpus


@dataclass(frozen=True)
class CorpusQualityReport:
    documents: int
    chunks: int
    entities: int
    facts: int
    provenance_coverage: float
    valid_argument_rate: float
    duplicate_chunk_rate: float
    sampled_fact_ids: list[str]


def audit_corpus(corpus: Corpus, sample_size: int = 30, seed: int = 42) -> CorpusQualityReport:
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    entity_ids = {entity.entity_id for entity in corpus.entities}
    sourced = sum(bool(set(fact.evidence_chunk_ids) & chunk_ids) for fact in corpus.evidence_hyperedges)
    valid = sum(all(argument.entity_id in entity_ids for argument in fact.arguments) for fact in corpus.evidence_hyperedges)
    normalized_chunks = [" ".join(chunk.text.casefold().split()) for chunk in corpus.chunks]
    duplicate_count = len(normalized_chunks) - len(set(normalized_chunks))
    rng = random.Random(seed)
    fact_ids = [fact.hyperedge_id for fact in corpus.evidence_hyperedges]
    sampled = rng.sample(fact_ids, min(sample_size, len(fact_ids)))
    return CorpusQualityReport(
        documents=len(corpus.documents), chunks=len(corpus.chunks), entities=len(corpus.entities),
        facts=len(corpus.evidence_hyperedges), provenance_coverage=sourced / max(len(fact_ids), 1),
        valid_argument_rate=valid / max(len(fact_ids), 1),
        duplicate_chunk_rate=duplicate_count / max(len(normalized_chunks), 1), sampled_fact_ids=sampled,
    )

