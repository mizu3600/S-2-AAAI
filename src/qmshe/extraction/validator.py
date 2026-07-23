from qmshe.ingest.schemas import Corpus, EvidenceHyperedge


def validate_evidence(fact: EvidenceHyperedge, corpus: Corpus) -> list[str]:
    errors: list[str] = []
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    entity_ids = {entity.entity_id for entity in corpus.entities}
    if not set(fact.evidence_chunk_ids) <= chunk_ids:
        errors.append("missing evidence chunk")
    if any(argument.entity_id not in entity_ids for argument in fact.arguments):
        errors.append("missing entity")
    if not fact.evidence_sentence.strip():
        errors.append("empty evidence sentence")
    return errors

