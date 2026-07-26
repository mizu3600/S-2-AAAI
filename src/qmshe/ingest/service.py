from pathlib import Path

from qmshe.extraction.canonicalizer import canonicalize_entities
from qmshe.extraction.entity_extractor import extract_entities_rule_based
from qmshe.extraction.fact_extractor import extract_facts_rule_based, extract_facts_with_llm
from qmshe.ingest.chunker import chunk_document
from qmshe.ingest.pdf_parser import parse_document
from qmshe.ingest.schemas import Corpus
from qmshe.providers import DeepSeekClient, ProviderError


def ingest_document(source: str | Path, domain: str = "PSC") -> Corpus:
    parsed = parse_document(source, domain)
    chunks = chunk_document(parsed)
    entities = canonicalize_entities(extract_entities_rule_based(chunks))
    try:
        facts = extract_facts_with_llm(chunks, entities, DeepSeekClient())
    except ProviderError:
        facts = extract_facts_rule_based(chunks, entities)
    return Corpus(
        documents=[parsed.document],
        chunks=chunks,
        entities=entities,
        evidence_hyperedges=facts,
    )
