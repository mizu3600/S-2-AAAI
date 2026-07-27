from pathlib import Path

from s2rag.extraction.canonicalizer import canonicalize_entities
from s2rag.extraction.entity_extractor import extract_entities_with_llm
from s2rag.extraction.fact_extractor import extract_facts_with_llm
from s2rag.ingest.chunker import chunk_document
from s2rag.ingest.pdf_parser import parse_document
from s2rag.ingest.schemas import Corpus
from s2rag.providers import DeepSeekClient


def ingest_document(
    source: str | Path,
    domain: str = "general",
    client: DeepSeekClient | None = None,
) -> Corpus:
    parsed = parse_document(source, domain)
    chunks = chunk_document(parsed)
    client = client or DeepSeekClient()
    entities = canonicalize_entities(extract_entities_with_llm(chunks, client))
    facts = extract_facts_with_llm(chunks, entities, client)
    return Corpus(
        documents=[parsed.document],
        chunks=chunks,
        entities=entities,
        evidence_hyperedges=facts,
    )
