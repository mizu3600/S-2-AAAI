from s2rag.ingest.schemas import (
    Argument,
    Chunk,
    Corpus,
    Document,
    Entity,
    EvidenceHyperedge,
)


def make_synthetic_corpus() -> Corpus:
    document = Document(
        document_id="doc_synthetic",
        title="Synthetic General Evidence",
        source_uri="synthetic://general",
    )
    cache_evidence = (
        "A request cache avoids repeated database reads in an API service and reduces backend load."
    )
    latency_evidence = (
        "Reduced backend load improves median response latency under 100 concurrent requests."
    )
    availability_evidence = (
        "Network congestion may reduce service availability and is not evidence for latency "
        "improvement."
    )
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id=document.document_id,
            section="Results",
            page=1,
            start_char=0,
            end_char=len(cache_evidence),
            text=cache_evidence,
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id=document.document_id,
            section="Results",
            page=2,
            start_char=len(cache_evidence) + 1,
            end_char=len(cache_evidence) + len(latency_evidence) + 1,
            text=latency_evidence,
        ),
        Chunk(
            chunk_id="chunk_3",
            document_id=document.document_id,
            section="Discussion",
            page=3,
            start_char=len(cache_evidence) + len(latency_evidence) + 2,
            end_char=(
                len(cache_evidence)
                + len(latency_evidence)
                + len(availability_evidence)
                + 2
            ),
            text=availability_evidence,
        ),
    ]
    entities = [
        Entity(
            entity_id="ent_cache",
            canonical_name="request cache",
            aliases=["cache"],
            entity_type="software_component",
            description="A component that reuses stored request results.",
        ),
        Entity(
            entity_id="ent_reads",
            canonical_name="repeated database reads",
            aliases=[],
            entity_type="operation",
            description="Repeated reads sent to a database.",
        ),
        Entity(
            entity_id="ent_load",
            canonical_name="backend load",
            aliases=[],
            entity_type="system_metric",
            description="Work handled by backend services.",
        ),
        Entity(
            entity_id="ent_latency",
            canonical_name="median response latency",
            aliases=["response latency"],
            entity_type="performance_metric",
            description="Median time required to return a response.",
        ),
        Entity(
            entity_id="ent_service",
            canonical_name="API service",
            aliases=[],
            entity_type="software_system",
            description="A service that exposes an application programming interface.",
        ),
        Entity(
            entity_id="ent_congestion",
            canonical_name="network congestion",
            aliases=[],
            entity_type="operating_condition",
            description="Contention that limits network capacity.",
        ),
        Entity(
            entity_id="ent_availability",
            canonical_name="service availability",
            aliases=[],
            entity_type="reliability_metric",
            description="The proportion of time a service remains available.",
        ),
    ]
    facts = [
        EvidenceHyperedge(
            hyperedge_id="fact_1",
            predicate="avoids_repeated_database_reads",
            arguments=[
                Argument(role="component", entity_id="ent_cache"),
                Argument(role="operation", entity_id="ent_reads"),
                Argument(role="system", entity_id="ent_service"),
            ],
            evidence_chunk_ids=["chunk_1"],
            evidence_sentence=chunks[0].text,
            confidence=0.96,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_2",
            predicate="reduces_backend_load",
            arguments=[
                Argument(role="cause", entity_id="ent_reads"),
                Argument(role="result", entity_id="ent_load"),
            ],
            evidence_chunk_ids=["chunk_1"],
            evidence_sentence=chunks[0].text,
            confidence=0.94,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_3",
            predicate="improves_response_latency",
            arguments=[
                Argument(role="cause", entity_id="ent_load"),
                Argument(role="result", entity_id="ent_latency"),
            ],
            qualifiers={"measurement_condition": "100 concurrent requests"},
            evidence_chunk_ids=["chunk_2"],
            evidence_sentence=chunks[1].text,
            confidence=0.93,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_4",
            predicate="reduces_service_availability",
            arguments=[
                Argument(role="condition", entity_id="ent_congestion"),
                Argument(role="result", entity_id="ent_availability"),
            ],
            evidence_chunk_ids=["chunk_3"],
            evidence_sentence=chunks[2].text,
            confidence=0.82,
        ),
    ]
    return Corpus(
        documents=[document],
        chunks=chunks,
        entities=entities,
        evidence_hyperedges=facts,
    )
