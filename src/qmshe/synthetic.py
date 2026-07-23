from qmshe.ingest.schemas import (
    Argument,
    Chunk,
    Corpus,
    Document,
    Entity,
    EvidenceHyperedge,
)


def make_synthetic_corpus() -> Corpus:
    document = Document(
        document_id="doc_synthetic", title="Synthetic PSC Evidence", source_uri="synthetic://psc"
    )
    chunks = [
        Chunk(chunk_id="chunk_1", document_id=document.document_id, section="Results", page=1,
              start_char=0, end_char=132,
              text="PEAI passivates surface defects in inverted PSCs and reduces non-radiative recombination."),
        Chunk(chunk_id="chunk_2", document_id=document.document_id, section="Results", page=2,
              start_char=133, end_char=250,
              text="Reduced non-radiative recombination improves open-circuit voltage under one-sun testing."),
        Chunk(chunk_id="chunk_3", document_id=document.document_id, section="Discussion", page=3,
              start_char=251, end_char=350,
              text="Humidity may reduce long-term device stability and is not evidence for voltage improvement."),
    ]
    entities = [
        Entity(entity_id="ent_peai", canonical_name="phenethylammonium iodide", aliases=["PEAI"],
               entity_type="passivation_material", description="A surface passivation material."),
        Entity(entity_id="ent_passivation", canonical_name="surface defect passivation", aliases=[],
               entity_type="mechanism", description="Passivation of surface defects."),
        Entity(entity_id="ent_recombination", canonical_name="non-radiative recombination", aliases=[],
               entity_type="mechanism", description="A carrier loss mechanism."),
        Entity(entity_id="ent_voc", canonical_name="open-circuit voltage", aliases=["Voc"],
               entity_type="performance_metric", description="Device open-circuit voltage."),
        Entity(entity_id="ent_inverted", canonical_name="inverted perovskite solar cell", aliases=["inverted PSC"],
               entity_type="device_architecture", description="An inverted PSC architecture."),
        Entity(entity_id="ent_humidity", canonical_name="humidity", aliases=[],
               entity_type="experimental_condition", description="Environmental humidity."),
        Entity(entity_id="ent_stability", canonical_name="device stability", aliases=[],
               entity_type="performance_metric", description="Long-term stability."),
    ]
    facts = [
        EvidenceHyperedge(
            hyperedge_id="fact_1", predicate="passivates_surface_defects",
            arguments=[Argument(role="material", entity_id="ent_peai"),
                       Argument(role="mechanism", entity_id="ent_passivation"),
                       Argument(role="architecture", entity_id="ent_inverted")],
            evidence_chunk_ids=["chunk_1"], evidence_sentence=chunks[0].text, confidence=0.96,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_2", predicate="reduces_nonradiative_recombination",
            arguments=[Argument(role="mechanism", entity_id="ent_passivation"),
                       Argument(role="affected_property", entity_id="ent_recombination")],
            evidence_chunk_ids=["chunk_1"], evidence_sentence=chunks[0].text, confidence=0.94,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_3", predicate="improves_open_circuit_voltage",
            arguments=[Argument(role="mechanism", entity_id="ent_recombination"),
                       Argument(role="result", entity_id="ent_voc")],
            qualifiers={"measurement_condition": "1 sun"}, evidence_chunk_ids=["chunk_2"],
            evidence_sentence=chunks[1].text, confidence=0.93,
        ),
        EvidenceHyperedge(
            hyperedge_id="fact_4", predicate="reduces_device_stability",
            arguments=[Argument(role="condition", entity_id="ent_humidity"),
                       Argument(role="result", entity_id="ent_stability")],
            evidence_chunk_ids=["chunk_3"], evidence_sentence=chunks[2].text, confidence=0.82,
        ),
    ]
    return Corpus(documents=[document], chunks=chunks, entities=entities, evidence_hyperedges=facts)

