import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from qmshe.extraction.canonicalizer import canonicalize_entities
from qmshe.extraction.entity_extractor import extract_entities_rule_based
from qmshe.extraction.fact_extractor import extract_facts_rule_based, extract_facts_with_llm
from qmshe.ingest.chunker import chunk_document
from qmshe.ingest.schemas import Corpus
from qmshe.providers import DeepSeekClient, ProviderError

PSC_ENTITY_TYPES = {
    "material", "composition", "additive", "passivation_material", "fabrication_process",
    "device_architecture", "layer", "experimental_condition", "measurement_method",
    "performance_metric", "degradation_mechanism", "stability_condition", "organization",
    "author", "paper", "mechanism",
}

PSC_PREDICATES = {
    "improves_device_performance", "passivates_surface_defects", "reduces_nonradiative_recombination",
    "increases_open_circuit_voltage", "changes_power_conversion_efficiency", "improves_stability",
    "causes_degradation", "uses_fabrication_process", "measured_under", "composed_of",
}

UNIT_FACTORS = {
    ("mv", "V"): 1e-3, ("v", "V"): 1.0, ("ma/cm2", "A/m²"): 10.0,
    ("ma cm-2", "A/m²"): 10.0, ("h", "s"): 3600.0, ("min", "s"): 60.0,
    ("%", "fraction"): 0.01,
}


class NormalizedMeasurement(BaseModel):
    value: float
    unit: str
    original_text: str


class PSCBenchmarkItem(BaseModel):
    question_id: str
    question: str
    answer: str | list[str]
    supporting_chunk_ids: list[str]
    supporting_hyperedge_ids: list[str]
    bridge_entity_ids: list[str] = Field(default_factory=list)
    gold_path: list[str] = Field(default_factory=list)
    hop_count: int = Field(ge=1, le=8)
    query_type: str
    source_document_ids: list[str]
    annotator: str | None = None
    review_status: str = "draft"

    @model_validator(mode="after")
    def sources_are_present(self):
        if not self.source_document_ids or not self.supporting_chunk_ids:
            raise ValueError("PSC benchmark items require documents and supporting chunks")
        return self


def normalize_measurement(text: str) -> NormalizedMeasurement | None:
    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*(mV|V|mA\s*(?:/cm2|cm-2)|h|min|%)(?!\w)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value, raw_unit = float(match.group(1)), re.sub(r"\s+", " ", match.group(2).lower())
    canonical_unit = next((target for (source, target), _ in UNIT_FACTORS.items() if source == raw_unit), raw_unit)
    factor = UNIT_FACTORS.get((raw_unit, canonical_unit), 1.0)
    return NormalizedMeasurement(value=value * factor, unit=canonical_unit, original_text=match.group(0))


def validate_psc_corpus(corpus: Corpus) -> list[str]:
    errors = []
    entity_ids = {entity.entity_id for entity in corpus.entities}
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    for entity in corpus.entities:
        if entity.entity_type not in PSC_ENTITY_TYPES:
            errors.append(f"unknown entity type: {entity.entity_type}")
    for fact in corpus.evidence_hyperedges:
        if not set(fact.evidence_chunk_ids) <= chunk_ids:
            errors.append(f"{fact.hyperedge_id}: missing evidence chunk")
        if any(argument.entity_id not in entity_ids for argument in fact.arguments):
            errors.append(f"{fact.hyperedge_id}: missing argument entity")
        if len(fact.arguments) > 12:
            errors.append(f"{fact.hyperedge_id}: oversized hyperedge")
    return errors


def detect_qualifier_conflicts(facts) -> list[tuple[str, str, str]]:
    conflicts = []
    for index, left in enumerate(facts):
        left_entities = {argument.entity_id for argument in left.arguments}
        for right in facts[index + 1 :]:
            if left.predicate != right.predicate:
                continue
            if left_entities != {argument.entity_id for argument in right.arguments}:
                continue
            for key in set(left.qualifiers) & set(right.qualifiers):
                if left.qualifiers[key] != right.qualifiers[key]:
                    conflicts.append((left.hyperedge_id, right.hyperedge_id, key))
    return conflicts


def build_psc_corpus(directory: str | Path, use_llm: bool = True) -> Corpus:
    from qmshe.ingest.pdf_parser import parse_document

    directory = Path(directory)
    documents, chunks = [], []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        parsed = parse_document(path, domain="PSC")
        documents.append(parsed.document)
        chunks.extend(chunk_document(parsed))
    entities = canonicalize_entities(extract_entities_rule_based(chunks))
    if use_llm:
        try:
            facts = extract_facts_with_llm(chunks, entities, DeepSeekClient())
        except ProviderError:
            facts = extract_facts_rule_based(chunks, entities)
    else:
        facts = extract_facts_rule_based(chunks, entities)
    return Corpus(documents=documents, chunks=chunks, entities=entities, evidence_hyperedges=facts,
                  graph_version="psc-graph-v1")
