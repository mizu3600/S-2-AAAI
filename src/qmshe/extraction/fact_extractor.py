import hashlib
import json

from qmshe.ingest.schemas import Argument, Chunk, Entity, EvidenceHyperedge
from qmshe.providers import DeepSeekClient


SYSTEM_PROMPT = """You extract only explicitly supported n-ary facts from scientific text.
Return JSON with key facts. Each fact contains predicate, arguments [{role, entity_name}], qualifiers,
evidence_sentence, and confidence. Never infer causality or invent entities."""


def extract_facts_with_llm(
    chunks: list[Chunk], entities: list[Entity], client: DeepSeekClient
) -> list[EvidenceHyperedge]:
    by_name = {entity.canonical_name.casefold(): entity for entity in entities}
    facts: list[EvidenceHyperedge] = []
    for chunk in chunks:
        payload = client.complete_json(
            SYSTEM_PROMPT,
            json.dumps(
                {"chunk_id": chunk.chunk_id, "text": chunk.text, "known_entities": list(by_name)},
                ensure_ascii=False,
            ),
        )
        for raw in payload.get("facts", []):
            arguments = []
            for argument in raw.get("arguments", []):
                entity = by_name.get(str(argument.get("entity_name", "")).casefold())
                if entity:
                    arguments.append(Argument(role=argument["role"], entity_id=entity.entity_id))
            if len({arg.entity_id for arg in arguments}) < 2:
                continue
            digest = hashlib.sha1(
                f"{chunk.chunk_id}:{raw.get('predicate')}:{arguments}".encode()
            ).hexdigest()[:12]
            facts.append(
                EvidenceHyperedge(
                    hyperedge_id=f"fact_{digest}",
                    predicate=raw["predicate"],
                    arguments=arguments,
                    qualifiers=raw.get("qualifiers", {}),
                    evidence_chunk_ids=[chunk.chunk_id],
                    evidence_sentence=raw.get("evidence_sentence", chunk.text),
                    confidence=float(raw.get("confidence", 0.5)),
                )
            )
    return facts


def extract_facts_rule_based(chunks: list[Chunk], entities: list[Entity]) -> list[EvidenceHyperedge]:
    """Deterministic demo extractor for the canonical PEAI example."""
    facts: list[EvidenceHyperedge] = []
    for chunk in chunks:
        present = [entity for entity in entities if _mentioned(entity, chunk.text)]
        if len(present) < 2:
            continue
        roles = {
            "passivation_material": "material",
            "mechanism": "mechanism",
            "performance_metric": "result",
            "device_architecture": "architecture",
        }
        args = [Argument(role=roles.get(e.entity_type, "related"), entity_id=e.entity_id) for e in present]
        digest = hashlib.sha1(f"{chunk.chunk_id}:{','.join(a.entity_id for a in args)}".encode()).hexdigest()[:12]
        facts.append(
            EvidenceHyperedge(
                hyperedge_id=f"fact_{digest}",
                predicate="improves_device_performance",
                arguments=args,
                qualifiers={},
                evidence_chunk_ids=[chunk.chunk_id],
                evidence_sentence=chunk.text,
                confidence=0.9,
            )
        )
    return facts


def _mentioned(entity: Entity, text: str) -> bool:
    lowered = text.casefold()
    return entity.canonical_name.casefold() in lowered or any(a.casefold() in lowered for a in entity.aliases)

