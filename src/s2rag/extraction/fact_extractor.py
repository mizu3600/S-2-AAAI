import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from s2rag.extraction.batching import passage_batches
from s2rag.ingest.schemas import Argument, Chunk, Entity, EvidenceHyperedge
from s2rag.providers import DeepSeekClient
from s2rag.settings import get_settings


FACT_SYSTEM_PROMPT = """You are the relation extraction stage of an open-domain OpenIE system.
Convert the supplied source text into explicit, independently useful factual relations. The
known_entities list was produced by a separate LLM entity pass and is the only entity vocabulary
you may use.

Return one JSON object with key facts. Each fact must contain:
- predicate: a concise, source-faithful open relation phrase; do not use a fixed relation ontology;
- arguments: two or more {"role": lower_snake_case_role, "entity_id": known_entity_id} objects;
- qualifiers: a flat JSON object only for an explicit scalar or proposition-level condition that
  has no corresponding item in known_entities;
- evidence_sentence: the shortest exact contiguous quote from the source text that fully supports
  the fact;
- evidence_chunk_ids: the supplied chunk_id values containing that exact quote;
- confidence: a number from 0 to 1 measuring extraction confidence, not real-world plausibility.

Requirements:
1. Extract only propositions explicitly supported by the source. Do not add background knowledge,
   infer unstated causality, or turn correlation into causation.
2. Resolve pronouns and abbreviated references to the appropriate known entity_id when the
   antecedent is unambiguous in the supplied text. Do not create an entity for a pronoun. When
   resolution is needed, evidence_sentence must also include the explicit antecedent.
3. Each fact should contain central known entities and preferably connect multiple
   meaningful entities. Use only IDs from known_entities.
4. Split compound sentences into atomic facts when they express independent propositions. Preserve
   a genuinely multi-participant event as one n-ary fact with semantic roles instead of forcing it
   into unrelated binary triples.
5. Keep the predicate minimal. Do not hide an argument-worthy category or value in the predicate.
   Every known entity that explicitly fills a participant, category, time, place, amount, comparator,
   or condition role in the fact must appear in arguments. Do not repeat it as a qualifier and never
   place an entity_id inside qualifiers.
6. Use specific semantic roles such as agent, system, method, dataset, metric, amount, recipient,
   time, location, condition, input, and output. Roles and predicates are open-domain.
7. Copy evidence_sentence verbatim. If the quote does not occur in the source, the fact is invalid.

Example source:
"In 2021, the Atlas team evaluated S2RAG on HotpotQA with 500 questions. It improved recall by 8%
under a zero-shot setting."
Example known entities:
[{"entity_id": "e_team", "canonical_name": "Atlas team"},
 {"entity_id": "e_system", "canonical_name": "S2RAG"},
 {"entity_id": "e_dataset", "canonical_name": "HotpotQA"},
 {"entity_id": "e_size", "canonical_name": "500 questions"},
 {"entity_id": "e_year", "canonical_name": "2021"},
 {"entity_id": "e_metric", "canonical_name": "recall"},
 {"entity_id": "e_amount", "canonical_name": "8%"},
 {"entity_id": "e_setting", "canonical_name": "zero-shot setting"}]
Example output:
{"facts": [
  {"predicate": "evaluated", "arguments": [
    {"role": "agent", "entity_id": "e_team"},
    {"role": "system", "entity_id": "e_system"},
    {"role": "dataset", "entity_id": "e_dataset"},
    {"role": "sample_size", "entity_id": "e_size"},
    {"role": "time", "entity_id": "e_year"}],
   "qualifiers": {},
   "evidence_sentence": "In 2021, the Atlas team evaluated S2RAG on HotpotQA with 500 questions.",
   "confidence": 0.99},
  {"predicate": "improved", "arguments": [
    {"role": "system", "entity_id": "e_system"},
    {"role": "metric", "entity_id": "e_metric"},
    {"role": "amount", "entity_id": "e_amount"},
    {"role": "condition", "entity_id": "e_setting"}],
   "qualifiers": {},
   "evidence_sentence": "In 2021, the Atlas team evaluated S2RAG on HotpotQA with 500 questions. It improved recall by 8% under a zero-shot setting.",
   "confidence": 0.97}
]}"""


def extract_facts_with_llm(
    chunks: list[Chunk],
    entities: list[Entity],
    client: DeepSeekClient,
    *,
    system_prompt: str = FACT_SYSTEM_PROMPT,
    batch_max_chars: int | None = None,
    max_workers: int | None = None,
) -> list[EvidenceHyperedge]:
    settings = get_settings()
    batches = passage_batches(
        chunks,
        batch_max_chars or settings.extraction_batch_max_chars,
    )
    workers = max_workers or settings.extraction_workers
    with ThreadPoolExecutor(max_workers=min(workers, max(len(batches), 1))) as executor:
        responses = list(
            executor.map(
                lambda batch: (
                    batch,
                    _request_facts(batch, entities, client, system_prompt),
                ),
                batches,
            )
        )
    facts: dict[str, EvidenceHyperedge] = {}
    for batch, payload in responses:
        chunk_ids = {chunk.chunk_id for chunk in batch}
        batch_entities = [
            entity
            for entity in entities
            if any(
                mention.split(":", 1)[0] in chunk_ids
                for mention in entity.source_mentions
            )
        ]
        by_id = {entity.entity_id: entity for entity in batch_entities}
        for raw in payload.get("facts", []):
            if not isinstance(raw, dict):
                continue
            predicate = _compact_text(raw.get("predicate"))
            if not predicate:
                continue
            arguments = _arguments(raw.get("arguments"), by_id)
            if len({argument.entity_id for argument in arguments}) < 2:
                continue
            grounded = _grounded_batch_evidence(
                str(raw.get("evidence_sentence", "")).strip(),
                batch,
                raw.get("evidence_chunk_ids"),
            )
            if grounded is None:
                continue
            evidence_sentence, evidence_chunk_ids = grounded
            confidence = _confidence(raw.get("confidence", 0.5))
            qualifiers = _qualifiers(raw.get("qualifiers"))
            signature = json.dumps(
                {
                    "chunk_ids": evidence_chunk_ids,
                    "predicate": predicate.casefold(),
                    "arguments": sorted(
                        [{"role": item.role, "entity_id": item.entity_id} for item in arguments],
                        key=lambda item: (item["role"], item["entity_id"]),
                    ),
                    "qualifiers": qualifiers,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            digest = hashlib.sha1(signature.encode()).hexdigest()[:12]
            fact_id = f"fact_{digest}"
            candidate = EvidenceHyperedge(
                hyperedge_id=fact_id,
                predicate=predicate,
                arguments=arguments,
                qualifiers=qualifiers,
                evidence_chunk_ids=evidence_chunk_ids,
                evidence_sentence=evidence_sentence,
                confidence=confidence,
            )
            current = facts.get(fact_id)
            if current is None or candidate.confidence > current.confidence:
                facts[fact_id] = candidate
    return list(facts.values())


def _request_facts(batch, entities, client, system_prompt):
    chunk_ids = {chunk.chunk_id for chunk in batch}
    request_ids = {
        chunk.chunk_id: f"c{index}"
        for index, chunk in enumerate(batch)
    }
    batch_entities = [
        entity
        for entity in entities
        if any(
            mention.split(":", 1)[0] in chunk_ids
            for mention in entity.source_mentions
        )
    ]
    known_entities = [
        {
            "entity_id": entity.entity_id,
            "canonical_name": entity.canonical_name,
            "aliases": entity.aliases,
            "entity_type": entity.entity_type,
            "mentions": {
                request_ids[chunk.chunk_id]: _chunk_surfaces(
                    entity, chunk.chunk_id
                )
                for chunk in batch
                if _chunk_surfaces(entity, chunk.chunk_id)
            },
        }
        for entity in batch_entities
    ]
    prompt_payload = (
        {
            "chunk_id": request_ids[batch[0].chunk_id],
            "text": batch[0].text,
            "known_entities": [
                {
                    **item,
                    "mentions": item["mentions"].get(
                        request_ids[batch[0].chunk_id], []
                    ),
                }
                for item in known_entities
            ],
        }
        if len(batch) == 1
        else {
            "chunks": [
                {
                    "chunk_id": request_ids[chunk.chunk_id],
                    "text": chunk.text,
                }
                for chunk in batch
            ],
            "known_entities": known_entities,
        }
    )
    prompt = json.dumps(prompt_payload, ensure_ascii=False)
    if isinstance(client, DeepSeekClient):
        return client.complete_json(
            system_prompt,
            prompt,
            cache_namespace="s2rag.fact_extraction",
            max_tokens=client.settings.deepseek_extraction_max_tokens,
        )
    return client.complete_json(system_prompt, prompt)


def _grounded_batch_evidence(candidate, chunks, declared_ids):
    declared = (
        {str(item) for item in declared_ids}
        if isinstance(declared_ids, list)
        else set()
    )
    ordered = [
        chunk for chunk in chunks
        if not declared or chunk.chunk_id in declared
    ]
    ordered.extend(chunk for chunk in chunks if chunk not in ordered)
    for chunk in ordered:
        grounded = _grounded_evidence(candidate, chunk.text)
        if grounded is not None:
            return grounded, [chunk.chunk_id]
    return None


def _arguments(raw_arguments, by_id: dict[str, Entity]) -> list[Argument]:
    arguments: list[Argument] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(raw_arguments, list):
        return arguments
    for raw in raw_arguments:
        if not isinstance(raw, dict):
            continue
        role = _role(raw.get("role"))
        entity_id = str(raw.get("entity_id", "")).strip()
        key = (role, entity_id)
        if role and entity_id in by_id and key not in seen:
            seen.add(key)
            arguments.append(Argument(role=role, entity_id=entity_id))
    return arguments


def _grounded_evidence(candidate: str, chunk_text: str) -> str | None:
    if not candidate:
        return None
    if candidate in chunk_text:
        return candidate
    parts = re.split(r"\s+", candidate.strip())
    match = re.search(r"\s+".join(re.escape(part) for part in parts), chunk_text)
    return match.group(0) if match else None


def _confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, confidence))


def _chunk_surfaces(entity: Entity, chunk_id: str) -> list[str]:
    prefix = f"{chunk_id}:"
    surfaces = [
        mention.split(":", 2)[2]
        for mention in entity.source_mentions
        if mention.startswith(prefix) and mention.count(":") >= 2
    ]
    return list(dict.fromkeys(surfaces))


def _compact_text(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _role(value) -> str:
    normalized = re.sub(r"[^\w]+", "_", _compact_text(value).casefold())
    return normalized.strip("_")


def _qualifiers(value) -> dict[str, str | float | None]:
    if not isinstance(value, dict):
        return {}
    qualifiers: dict[str, str | float | None] = {}
    for raw_key, raw_value in value.items():
        key = _role(raw_key)
        if not key:
            continue
        if raw_value is None:
            qualifiers[key] = None
        elif isinstance(raw_value, bool):
            qualifiers[key] = str(raw_value).casefold()
        elif isinstance(raw_value, (int, float)):
            qualifiers[key] = float(raw_value)
        elif isinstance(raw_value, str) and (text := _compact_text(raw_value)):
            qualifiers[key] = text
    return qualifiers
