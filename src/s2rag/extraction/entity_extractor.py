import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

from s2rag.extraction.canonicalizer import normalize_name
from s2rag.extraction.batching import passage_batches
from s2rag.ingest.schemas import Chunk, Entity
from s2rag.providers import DeepSeekClient
from s2rag.settings import get_settings


ENTITY_SYSTEM_PROMPT = """You are the open-domain entity extraction stage of an OpenIE system.
Extract reusable entities and concepts that are explicitly mentioned in the supplied source text.
Do not rely on a fixed domain ontology. A useful entity is anything that can serve as an argument
of a factual relation, including people, organizations, places, works, products, events, methods,
systems, scientific or technical concepts, conditions, dates, quantities, measurements, and
specific descriptive concepts. Do not emit isolated pronouns, generic filler words, or concepts
that are only implied.

Return one JSON object with key entities. Each entity must contain:
- canonical_name: the clearest source-grounded name for the entity;
- aliases: other exact surface forms in the supplied text;
- entity_type: a concise, specific lower_snake_case type chosen for this text, not from a fixed list;
- description: a short description supported only by the supplied text;
- mentions: exact source mentions as [{"chunk_id": supplied_chunk_id, "surface": exact text}].
  For backward compatibility, a single-chunk request may use mention instead.

Prefer complete, meaningful spans over fragmented tokens. Merge aliases that clearly denote the same
entity, but keep ambiguous entities separate. Dates, quantities, and relation-bearing common-noun
phrases should be included when a fact would lose important information without them. Never add
outside knowledge or an entity that has no exact surface form in the text.
Do not hide an argument-worthy category or value inside a later predicate: extract phrases such as
"private FM radio station" or "standard atmospheric pressure" as entities. When the text gives a
coordinated list of distinct values, extract each value separately rather than only the whole list.

Example source:
"In 2021, the Atlas team evaluated S2RAG on HotpotQA with 500 questions. It improved recall by 8%
under a zero-shot setting."
Example output:
{"entities": [
  {"canonical_name": "2021", "aliases": [], "entity_type": "date",
   "description": "The evaluation year.", "mention": "2021"},
  {"canonical_name": "Atlas team", "aliases": [], "entity_type": "research_team",
   "description": "The team that performed the evaluation.", "mention": "Atlas team"},
  {"canonical_name": "S2RAG", "aliases": [], "entity_type": "system",
   "description": "The system that was evaluated.", "mention": "S2RAG"},
  {"canonical_name": "HotpotQA", "aliases": [], "entity_type": "benchmark",
   "description": "The benchmark used for evaluation.", "mention": "HotpotQA"},
  {"canonical_name": "500 questions", "aliases": [], "entity_type": "sample_size",
   "description": "The evaluation sample size.", "mention": "500 questions"},
  {"canonical_name": "recall", "aliases": [], "entity_type": "evaluation_metric",
   "description": "The metric reported as improved.", "mention": "recall"},
  {"canonical_name": "8%", "aliases": [], "entity_type": "relative_change",
   "description": "The reported recall improvement.", "mention": "8%"},
  {"canonical_name": "zero-shot setting", "aliases": [], "entity_type": "evaluation_setting",
   "description": "The setting in which recall improved.", "mention": "zero-shot setting"}
]}"""


def extract_entities_with_llm(
    chunks: list[Chunk],
    client: DeepSeekClient,
    *,
    system_prompt: str = ENTITY_SYSTEM_PROMPT,
    batch_max_chars: int | None = None,
    max_workers: int | None = None,
) -> list[Entity]:
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
                    _request_entities(batch, client, system_prompt),
                ),
                batches,
            )
        )
    entities: dict[tuple[str, str], Entity] = {}
    for batch, payload in responses:
        chunks_by_id = {
            identifier: chunk
            for index, chunk in enumerate(batch)
            for identifier in (chunk.chunk_id, f"c{index}")
        }
        for raw in payload.get("entities", []):
            if not isinstance(raw, dict):
                continue
            canonical_name = str(raw.get("canonical_name", "")).strip()
            entity_type = _entity_type(raw.get("entity_type"))
            if not canonical_name:
                continue

            aliases = _string_list(raw.get("aliases"))
            mention = str(raw.get("mention", "")).strip()
            surfaces = list(dict.fromkeys([mention, *aliases, canonical_name]))
            matched = [
                (chunk.chunk_id, surface, offset)
                for chunk in batch
                for surface in surfaces
                for offset in _surface_offsets(surface, chunk.text)
            ]
            for item in raw.get("mentions", []):
                if not isinstance(item, dict):
                    continue
                chunk = chunks_by_id.get(str(item.get("chunk_id", "")))
                surface = str(item.get("surface", "")).strip()
                if chunk is None:
                    continue
                matched.extend(
                    (chunk.chunk_id, surface, offset)
                    for offset in _surface_offsets(surface, chunk.text)
                )
            matched = list(dict.fromkeys(matched))
            if not matched:
                continue

            key = (normalize_name(canonical_name), entity_type)
            if key not in entities:
                digest = hashlib.sha1(f"{entity_type}:{key[0]}".encode()).hexdigest()[:12]
                entities[key] = Entity(
                    entity_id=f"ent_{digest}",
                    canonical_name=canonical_name,
                    aliases=[],
                    entity_type=entity_type,
                    description=str(raw.get("description", "")).strip(),
                    source_mentions=[],
                )
            entity = entities[key]
            entity.aliases = sorted(
                set(entity.aliases)
                | {
                    surface
                    for _, surface, _ in matched
                    if normalize_name(surface) != key[0]
                }
            )
            entity.source_mentions = sorted(
                set(entity.source_mentions)
                | {
                    f"{chunk_id}:{offset}:{surface}"
                    for chunk_id, surface, offset in matched
                }
            )
            if not entity.description:
                entity.description = str(raw.get("description", "")).strip()
    return list(entities.values())


def _request_entities(batch, client, system_prompt):
    request_chunks = [
        {"chunk_id": f"c{index}", "text": chunk.text}
        for index, chunk in enumerate(batch)
    ]
    prompt_payload = (
        request_chunks[0]
        if len(batch) == 1
        else {"chunks": request_chunks}
    )
    prompt = json.dumps(prompt_payload, ensure_ascii=False)
    if isinstance(client, DeepSeekClient):
        return client.complete_json(
            system_prompt,
            prompt,
            cache_namespace="s2rag.entity_extraction",
            max_tokens=client.settings.deepseek_extraction_max_tokens,
        )
    return client.complete_json(system_prompt, prompt)


def _string_list(value) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _entity_type(value) -> str:
    normalized = re.sub(r"[^\w]+", "_", str(value or "").strip().casefold())
    return normalized.strip("_") or "entity"


def _surface_offsets(surface: str, text: str) -> list[int]:
    if not surface:
        return []
    return [match.start() for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE)]
