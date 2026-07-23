import re
import unicodedata

from qmshe.ingest.schemas import Entity


def normalize_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).casefold()
    value = re.sub(r"[^\w\s+-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonicalize_entities(entities: list[Entity]) -> list[Entity]:
    """Conservative exact alias merge. Ambiguous semantic candidates remain separate."""
    merged: dict[tuple[str, str], Entity] = {}
    for entity in entities:
        key = (normalize_name(entity.canonical_name), entity.entity_type)
        if key not in merged:
            merged[key] = entity.model_copy(deep=True)
            continue
        current = merged[key]
        current.aliases = sorted(set(current.aliases + entity.aliases))
        current.source_mentions = sorted(set(current.source_mentions + entity.source_mentions))
    return list(merged.values())

