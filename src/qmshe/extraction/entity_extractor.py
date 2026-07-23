import hashlib
import re

from qmshe.ingest.schemas import Chunk, Entity

ENTITY_PATTERNS = {
    "passivation_material": [r"\bPEAI\b", r"phenethyl\s*ammonium iodide"],
    "performance_metric": [r"\bVoc\b", r"open[- ]circuit voltage"],
    "mechanism": [r"non[- ]radiative recombination", r"surface (?:defect )?passivation"],
    "device_architecture": [r"inverted (?:perovskite )?solar cells?", r"inverted PSCs?"],
}


def extract_entities_rule_based(chunks: list[Chunk]) -> list[Entity]:
    grouped: dict[tuple[str, str], dict] = {}
    for chunk in chunks:
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, chunk.text, flags=re.IGNORECASE):
                    canonical = _canonical(match.group(), entity_type)
                    key = (canonical.lower(), entity_type)
                    item = grouped.setdefault(
                        key,
                        {"canonical": canonical, "aliases": set(), "mentions": [], "contexts": []},
                    )
                    item["aliases"].add(match.group())
                    item["mentions"].append(f"{chunk.chunk_id}:{match.start()}")
                    item["contexts"].append(chunk.text[:240])
    entities = []
    for (name, entity_type), item in sorted(grouped.items()):
        digest = hashlib.sha1(f"{entity_type}:{name}".encode()).hexdigest()[:12]
        entities.append(
            Entity(
                entity_id=f"ent_{digest}",
                canonical_name=item["canonical"],
                aliases=sorted(item["aliases"]),
                entity_type=entity_type,
                description=item["contexts"][0],
                source_mentions=item["mentions"],
            )
        )
    return entities


def _canonical(text: str, entity_type: str) -> str:
    lowered = text.lower()
    if entity_type == "passivation_material":
        return "phenethylammonium iodide"
    if entity_type == "performance_metric":
        return "open-circuit voltage"
    if "recombination" in lowered:
        return "non-radiative recombination"
    if "passivation" in lowered:
        return "surface defect passivation"
    if entity_type == "device_architecture":
        return "inverted perovskite solar cell"
    return lowered.strip()

