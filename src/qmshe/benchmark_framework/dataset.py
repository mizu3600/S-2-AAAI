from __future__ import annotations

import json
from pathlib import Path

from qmshe.benchmark_framework.schemas import (
    CanonicalDocument,
    CanonicalExample,
    CanonicalFact,
)


def load_canonical_examples(path: str | Path) -> list[CanonicalExample]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        CanonicalExample(
            example_id=row["example_id"],
            question=row["question"],
            answer=row["answer"],
            documents=tuple(CanonicalDocument(**document) for document in row["documents"]),
            facts=tuple(CanonicalFact(**fact) for fact in row["facts"]),
            gold_fact_ids=frozenset(row["gold_fact_ids"]),
        )
        for row in rows
    ]


def induce_fact_ranking(example: CanonicalExample, document_ranking: list[str]) -> list[str]:
    """Uniform fallback for systems that expose documents but not text units."""
    facts_by_document: dict[str, list[str]] = {}
    for fact in example.facts:
        facts_by_document.setdefault(fact.document_id, []).append(fact.fact_id)
    return [
        fact_id
        for document_id in document_ranking
        for fact_id in facts_by_document.get(document_id, [])
    ]


def map_text_units_to_facts(example: CanonicalExample, texts: list[str]) -> list[str]:
    ranking = []
    for text in texts:
        for fact in example.facts:
            if fact.fact_id not in ranking and (
                fact.sentence in text or text in fact.sentence or fact.text in text
            ):
                ranking.append(fact.fact_id)
    return ranking
