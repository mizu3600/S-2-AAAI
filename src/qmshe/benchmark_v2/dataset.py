from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from qmshe.benchmark_v2.extraction import StructuredFactExtractor
from qmshe.benchmark_v2.schemas import CandidateView, V2Example, V2Passage


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "item"


def load_hotpot_dev(path: str | Path, limit: int | None = None) -> list[V2Example]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    examples = []
    for row in rows:
        example_id = str(row["_id"])
        passages = tuple(
            V2Passage(
                passage_id=f"{_slug(example_id)}_p{index}",
                title=str(title),
                sentences=tuple(str(sentence) for sentence in sentences),
            )
            for index, (title, sentences) in enumerate(row["context"])
        )
        title_to_id = {passage.title: passage.passage_id for passage in passages}
        support = frozenset(
            (title_to_id[title], int(sentence_index))
            for title, sentence_index in row["supporting_facts"]
            if title in title_to_id
        )
        examples.append(
            V2Example(
                example_id=example_id,
                question=str(row["question"]),
                answer=str(row["answer"]),
                passages=passages,
                supporting_facts=support,
                query_type=str(row.get("type", "unknown")),
                level=str(row.get("level", "unknown")),
            )
        )
    return examples


def build_candidate_view(
    example: V2Example,
    corpus_passages: tuple[V2Passage, ...],
    candidate_count: int,
    extractor: StructuredFactExtractor | None = None,
    seed: int = 42,
) -> CandidateView:
    """Build a deterministic 10/100/1000-passage view without leaking labels.

    The ten official passages are kept. Extra distractors are sampled from the
    rest of the dev corpus by an example-specific seed; gold labels never take
    part in sampling or ordering.
    """
    if candidate_count < len(example.passages):
        raise ValueError("candidate_count cannot discard official HotpotQA passages")
    own_titles = {passage.title.casefold() for passage in example.passages}
    pool = [p for p in corpus_passages if p.title.casefold() not in own_titles]
    stable = int(hashlib.sha1(example.example_id.encode()).hexdigest()[:12], 16)
    rng = random.Random(seed ^ stable)
    needed = candidate_count - len(example.passages)
    if needed > len(pool):
        raise ValueError(f"requested {candidate_count} passages but corpus has only {len(pool)} distractors")
    distractors = rng.sample(pool, needed)
    passages = tuple(example.passages) + tuple(distractors)
    fact_extractor = extractor or StructuredFactExtractor()
    facts = tuple(
        fact
        for passage in passages
        for fact in fact_extractor.extract_passage(passage)
    )
    gold_fact_ids = frozenset(
        fact.fact_id
        for fact in facts
        if (fact.passage_id, fact.sentence_index) in example.supporting_facts
    )
    return CandidateView(
        example=example,
        passages=passages,
        facts=facts,
        gold_fact_ids=gold_fact_ids,
        gold_passage_ids=frozenset(passage_id for passage_id, _ in example.supporting_facts),
        candidate_count=candidate_count,
    )


def global_passage_pool(examples: list[V2Example]) -> tuple[V2Passage, ...]:
    """Deduplicate dev passages by normalized title and exact sentence content."""
    output = {}
    for example in examples:
        for passage in example.passages:
            key = (passage.title.casefold(), passage.sentences)
            output.setdefault(key, passage)
    return tuple(output.values())
