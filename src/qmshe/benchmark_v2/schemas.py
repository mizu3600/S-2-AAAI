from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class V2Passage:
    passage_id: str
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class V2Example:
    example_id: str
    question: str
    answer: str
    passages: tuple[V2Passage, ...]
    supporting_facts: frozenset[tuple[str, int]]
    query_type: str
    level: str


@dataclass(frozen=True)
class StructuredFact:
    fact_id: str
    passage_id: str
    sentence_index: int
    text: str
    subject: str
    predicate: str
    object: str
    roles: tuple[tuple[str, str], ...]
    qualifiers: tuple[tuple[str, str], ...] = ()
    entity_ids: tuple[str, ...] = ()

    def role_values(self, role: str) -> tuple[str, ...]:
        return tuple(value for name, value in self.roles if name == role)


@dataclass(frozen=True)
class CandidateView:
    example: V2Example
    passages: tuple[V2Passage, ...]
    facts: tuple[StructuredFact, ...]
    gold_fact_ids: frozenset[str]
    gold_passage_ids: frozenset[str]
    candidate_count: int


@dataclass(frozen=True)
class RankingPrediction:
    system: str
    fact_ranking: tuple[str, ...]
    passage_ranking: tuple[str, ...]
    path: tuple[str, ...]
    answer: str = ""
    citations: tuple[str, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)
