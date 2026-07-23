from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalFact:
    fact_id: str
    document_id: str
    text: str
    sentence: str


@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class CanonicalExample:
    example_id: str
    question: str
    answer: str
    documents: tuple[CanonicalDocument, ...]
    facts: tuple[CanonicalFact, ...]
    gold_fact_ids: frozenset[str]

    @property
    def gold_document_ids(self) -> frozenset[str]:
        fact_to_document = {fact.fact_id: fact.document_id for fact in self.facts}
        return frozenset(fact_to_document[item] for item in self.gold_fact_ids)


@dataclass
class UsageTrace:
    llm_calls: int = 0
    embedding_calls: int = 0
    reranker_calls: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    embedding_tokens: int | None = 0
    api_cost_usd: float | None = 0.0
    retry_count: int = 0
    token_count_mode: str = "measured"


@dataclass
class TimingTrace:
    index_seconds: float | None = None
    retrieval_seconds: float | None = None
    generation_seconds: float | None = None
    total_seconds: float | None = None
    timing_scope: str = "query"


@dataclass
class StandardTrace:
    system: str
    example_id: str
    status: str
    document_ranking: list[str] = field(default_factory=list)
    text_unit_ranking: list[str] = field(default_factory=list)
    fact_ranking: list[str] = field(default_factory=list)
    native_paths: list[list[str]] = field(default_factory=list)
    induced_path: list[str] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    citation_level: str = "document"
    ranking_origin: str = "unknown"
    path_origin: str = "unavailable"
    usage: UsageTrace = field(default_factory=UsageTrace)
    timing: TimingTrace = field(default_factory=TimingTrace)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
