from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    document_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source_uri: str
    domain: str = "PSC"
    version: int = 1


class Section(BaseModel):
    title: str
    text: str
    page: int | None = None
    start_char: int = 0


class ParsedDocument(BaseModel):
    document: Document
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    section: str
    text: str
    start_char: int
    end_char: int
    page: int | None = None


class Mention(BaseModel):
    mention_id: str
    chunk_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str
    description: str = ""
    source_mentions: list[str] = Field(default_factory=list)


class Argument(BaseModel):
    role: str
    entity_id: str


class EvidenceHyperedge(BaseModel):
    hyperedge_id: str
    predicate: str
    arguments: list[Argument]
    qualifiers: dict[str, str | float | None] = Field(default_factory=dict)
    evidence_chunk_ids: list[str]
    evidence_sentence: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def require_distinct_arguments(self):
        if len({arg.entity_id for arg in self.arguments}) < 2:
            raise ValueError("evidence hyperedge must contain at least two distinct entities")
        return self


class SemanticHyperedge(BaseModel):
    semantic_edge_id: str
    member_ids: list[str]
    topic: str
    construction_method: str = "mutual_knn"
    confidence: float = Field(ge=0, le=1)
    evidence_status: Literal["retrieval_only"] = "retrieval_only"


class Corpus(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    evidence_hyperedges: list[EvidenceHyperedge] = Field(default_factory=list)
    semantic_hyperedges: list[SemanticHyperedge] = Field(default_factory=list)
    graph_version: str = "graph-v1"

