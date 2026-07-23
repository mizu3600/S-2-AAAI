from pydantic import BaseModel, Field, model_validator


class Passage(BaseModel):
    passage_id: str
    title: str
    sentences: list[str]
    source_uri: str | None = None


class SupportingFact(BaseModel):
    passage_id: str
    sentence_index: int = Field(ge=0)


class BenchmarkExample(BaseModel):
    example_id: str
    question: str
    answer: str | list[str]
    passages: list[Passage]
    supporting_facts: list[SupportingFact] = Field(default_factory=list)
    bridge_entities: list[str] = Field(default_factory=list)
    gold_path: list[str] = Field(default_factory=list)
    hop_count: int = Field(default=1, ge=1)
    query_type: str = "unknown"
    dataset: str
    split: str = "validation"
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def supporting_facts_reference_passages(self):
        passage_ids = {passage.passage_id for passage in self.passages}
        missing = {fact.passage_id for fact in self.supporting_facts} - passage_ids
        if missing:
            raise ValueError(f"supporting facts reference missing passages: {sorted(missing)}")
        return self


class BenchmarkSuite(BaseModel):
    name: str
    split: str
    examples: list[BenchmarkExample]
    source: str
    version: str = "v1"

