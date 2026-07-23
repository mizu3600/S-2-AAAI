from typing import Literal

from pydantic import BaseModel, Field


class OrdinaryGraphNode(BaseModel):
    node_id: str
    node_type: Literal["entity", "fact"]
    text: str
    source_fact_id: str | None = None


class OrdinaryGraphEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source_role: str | None = None
    target_role: str | None = None
    evidence_fact_ids: list[str] = Field(default_factory=list)
    weight: float = Field(gt=0)
