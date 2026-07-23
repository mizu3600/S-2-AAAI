from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from qmshe.api.dependencies import get_graph_pipeline, get_pipeline
from qmshe.graph.ordinary import GraphProfile

router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=12, ge=1, le=100)
    return_debug: bool = False
    mode: Literal["hypergraph", "graph"] = "graph"
    graph_profile: GraphProfile = GraphProfile.REIFIED_FACT


@router.post("/query")
def query(request: QueryRequest) -> dict:
    if request.mode == "graph":
        return get_graph_pipeline(request.graph_profile).query(
            request.question, request.top_k, request.return_debug
        ).__dict__
    return get_pipeline().query(request.question, request.top_k, request.return_debug).__dict__


@router.get("/metrics")
def metrics(
    mode: Literal["hypergraph", "graph"] = "graph",
    graph_profile: GraphProfile = GraphProfile.REIFIED_FACT,
) -> dict:
    if mode == "graph":
        return get_graph_pipeline(graph_profile).metrics.snapshot()
    return get_pipeline().metrics.snapshot()
