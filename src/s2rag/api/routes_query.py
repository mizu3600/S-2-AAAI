from fastapi import APIRouter
from pydantic import BaseModel, Field

from s2rag.api.dependencies import get_pipeline


router = APIRouter(prefix="/v1", tags=["pipeline"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=12, ge=1, le=100)
    return_debug: bool = False


@router.post("/query")
def query(request: QueryRequest) -> dict:
    return get_pipeline().query(
        request.question, request.top_k, request.return_debug
    ).__dict__


@router.get("/metrics")
def metrics() -> dict:
    return get_pipeline().metrics.snapshot()
