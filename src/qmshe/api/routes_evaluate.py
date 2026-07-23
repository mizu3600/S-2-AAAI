from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from qmshe.api.dependencies import get_active_pipeline
from qmshe.evaluation.retrieval_metrics import recall_at_k
from qmshe.graph_pipeline import QMSGEGraphPipeline

router = APIRouter(prefix="/v1/evaluate", tags=["evaluation"])
_runs: dict[str, dict] = {}


@router.post("/run")
def run_evaluation(pipeline: QMSGEGraphPipeline = Depends(get_active_pipeline)) -> dict:
    run_id = f"eval_{uuid4().hex[:12]}"
    result = pipeline.query("How does PEAI improve Voc?", top_k=20, return_debug=False)
    gold = {"fact_1", "fact_2", "fact_3"} & {
        fact.hyperedge_id for fact in pipeline.corpus.evidence_hyperedges
    }
    metrics = {"supporting_fact_recall@20": recall_at_k(result.retrieved_facts, gold, 20)}
    _runs[run_id] = {
        "run_id": run_id,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    return _runs[run_id]


@router.get("/{run_id}")
def get_evaluation(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="evaluation run not found")
    return _runs[run_id]
