"""Independent, protocol-first benchmark chain for QMSxE-RAG.

This package deliberately does not import or mutate the production retrieval
pipelines.  It gives every topology the same fact candidates, budget, and
reranker contract, then evaluates their internal fact rankings.
"""

from qmshe.benchmark_v2.dataset import build_candidate_view, load_hotpot_dev
from qmshe.benchmark_v2.evaluator import evaluate_prediction
from qmshe.benchmark_v2.systems import ControlledTopologyRetriever

__all__ = [
    "ControlledTopologyRetriever",
    "build_candidate_view",
    "evaluate_prediction",
    "load_hotpot_dev",
]
