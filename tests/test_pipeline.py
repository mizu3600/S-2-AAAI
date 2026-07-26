from qmshe.pipeline import QMSHERAGPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def test_single_pipeline_builds_retrieves_and_generates():
    pipeline = QMSHERAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        enable_remote_reranker=False,
    )
    pipeline.generator.client = None

    result = pipeline.query(
        "How does PEAI improve Voc?", top_k=4, return_debug=True
    )

    assert pipeline.artifacts.graph.graph["mode"] == "reified_fact"
    assert result.retrieved_nodes
    assert result.retrieved_facts
    assert result.citations
    assert result.answer
    assert set(result.band_weights) == {"raw", "low", "mid", "high"}
    assert abs(sum(result.band_weights.values()) - 1.0) < 1e-6
    assert result.scores


def test_candidate_count_must_cover_top_k():
    pipeline = QMSHERAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        enable_remote_reranker=False,
    )

    try:
        pipeline.query("question", top_k=5, candidate_count=4)
    except ValueError as exc:
        assert "candidate_count" in str(exc)
    else:
        raise AssertionError("expected candidate_count validation")
