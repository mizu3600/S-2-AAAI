from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def __init__(self):
        self.encoder = DeterministicEmbedder(64)

    def encode(self, texts):
        return self.encoder.embed(texts)


def test_end_to_end_query_returns_traceable_evidence():
    pipeline = QMSHEPipeline(make_synthetic_corpus(), text_encoder=LocalEncoder())
    assert pipeline.use_graph_rerank is False
    result = pipeline.query(
        "How does PEAI improve Voc in inverted PSCs?", top_k=4, return_debug=True
    )
    assert result.retrieved_hyperedges
    assert all(item.startswith("fact_") for item in result.retrieved_hyperedges)
    assert result.citations
    assert abs(sum(result.band_weights.values()) - 1.0) < 1e-6
    assert abs(sum(result.relation_weights.values()) - 1.0) < 1e-6
    assert result.evidence_path


def test_runtime_ablation_switches_preserve_a_runnable_pipeline():
    pipeline = QMSHEPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        enable_remote_reranker=False,
        dense_only=True,
        use_bm25=False,
        use_graph_rerank=False,
    )
    pipeline.generator.client = None
    result = pipeline.query("Which material has high conductivity?", top_k=2)
    assert result.retrieved_hyperedges


def test_graph_rerank_remains_an_explicit_opt_in():
    pipeline = QMSHEPipeline(
        make_synthetic_corpus(), text_encoder=LocalEncoder(), use_graph_rerank=True
    )

    assert pipeline.use_graph_rerank is True
