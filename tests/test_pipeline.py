import hashlib

import numpy as np

from s2rag.pipeline import S2RAGPipeline
from s2rag.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        matrix = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                index = (
                    int.from_bytes(
                        hashlib.blake2b(token.encode(), digest_size=8).digest(),
                        "little",
                    )
                    % matrix.shape[1]
                )
                matrix[row, index] += 1
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class FakeGenerator:
    def generate(self, question, context):
        return "Generated from evidence."


def test_single_pipeline_builds_retrieves_and_generates():
    pipeline = S2RAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
    )

    result = pipeline.query(
        "How does request caching improve response latency?",
        top_k=4,
        return_debug=True,
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
    pipeline = S2RAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
    )

    try:
        pipeline.query("question", top_k=5, candidate_count=4)
    except ValueError as exc:
        assert "candidate_count" in str(exc)
    else:
        raise AssertionError("expected candidate_count validation")


def test_training_free_graph_encoder_is_seed_independent():
    first = S2RAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
        seed=1,
    )
    second = S2RAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
        seed=999,
    )

    first_facts = first.retrieve_fact_candidates(
        "How does caching help?",
        per_channel_k=8,
        candidate_count=8,
    ).facts
    second_facts = second.retrieve_fact_candidates(
        "How does caching help?",
        per_channel_k=8,
        candidate_count=8,
    ).facts

    assert list(first.model.parameters()) == []
    assert first_facts == second_facts
