from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s2rag.embedding.text_encoder import LocalBGEEncoder
from s2rag.providers import ProviderError
from s2rag.retrieval.local_reranker import LocalBGEReranker


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return np.asarray(
            [[float(len(text)), 1.0] for text in texts],
            dtype=np.float32,
        )


class FakeTokenizer:
    def __call__(self, queries, documents, **kwargs):
        assert len(queries) == len(documents)
        return {
            "input_ids": torch.tensor(
                [[len(document)] for document in documents],
                dtype=torch.long,
            )
        }


class FakeRerankerModel:
    def __call__(self, input_ids):
        return SimpleNamespace(logits=input_ids.float())


def test_local_bge_encoder_uses_injected_local_model():
    encoder = LocalBGEEncoder(model=FakeEmbeddingModel(), device="cpu")

    vectors = encoder.encode(["short", "longer"])

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32


def test_local_bge_reranker_orders_model_scores():
    reranker = LocalBGEReranker(
        tokenizer=FakeTokenizer(),
        model=FakeRerankerModel(),
        device="cpu",
        batch_size=2,
    )

    assert reranker.rank("query", ["medium", "x", "longest document"]) == [2, 0, 1]


def test_local_models_do_not_fall_back_to_network(tmp_path):
    missing_embedding = tmp_path / "missing-bge-m3"
    missing_reranker = tmp_path / "missing-reranker"

    with pytest.raises(ProviderError, match="does not exist"):
        LocalBGEEncoder(model_path=missing_embedding, device="cpu").encode(["query"])
    with pytest.raises(ProviderError, match="does not exist"):
        LocalBGEReranker(model_path=missing_reranker, device="cpu").rank("query", ["document"])


@pytest.mark.parametrize(
    ("factory", "invoke"),
    [
        (
            lambda path: LocalBGEEncoder(model_path=path, device="cpu"),
            lambda model: model.encode(["query"]),
        ),
        (
            lambda path: LocalBGEReranker(model_path=path, device="cpu"),
            lambda model: model.rank("query", ["document"]),
        ),
    ],
)
def test_local_models_reject_peft_adapter_directories(tmp_path, factory, invoke):
    adapter_directory = tmp_path / "adapter"
    adapter_directory.mkdir()
    (adapter_directory / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}',
        encoding="utf-8",
    )

    with pytest.raises(ProviderError, match="original base-model directory"):
        invoke(factory(adapter_directory))
