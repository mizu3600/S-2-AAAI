import pytest

from qmshe.domain.annotation import PSCBenchmarkStore, split_by_document
from qmshe.domain.psc import PSCBenchmarkItem, normalize_measurement
from qmshe.synthetic import make_synthetic_corpus


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [("Voc increased by 120 mV", 0.12, "V"), ("PCE was 25 %", 0.25, "fraction"),
     ("aged for 10 h", 36000, "s")],
)
def test_unit_normalization(text, value, unit):
    measurement = normalize_measurement(text)
    assert measurement is not None
    assert measurement.value == pytest.approx(value)
    assert measurement.unit == unit


def test_psc_benchmark_store_validates_provenance(tmp_path):
    corpus = make_synthetic_corpus()
    item = PSCBenchmarkItem(
        question_id="q1", question="How does PEAI improve Voc?", answer="Through passivation",
        supporting_chunk_ids=["chunk_1"], supporting_hyperedge_ids=["fact_1"],
        bridge_entity_ids=["ent_passivation"], gold_path=["fact_1", "ent_passivation"],
        hop_count=2, query_type="mechanism_chain_2_4_hop", source_document_ids=["doc_synthetic"],
    )
    store = PSCBenchmarkStore(tmp_path / "psc.jsonl")
    store.upsert(item, corpus)
    assert store.load()[0].question_id == "q1"
    train, test = split_by_document(store.load())
    assert len(train) + len(test) == 1

