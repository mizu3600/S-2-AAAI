from qmshe.ingest.schemas import SemanticHyperedge
from qmshe.retrieval.evidence_verifier import verify_candidates
from qmshe.synthetic import make_synthetic_corpus


def test_semantic_edge_cannot_be_evidence():
    corpus = make_synthetic_corpus()
    corpus.semantic_hyperedges.append(
        SemanticHyperedge(semantic_edge_id="sem_1", member_ids=["fact_1", "fact_2"], topic="passivation", confidence=0.9)
    )
    result = verify_candidates(["sem_1", "fact_1"], corpus)
    assert result.accepted_ids == ["fact_1"]
    assert "retrieval-only" in result.rejected["sem_1"]

