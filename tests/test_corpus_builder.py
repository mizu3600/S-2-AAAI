from s2rag.benchmarks.corpus_builder import build_example_corpus
from s2rag.benchmarks.schemas import BenchmarkExample, Passage, SupportingFact


class ExtractionClient:
    def complete_json(self, system, prompt):
        import json

        payload = json.loads(prompt)
        if "known_entities" not in payload:
            return {
                "entities": [
                    {
                        "canonical_name": "Ada",
                        "aliases": [],
                        "entity_type": "person",
                        "description": "The winner.",
                        "mention": "Ada",
                    },
                    {
                        "canonical_name": "contest",
                        "aliases": [],
                        "entity_type": "event",
                        "description": "The event Ada won.",
                        "mention": "contest",
                    },
                ]
            }
        by_name = {item["canonical_name"]: item["entity_id"] for item in payload["known_entities"]}
        return {
            "facts": [
                {
                    "predicate": "won",
                    "arguments": [
                        {"role": "winner", "entity_id": by_name["Ada"]},
                        {"role": "event", "entity_id": by_name["contest"]},
                    ],
                    "qualifiers": {},
                    "evidence_sentence": "Ada won the contest.",
                    "confidence": 1.0,
                }
            ]
        }


def test_llm_corpus_maps_grounded_facts_to_official_hotpot_evidence():
    example = BenchmarkExample(
        example_id="llm-corpus",
        question="Who won?",
        answer="Ada",
        passages=[
            Passage(
                passage_id="p1",
                title="Result",
                sentences=["Ada won the contest."],
            )
        ],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )

    built = build_example_corpus(example, ExtractionClient())

    assert len(built.corpus.entities) == 2
    assert len(built.corpus.evidence_hyperedges) == 1
    assert built.gold_fact_ids == {built.corpus.evidence_hyperedges[0].hyperedge_id}
    assert built.gold_passage_ids == {"p1"}
    assert built.extraction_coverage == 1.0
