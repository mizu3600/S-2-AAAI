import json

from s2rag.ingest.service import ingest_document
from s2rag.pipeline import load_corpus, save_corpus


class FakeExtractionClient:
    def __init__(self):
        self.calls = []

    def complete_json(self, system: str, prompt: str) -> dict:
        payload = json.loads(prompt)
        self.calls.append((system, payload))
        if "key entities" in system:
            return {
                "entities": [
                    {
                        "canonical_name": "request cache",
                        "aliases": ["cache"],
                        "entity_type": "software_component",
                        "description": "Stored request results.",
                        "mention": "cache",
                    },
                    {
                        "canonical_name": "response latency",
                        "aliases": ["latency"],
                        "entity_type": "performance_metric",
                        "description": "Time required to return a response.",
                        "mention": "response latency",
                    },
                ]
            }
        entities = {
            item["canonical_name"]: item["entity_id"]
            for item in payload["known_entities"]
        }
        return {
            "facts": [
                {
                    "predicate": "improves",
                    "arguments": [
                        {
                            "role": "component",
                            "entity_id": entities["request cache"],
                        },
                        {
                            "role": "result",
                            "entity_id": entities["response latency"],
                        },
                    ],
                    "qualifiers": {},
                    "evidence_sentence": payload["text"],
                    "confidence": 0.93,
                },
                {
                    "predicate": "invented_relation",
                    "arguments": [
                        {"role": "source", "entity_id": "ent_not_declared"},
                        {
                            "role": "result",
                            "entity_id": entities["response latency"],
                        },
                    ],
                    "qualifiers": {},
                    "evidence_sentence": payload["text"],
                    "confidence": 0.99,
                },
                {
                    "predicate": "unsupported_claim",
                    "arguments": [
                        {
                            "role": "component",
                            "entity_id": entities["request cache"],
                        },
                        {
                            "role": "result",
                            "entity_id": entities["response latency"],
                        },
                    ],
                    "qualifiers": {},
                    "evidence_sentence": "This sentence does not occur in the chunk.",
                    "confidence": 0.99,
                },
            ]
        }


def test_ingest_markdown_to_corpus(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Results\n"
        "A request cache reduces backend load and improves response latency.",
        encoding="utf-8",
    )

    client = FakeExtractionClient()
    corpus = ingest_document(source, client=client)

    assert corpus.documents
    assert corpus.documents[0].domain == "general"
    assert corpus.chunks
    assert len(corpus.entities) == 2
    assert len(corpus.evidence_hyperedges) == 1
    assert len(client.calls) == 2
    assert {
        argument.entity_id
        for argument in corpus.evidence_hyperedges[0].arguments
    } == {entity.entity_id for entity in corpus.entities}

    output = tmp_path / "corpus.json"
    save_corpus(corpus, output)
    assert load_corpus(output) == corpus
