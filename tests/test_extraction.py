import json

import httpx

from s2rag.extraction.entity_extractor import (
    ENTITY_SYSTEM_PROMPT,
    extract_entities_with_llm,
)
from s2rag.extraction.fact_extractor import FACT_SYSTEM_PROMPT, extract_facts_with_llm
from s2rag.ingest.schemas import Chunk, Entity
from s2rag.providers import DeepSeekClient
from s2rag.settings import Settings


class StaticJsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system: str, prompt: str) -> dict:
        self.calls.append((system, json.loads(prompt)))
        return self.payload


def test_open_domain_entity_extraction_keeps_general_relation_arguments():
    chunk = Chunk(
        chunk_id="chunk_radio",
        document_id="doc_radio",
        section="History",
        text=("Radio City is a private FM radio station. Radio City started on 3 July 2001."),
        start_char=0,
        end_char=80,
    )
    client = StaticJsonClient(
        {
            "entities": [
                {
                    "canonical_name": "Radio City",
                    "aliases": [],
                    "entity_type": "Radio Station",
                    "description": "A private FM radio station.",
                    "mention": "Radio City",
                },
                {
                    "canonical_name": "private FM radio station",
                    "aliases": [],
                    "entity_type": "Descriptive Concept",
                    "description": "The stated category of Radio City.",
                    "mention": "private FM radio station",
                },
                {
                    "canonical_name": "3 July 2001",
                    "aliases": [],
                    "description": "The stated start date.",
                    "mention": "3 July 2001",
                },
                {
                    "canonical_name": "Atlantis",
                    "aliases": [],
                    "entity_type": "place",
                    "description": "Not present in the source.",
                    "mention": "Atlantis",
                },
            ]
        }
    )

    entities = extract_entities_with_llm([chunk], client)

    assert {entity.canonical_name for entity in entities} == {
        "Radio City",
        "private FM radio station",
        "3 July 2001",
    }
    by_name = {entity.canonical_name: entity for entity in entities}
    assert by_name["Radio City"].entity_type == "radio_station"
    assert by_name["private FM radio station"].entity_type == "descriptive_concept"
    assert by_name["3 July 2001"].entity_type == "entity"
    assert len(by_name["Radio City"].source_mentions) == 2
    assert client.calls[0][0] == ENTITY_SYSTEM_PROMPT


def test_entity_extraction_accepts_a_comparison_prompt_override():
    chunk = Chunk(
        chunk_id="chunk_override",
        document_id="doc_override",
        section="Test",
        text="S2RAG",
        start_char=0,
        end_char=5,
    )
    client = StaticJsonClient(
        {
            "entities": [
                {
                    "canonical_name": "S2RAG",
                    "aliases": [],
                    "entity_type": "system",
                    "description": "",
                    "mention": "S2RAG",
                }
            ]
        }
    )

    extract_entities_with_llm([chunk], client, system_prompt="comparison prompt")

    assert client.calls[0][0] == "comparison prompt"


def test_passage_extraction_cache_reuses_content_across_example_chunk_ids(tmp_path):
    request_count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        if system == ENTITY_SYSTEM_PROMPT:
            result = {
                "entities": [
                    {
                        "canonical_name": "Ada",
                        "aliases": [],
                        "entity_type": "person",
                        "description": "",
                        "mentions": [{"chunk_id": "c0", "surface": "Ada"}],
                    },
                    {
                        "canonical_name": "Paris",
                        "aliases": [],
                        "entity_type": "place",
                        "description": "",
                        "mentions": [{"chunk_id": "c0", "surface": "Paris"}],
                    },
                ]
            }
        else:
            assert system == FACT_SYSTEM_PROMPT
            result = {
                "facts": [
                    {
                        "predicate": "visited",
                        "arguments": [
                            {"role": "visitor", "entity_id": _stable_entity_id("person", "ada")},
                            {"role": "place", "entity_id": _stable_entity_id("place", "paris")},
                        ],
                        "qualifiers": {},
                        "evidence_sentence": "Ada visited Paris.",
                        "evidence_chunk_ids": ["c0"],
                        "confidence": 0.9,
                    }
                ]
            }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(result)}}
                ]
            },
        )

    settings = Settings(
        deepseek_api_key="test",
        deepseek_model="test-model",
        deepseek_response_cache_dir=tmp_path / "deepseek",
        extraction_batch_max_chars=1000,
        extraction_workers=1,
    )
    transport = httpx.MockTransport(respond)
    http_client = httpx.Client(
        base_url="https://example.test",
        transport=transport,
    )
    client = DeepSeekClient(settings, http_client=http_client)
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            section="Lead",
            text="Ada visited Paris.",
            start_char=0,
            end_char=18,
        )
        for chunk_id, document_id in (
            ("example_a_chunk", "example_a_passage"),
            ("example_b_chunk", "example_b_passage"),
        )
    ]

    first_entities = extract_entities_with_llm([chunks[0]], client)
    first_facts = extract_facts_with_llm([chunks[0]], first_entities, client)
    second_entities = extract_entities_with_llm([chunks[1]], client)
    second_facts = extract_facts_with_llm([chunks[1]], second_entities, client)

    assert request_count == 2
    assert all(
        mention.startswith("example_b_chunk:")
        for entity in second_entities
        for mention in entity.source_mentions
    )
    assert second_facts[0].evidence_chunk_ids == ["example_b_chunk"]
    assert first_facts[0].predicate == second_facts[0].predicate == "visited"


def test_conditioned_openie_keeps_nary_facts_and_rejects_ungrounded_output():
    text = "Radio City launched PlanetRadiocity.com in May 2008. It serves India."
    chunk = Chunk(
        chunk_id="chunk_launch",
        document_id="doc_radio",
        section="History",
        text=text,
        start_char=0,
        end_char=len(text),
    )
    entities = [
        _entity("ent_radio", "Radio City", "organization", "chunk_launch:0:Radio City"),
        _entity(
            "ent_portal",
            "PlanetRadiocity.com",
            "website",
            "chunk_launch:20:PlanetRadiocity.com",
        ),
        _entity("ent_date", "May 2008", "date", "chunk_launch:40:May 2008"),
        _entity("ent_india", "India", "country", "chunk_launch:61:India"),
    ]
    launch_fact = {
        "predicate": " launched ",
        "arguments": [
            {"role": "Launching Organization", "entity_id": "ent_radio"},
            {"role": "Product", "entity_id": "ent_portal"},
            {"role": "Time", "entity_id": "ent_date"},
        ],
        "qualifiers": {
            "Public Event": True,
            "unsupported_nested_value": {"source": "outside"},
        },
        "evidence_sentence": "Radio City launched PlanetRadiocity.com in May 2008.",
        "confidence": 0.91,
    }
    duplicate_with_higher_confidence = {
        **launch_fact,
        "arguments": list(reversed(launch_fact["arguments"])),
        "confidence": 0.95,
    }
    client = StaticJsonClient(
        {
            "facts": [
                launch_fact,
                duplicate_with_higher_confidence,
                {
                    "predicate": "serves",
                    "arguments": [
                        {"role": "Provider", "entity_id": "ent_radio"},
                        {"role": "Market", "entity_id": "ent_india"},
                    ],
                    "qualifiers": {},
                    "evidence_sentence": text,
                    "confidence": 0.9,
                },
                {
                    "predicate": "invented",
                    "arguments": [
                        {"role": "source", "entity_id": "ent_unknown"},
                        {"role": "target", "entity_id": "ent_india"},
                    ],
                    "qualifiers": {},
                    "evidence_sentence": "It serves India.",
                    "confidence": 1.0,
                },
                {
                    "predicate": "unsupported",
                    "arguments": [
                        {"role": "source", "entity_id": "ent_radio"},
                        {"role": "target", "entity_id": "ent_india"},
                    ],
                    "qualifiers": {},
                    "evidence_sentence": "Radio City is based in India.",
                    "confidence": 1.0,
                },
            ]
        }
    )

    facts = extract_facts_with_llm([chunk], entities, client)

    assert len(facts) == 2
    by_predicate = {fact.predicate: fact for fact in facts}
    launch = by_predicate["launched"]
    assert len(launch.arguments) == 3
    assert {argument.role for argument in launch.arguments} == {
        "launching_organization",
        "product",
        "time",
    }
    assert launch.confidence == 0.95
    assert launch.qualifiers == {"public_event": "true"}
    assert by_predicate["serves"].evidence_sentence == text

    system, prompt = client.calls[0]
    assert system == FACT_SYSTEM_PROMPT
    assert {item["entity_id"] for item in prompt["known_entities"]} == {
        entity.entity_id for entity in entities
    }
    assert prompt["known_entities"][0]["mentions"] == ["Radio City"]


def _entity(
    entity_id: str,
    canonical_name: str,
    entity_type: str,
    source_mention: str,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        canonical_name=canonical_name,
        entity_type=entity_type,
        source_mentions=[source_mention],
    )


def _stable_entity_id(entity_type: str, normalized_name: str) -> str:
    import hashlib

    digest = hashlib.sha1(f"{entity_type}:{normalized_name}".encode()).hexdigest()[:12]
    return f"ent_{digest}"
