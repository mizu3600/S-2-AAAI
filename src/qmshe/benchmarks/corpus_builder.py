import hashlib
import re
from dataclasses import dataclass

from qmshe.benchmarks.schemas import BenchmarkExample
from qmshe.ingest.schemas import (
    Argument,
    Chunk,
    Corpus,
    Document,
    Entity,
    EvidenceHyperedge,
)


@dataclass(frozen=True)
class BuiltBenchmarkCorpus:
    corpus: Corpus
    gold_fact_ids: set[str]
    gold_chunk_ids: set[str]
    bridge_entity_ids: set[str]
    gold_path: list[str]


@dataclass(frozen=True)
class BuiltBenchmarkSuiteCorpus:
    corpus: Corpus
    training_pairs: list[tuple[str, set[str]]]
    bridge_by_question: dict[str, set[str]]


def build_example_corpus(example: BenchmarkExample) -> BuiltBenchmarkCorpus:
    documents, chunks, facts = [], [], []
    entities: dict[str, Entity] = {}
    support_keys = {(item.passage_id, item.sentence_index) for item in example.supporting_facts}
    gold_fact_ids, gold_chunk_ids = set(), set()
    title_entity_by_passage: dict[str, str] = {}
    concept_ids_by_name: dict[str, str] = {}
    for passage in example.passages:
        document_id = _id("doc", f"{example.dataset}:{example.example_id}:{passage.passage_id}")
        documents.append(Document(
            document_id=document_id, title=passage.title, source_uri=passage.source_uri or f"benchmark://{example.dataset}/{example.example_id}/{passage.passage_id}",
            domain="benchmark",
        ))
        title_entity_id = _id("ent", f"title:{passage.title.casefold()}")
        title_entity_by_passage[passage.passage_id] = title_entity_id
        entities.setdefault(title_entity_id, Entity(
            entity_id=title_entity_id, canonical_name=passage.title, entity_type="document_topic",
            description=f"Benchmark passage titled {passage.title}",
        ))
        offset = 0
        for sentence_index, sentence in enumerate(passage.sentences):
            chunk_id = _id("chunk", f"{example.example_id}:{passage.passage_id}:{sentence_index}")
            chunks.append(Chunk(
                chunk_id=chunk_id, document_id=document_id, section=passage.title, text=sentence,
                start_char=offset, end_char=offset + len(sentence), page=None,
            ))
            offset += len(sentence) + 1
            concept_names = _concepts(sentence, passage.title)
            arguments = [Argument(role="source", entity_id=title_entity_id)]
            for concept_name in concept_names[:8]:
                normalized = concept_name.casefold()
                concept_id = concept_ids_by_name.setdefault(normalized, _id("ent", f"concept:{normalized}"))
                entities.setdefault(concept_id, Entity(
                    entity_id=concept_id, canonical_name=concept_name, entity_type="benchmark_concept",
                    description=f"Concept mentioned in {passage.title}",
                ))
                arguments.append(Argument(role="mention", entity_id=concept_id))
            if len({argument.entity_id for argument in arguments}) < 2:
                sentence_entity_id = _id("ent", f"sentence:{chunk_id}")
                entities[sentence_entity_id] = Entity(
                    entity_id=sentence_entity_id, canonical_name=f"sentence {sentence_index}",
                    entity_type="sentence_anchor", description=sentence,
                )
                arguments.append(Argument(role="statement", entity_id=sentence_entity_id))
            fact_id = _id("fact", chunk_id)
            facts.append(EvidenceHyperedge(
                hyperedge_id=fact_id, predicate="states", arguments=arguments,
                qualifiers={"dataset": example.dataset, "example_id": example.example_id},
                evidence_chunk_ids=[chunk_id], evidence_sentence=sentence, confidence=1.0,
            ))
            if (passage.passage_id, sentence_index) in support_keys:
                gold_fact_ids.add(fact_id)
                gold_chunk_ids.add(chunk_id)
    bridge_entity_ids = {
        concept_ids_by_name[name.casefold()]
        for name in example.bridge_entities
        if name.casefold() in concept_ids_by_name
    }
    path = [title_entity_by_passage[item] for item in example.gold_path if item in title_entity_by_passage]
    return BuiltBenchmarkCorpus(
        corpus=Corpus(documents=documents, chunks=chunks, entities=list(entities.values()), evidence_hyperedges=facts,
                      graph_version=f"{example.dataset}-{example.example_id}-v1"),
        gold_fact_ids=gold_fact_ids, gold_chunk_ids=gold_chunk_ids,
        bridge_entity_ids=bridge_entity_ids, gold_path=path,
    )


def build_suite_corpus(examples: list[BenchmarkExample]) -> BuiltBenchmarkSuiteCorpus:
    documents, chunks, entities, facts = {}, {}, {}, {}
    training_pairs = []
    bridge_by_question = {}
    for example in examples:
        built = build_example_corpus(example)
        documents.update((document.document_id, document) for document in built.corpus.documents)
        chunks.update((chunk.chunk_id, chunk) for chunk in built.corpus.chunks)
        entities.update((entity.entity_id, entity) for entity in built.corpus.entities)
        facts.update((fact.hyperedge_id, fact) for fact in built.corpus.evidence_hyperedges)
        if built.gold_fact_ids:
            training_pairs.append((example.question, built.gold_fact_ids))
            bridge_by_question[example.question] = built.bridge_entity_ids
    return BuiltBenchmarkSuiteCorpus(
        corpus=Corpus(
            documents=list(documents.values()), chunks=list(chunks.values()), entities=list(entities.values()),
            evidence_hyperedges=list(facts.values()), graph_version="benchmark-suite-v1",
        ),
        training_pairs=training_pairs, bridge_by_question=bridge_by_question,
    )


def _id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode()).hexdigest()[:16]}"


def _concepts(sentence: str, title: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*){0,4}\b", sentence)
    words = re.findall(r"\b[a-zA-Z][\w'-]{5,}\b", sentence)
    ordered = [title, *candidates, *words]
    seen, output = set(), []
    for item in ordered:
        normalized = item.casefold().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(item.strip())
    return output
