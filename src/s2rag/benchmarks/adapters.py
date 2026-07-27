import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from s2rag.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSuite,
    Passage,
    SupportingFact,
)


def _sentence_split(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", str(text)) if item.strip()]


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "item"


def _read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("data", "examples", "instances", "rows"):
        if isinstance(data.get(key), list):
            rows = data[key]
            return [row.get("row", row) for row in rows]
    raise ValueError(f"cannot find records in {path}")


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def convert(self, row: dict, split: str) -> BenchmarkExample:
        raise NotImplementedError

    def load(self, path: str | Path, split: str = "validation", limit: int | None = None) -> BenchmarkSuite:
        records = _read_records(path)
        if limit is not None:
            records = records[:limit]
        examples = [self.convert(record, split) for record in records]
        return BenchmarkSuite(name=self.name, split=split, examples=examples, source=str(Path(path)))


class HotpotAdapter(BenchmarkAdapter):
    name = "hotpotqa"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        raw_context = row.get("context", [])
        if isinstance(raw_context, dict):
            contexts = zip(raw_context.get("title", []), raw_context.get("sentences", []), strict=False)
        else:
            contexts = raw_context
        passages, title_to_id = [], {}
        example_id = str(row.get("_id", row.get("id", "unknown")))
        for index, item in enumerate(contexts):
            title, sentences = item
            passage_id = f"{_safe_id(example_id)}_p{index}"
            title_to_id[str(title)] = passage_id
            passages.append(Passage(passage_id=passage_id, title=str(title), sentences=list(sentences)))
        raw_support = row.get("supporting_facts", [])
        if isinstance(raw_support, dict):
            raw_support = zip(raw_support.get("title", []), raw_support.get("sent_id", []), strict=False)
        support = [
            SupportingFact(passage_id=title_to_id[str(title)], sentence_index=int(sentence_id))
            for title, sentence_id in raw_support
            if str(title) in title_to_id
        ]
        if not support:
            raise ValueError(f"HotpotQA example {example_id} has no valid supporting facts")
        support_titles = [title for title, pid in title_to_id.items() if any(x.passage_id == pid for x in support)]
        support_passage_ids = [title_to_id[title] for title in support_titles]
        bridges = support_titles[:-1] if row.get("type") == "bridge" else support_titles[1:-1]
        benchmark_config = row.get(
            "_s2rag_benchmark_config",
            "distractor" if len(passages) == 10 else "unspecified",
        )
        corpus_scope = row.get(
            "_s2rag_corpus_scope",
            (
                "per_question_candidate_passages"
                if benchmark_config == "distractor"
                else "unspecified"
            ),
        )
        return BenchmarkExample(
            example_id=example_id, question=row["question"], answer=row.get("answer", ""),
            passages=passages, supporting_facts=support, bridge_entities=bridges,
            gold_path=support_passage_ids, hop_count=max(2, len(set(support_titles))),
            query_type=row.get("type", "unknown"), dataset=self.name, split=split,
            metadata={
                "level": row.get("level"),
                "metric_profile": "hotpotqa_official",
                "evidence_level": "sentence",
                "answer_language": "en",
                "benchmark_config": benchmark_config,
                "corpus_scope": corpus_scope,
                "official_evaluator": "hotpot_evaluate_v1.py",
            },
        )


class TwoWikiAdapter(HotpotAdapter):
    name = "2wikimultihopqa"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        example = super().convert(row, split)
        example.dataset = self.name
        example.query_type = row.get("type", row.get("question_type", "unknown"))
        example.answer = _answers_with_aliases(row)
        example.metadata.update(
            {
                "metric_profile": "2wikimultihopqa_official",
                "benchmark_config": "dev",
                "corpus_scope": "per_question_candidate_passages",
                "official_evaluator": "2wikimultihop_evaluate_v1.1.py",
            }
        )
        evidence = row.get("evidences", row.get("evidence", []))
        if evidence:
            example.metadata["evidences"] = evidence
        return example


class MusiqueAdapter(BenchmarkAdapter):
    name = "musique"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        example_id = str(row.get("id", row.get("question_id", "unknown")))
        paragraphs = row.get("paragraphs", row.get("context", []))
        passages, support = [], []
        for index, paragraph in enumerate(paragraphs):
            passage_id = f"{_safe_id(example_id)}_p{index}"
            text = paragraph.get("paragraph_text", paragraph.get("text", ""))
            sentences = paragraph.get("sentences") or _sentence_split(text)
            passages.append(Passage(passage_id=passage_id, title=paragraph.get("title", f"Paragraph {index}"), sentences=sentences))
            if paragraph.get("is_supporting", paragraph.get("supporting", False)):
                support.extend(SupportingFact(passage_id=passage_id, sentence_index=i) for i in range(len(sentences)))
        decomposition = row.get("question_decomposition", row.get("decomposition", []))
        return BenchmarkExample(
            example_id=example_id,
            question=row["question"],
            answer=_answers_with_aliases(row),
            passages=passages, supporting_facts=support,
            bridge_entities=[str(item.get("answer", "")) for item in decomposition[:-1] if item.get("answer")],
            gold_path=[str(item.get("question", item)) for item in decomposition],
            hop_count=max(1, len(decomposition)), query_type=f"{max(1, len(decomposition))}-hop",
            dataset=self.name,
            split=split,
            metadata={
                "metric_profile": "musique_official",
                "evidence_level": "paragraph",
                "corpus_scope": "per_question_candidate_paragraphs",
                "official_evaluator": "evaluate_v1.0.py",
            },
        )


class UltraDomainAdapter(BenchmarkAdapter):
    name = "ultradomain"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        example_id = str(
            row.get("_id", row.get("id", row.get("example_id", "unknown")))
        )
        question = str(row.get("question", row.get("input", ""))).strip()
        if not question:
            raise ValueError(f"UltraDomain example {example_id} has no question/input")
        domain = row.get(
            "_s2rag_domain",
            row.get(
                "_ureval_domain",
                row.get("domain", row.get("category", row.get("dataset", "general"))),
            ),
        )
        context = row.get("context", row.get("paragraphs", []))
        passages = []
        if isinstance(context, list):
            for index, item in enumerate(context):
                if isinstance(item, dict):
                    title = item.get("title", f"Domain_{domain}_{index}")
                    sentences = item.get("sentences") or _sentence_split(item.get("text", ""))
                else:
                    title, sentences = f"Passage_{index}", _sentence_split(str(item))
                passages.append(Passage(passage_id=f"{_safe_id(example_id)}_p{index}", title=title, sentences=sentences))
        else:
            passages.append(Passage(passage_id=f"{_safe_id(example_id)}_p0", title=f"Domain_{domain}", sentences=_sentence_split(str(context))))
        
        return BenchmarkExample(
            example_id=example_id,
            question=question,
            answer=row.get("answer", row.get("answers", "")),
            passages=passages,
            supporting_facts=[],
            hop_count=1,
            query_type=f"domain_{domain}", dataset=self.name, split=split,
            metadata={
                "domain": domain,
                "context_length": sum(len(" ".join(p.sentences)) for p in passages),
                "evidence_level": "unavailable",
                "multi_hop_annotation": "unavailable",
                "metric_profile": "unified_squad_style",
                "corpus_scope": "per_example_long_document",
                "official_evaluator": "none",
            },
        )


class MixAdapter(BenchmarkAdapter):
    name = "mix"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        sub_dataset = str(row.get("dataset", "")).casefold()
        adapter = ADAPTERS.get(sub_dataset)
        if adapter is None or adapter is self:
            raise ValueError(f"unsupported dataset in mix row: {sub_dataset or 'missing'}")
        example = adapter.convert(row, split)
        example.dataset = self.name
        example.metadata["original_dataset"] = sub_dataset
        return example


ADAPTERS = {
    "hotpotqa": HotpotAdapter(), "2wiki": TwoWikiAdapter(), "2wikimultihopqa": TwoWikiAdapter(),
    "musique": MusiqueAdapter(), "ultradomain": UltraDomainAdapter(), "mix": MixAdapter(),
}


def load_benchmark(name: str, path: str | Path, split: str = "validation", limit: int | None = None) -> BenchmarkSuite:
    try:
        adapter = ADAPTERS[name.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark: {name}") from exc
    return adapter.load(path, split, limit)


def _answers_with_aliases(row: dict) -> str | list[str]:
    answer = row.get("answer", "")
    aliases = row.get("answer_aliases", row.get("_answer_aliases", []))
    values = [str(answer), *(str(item) for item in aliases)]
    unique = list(dict.fromkeys(item for item in values if item))
    return unique if len(unique) > 1 else (unique[0] if unique else "")
