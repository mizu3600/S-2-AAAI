import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from qmshe.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSuite,
    Passage,
    SupportingFact,
)


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "item"


def _read_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
        support_titles = [title for title, pid in title_to_id.items() if any(x.passage_id == pid for x in support)]
        support_passage_ids = [title_to_id[title] for title in support_titles]
        bridges = support_titles[:-1] if row.get("type") == "bridge" else support_titles[1:-1]
        return BenchmarkExample(
            example_id=example_id, question=row["question"], answer=row.get("answer", ""),
            passages=passages, supporting_facts=support, bridge_entities=bridges,
            gold_path=support_passage_ids, hop_count=max(2, len(set(support_titles))),
            query_type=row.get("type", "unknown"), dataset=self.name, split=split,
            metadata={"level": row.get("level")},
        )


class TwoWikiAdapter(HotpotAdapter):
    name = "2wikimultihopqa"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        example = super().convert(row, split)
        example.dataset = self.name
        example.query_type = row.get("type", row.get("question_type", "unknown"))
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
            example_id=example_id, question=row["question"], answer=row.get("answer", ""),
            passages=passages, supporting_facts=support,
            bridge_entities=[str(item.get("answer", "")) for item in decomposition[:-1] if item.get("answer")],
            gold_path=[str(item.get("question", item)) for item in decomposition],
            hop_count=max(1, len(decomposition)), query_type=f"{max(1, len(decomposition))}-hop",
            dataset=self.name, split=split,
        )


class QasperAdapter(BenchmarkAdapter):
    name = "qasper"

    def load(self, path: str | Path, split: str = "validation", limit: int | None = None) -> BenchmarkSuite:
        examples = []
        for paper in _read_records(path):
            qas = paper.get("qas", {})
            if not isinstance(qas, dict):
                examples.append(self.convert(paper, split))
                continue
            questions = qas.get("question", [])
            question_ids = qas.get("question_id", [])
            answers = qas.get("answers", [])
            for index, question in enumerate(questions):
                flattened = {
                    "id": question_ids[index] if index < len(question_ids) else f"{paper.get('id')}-{index}",
                    "question": question,
                    "answers": answers[index] if index < len(answers) else [],
                    "full_text": paper.get("full_text", {}),
                    "paper_id": paper.get("id"),
                    "paper_title": paper.get("title"),
                }
                examples.append(self.convert(flattened, split))
                if limit is not None and len(examples) >= limit:
                    return BenchmarkSuite(name=self.name, split=split, examples=examples,
                                          source=str(Path(path)))
        return BenchmarkSuite(name=self.name, split=split, examples=examples, source=str(Path(path)))

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        example_id = str(row.get("id", row.get("question_id", "unknown")))
        full_text = row.get("full_text", row.get("context", {}))
        if isinstance(full_text, dict):
            section_names = full_text.get("section_name", [])
            paragraphs = full_text.get("paragraphs", [])
        else:
            section_names, paragraphs = ["Paper"], [full_text]
        passages = []
        for index, paragraphs_in_section in enumerate(paragraphs):
            if isinstance(paragraphs_in_section, str):
                paragraphs_in_section = [paragraphs_in_section]
            passages.append(Passage(
                passage_id=f"{_safe_id(example_id)}_p{index}",
                title=str(section_names[index]) if index < len(section_names) else f"Section {index}",
                sentences=[sentence for paragraph in paragraphs_in_section for sentence in _sentence_split(paragraph)],
            ))
        answers = row.get("answers", row.get("answer", ""))
        answer = _qasper_answer(answers)
        evidence_strings = _qasper_evidence(answers)
        support = []
        for passage in passages:
            for sentence_index, sentence in enumerate(passage.sentences):
                if any(evidence in sentence or sentence in evidence for evidence in evidence_strings):
                    support.append(SupportingFact(passage_id=passage.passage_id, sentence_index=sentence_index))
        return BenchmarkExample(
            example_id=example_id, question=row["question"], answer=answer, passages=passages,
            supporting_facts=support, hop_count=max(1, len({x.passage_id for x in support})),
            query_type="scientific_qa", dataset=self.name, split=split,
            metadata={"paper_id": row.get("paper_id"), "paper_title": row.get("paper_title")},
        )


class MetaQAAdapter(BenchmarkAdapter):
    name = "metaqa"

    def convert(self, row: dict, split: str) -> BenchmarkExample:
        context = row.get("context", row.get("triples", []))
        passages = []
        for index, triple in enumerate(context):
            text = " | ".join(triple) if isinstance(triple, list) else str(triple)
            passages.append(Passage(passage_id=f"meta_{index}", title="Knowledge graph", sentences=[text]))
        hops = int(row.get("hop", row.get("hop_count", 1)))
        return BenchmarkExample(
            example_id=str(row.get("id", "unknown")), question=row["question"],
            answer=row.get("answer", row.get("answers", [])), passages=passages,
            gold_path=[str(item) for item in row.get("path", [])], hop_count=hops,
            query_type=f"{hops}-hop", dataset=self.name, split=split,
        )


def _sentence_split(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]


def _qasper_answer(answers) -> str | list[str]:
    if isinstance(answers, str):
        return answers
    if isinstance(answers, dict):
        answers = answers.get("answer", answers.get("answers", []))
    output = []
    for answer in answers or []:
        answer = answer.get("answer", answer) if isinstance(answer, dict) else answer
        if isinstance(answer, dict):
            value = answer.get("free_form_answer") or answer.get("extractive_spans") or answer.get("yes_no") or answer.get("unanswerable")
        else:
            value = answer
        if value not in (None, "", []):
            output.extend(value if isinstance(value, list) else [str(value)])
    return output


def _qasper_evidence(answers) -> list[str]:
    records = answers.get("answer", answers.get("answers", [])) if isinstance(answers, dict) else answers
    evidence = []
    for record in records or []:
        answer = record.get("answer", record) if isinstance(record, dict) else {}
        if isinstance(answer, dict):
            evidence.extend(answer.get("evidence", []))
    return evidence


ADAPTERS = {
    "hotpotqa": HotpotAdapter(), "2wiki": TwoWikiAdapter(), "2wikimultihopqa": TwoWikiAdapter(),
    "musique": MusiqueAdapter(), "qasper": QasperAdapter(), "metaqa": MetaQAAdapter(),
}


def load_benchmark(name: str, path: str | Path, split: str = "validation", limit: int | None = None) -> BenchmarkSuite:
    try:
        adapter = ADAPTERS[name.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark: {name}") from exc
    return adapter.load(path, split, limit)
