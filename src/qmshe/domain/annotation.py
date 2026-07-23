from pathlib import Path

from qmshe.domain.psc import PSCBenchmarkItem
from qmshe.ingest.schemas import Corpus


class PSCBenchmarkStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[PSCBenchmarkItem]:
        if not self.path.exists():
            return []
        return [PSCBenchmarkItem.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def save(self, items: list[PSCBenchmarkItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8")

    def upsert(self, item: PSCBenchmarkItem, corpus: Corpus) -> None:
        validate_benchmark_item(item, corpus)
        items = {current.question_id: current for current in self.load()}
        items[item.question_id] = item
        self.save(list(items.values()))


def validate_benchmark_item(item: PSCBenchmarkItem, corpus: Corpus) -> None:
    chunk_ids = {chunk.chunk_id for chunk in corpus.chunks}
    fact_ids = {fact.hyperedge_id for fact in corpus.evidence_hyperedges}
    entity_ids = {entity.entity_id for entity in corpus.entities}
    document_ids = {document.document_id for document in corpus.documents}
    checks = [
        (set(item.supporting_chunk_ids) <= chunk_ids, "unknown supporting chunk"),
        (set(item.supporting_hyperedge_ids) <= fact_ids, "unknown supporting hyperedge"),
        (set(item.bridge_entity_ids) <= entity_ids, "unknown bridge entity"),
        (set(item.source_document_ids) <= document_ids, "unknown source document"),
        (set(item.gold_path) <= entity_ids | fact_ids, "unknown gold path object"),
    ]
    for valid, message in checks:
        if not valid:
            raise ValueError(message)


def split_by_document(items: list[PSCBenchmarkItem], test_fraction: float = 0.2) -> tuple[list, list]:
    documents = sorted({document for item in items for document in item.source_document_ids})
    split_index = max(1, int(len(documents) * (1 - test_fraction)))
    train_documents = set(documents[:split_index])
    train, test = [], []
    for item in items:
        target = train if set(item.source_document_ids) <= train_documents else test
        target.append(item)
    return train, test
