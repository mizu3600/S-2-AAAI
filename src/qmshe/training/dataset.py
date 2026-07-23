from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExample:
    query: str
    positive_ids: list[str]
    supporting_chunk_ids: list[str]
    bridge_entity_ids: list[str]
    gold_path: list[str]
    query_type: str = "multi_hop"


def validate_split_no_document_leakage(train_documents: set[str], test_documents: set[str]) -> None:
    overlap = train_documents & test_documents
    if overlap:
        raise ValueError(f"document leakage detected: {sorted(overlap)[:5]}")

