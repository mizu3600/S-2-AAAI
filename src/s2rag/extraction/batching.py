from __future__ import annotations

from s2rag.ingest.schemas import Chunk


def passage_batches(
    chunks: list[Chunk],
    max_chars: int,
) -> list[list[Chunk]]:
    """Group sentence chunks by source passage and a conservative character budget."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    by_document: dict[str, list[Chunk]] = {}
    document_order: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in by_document:
            by_document[chunk.document_id] = []
            document_order.append(chunk.document_id)
        by_document[chunk.document_id].append(chunk)

    output: list[list[Chunk]] = []
    for document_id in document_order:
        document_chunks = by_document[document_id]
        document_chars = sum(len(chunk.text) for chunk in document_chunks)
        if document_chars <= max_chars:
            output.append(document_chunks)
            continue
        current: list[Chunk] = []
        current_chars = 0
        for chunk in document_chunks:
            if current and current_chars + len(chunk.text) > max_chars:
                output.append(current)
                current = []
                current_chars = 0
            current.append(chunk)
            current_chars += len(chunk.text)
        if current:
            output.append(current)
    return output
