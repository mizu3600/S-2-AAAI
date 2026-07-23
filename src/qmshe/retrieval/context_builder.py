from qmshe.ingest.schemas import Corpus


def build_context(corpus: Corpus, fact_ids: list[str], limit: int = 12) -> tuple[str, list[dict]]:
    facts = {fact.hyperedge_id: fact for fact in corpus.evidence_hyperedges}
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    documents = {document.document_id: document for document in corpus.documents}
    blocks, citations = [], []
    for fact_id in fact_ids[:limit]:
        fact = facts[fact_id]
        for chunk_id in fact.evidence_chunk_ids:
            chunk = chunks[chunk_id]
            document = documents.get(chunk.document_id)
            source = document.title if document else chunk.document_id
            location = f"{chunk.section}" + (f", page {chunk.page}" if chunk.page else "")
            blocks.append(
                f"[Evidence {fact_id}]\nSource: {source}, {location}\n"
                f"Fact: {fact.predicate}\nOriginal text: {fact.evidence_sentence}"
            )
            citations.append(
                {"evidence_id": fact_id, "chunk_id": chunk_id, "document_id": chunk.document_id,
                 "source": source, "section": chunk.section, "page": chunk.page}
            )
    return "\n\n".join(blocks), citations

