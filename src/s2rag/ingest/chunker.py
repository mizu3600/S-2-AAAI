import hashlib
import re

from s2rag.ingest.schemas import Chunk, ParsedDocument


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_document(
    parsed: ParsedDocument,
    chunk_tokens: int = 512,
    overlap_tokens: int = 96,
) -> list[Chunk]:
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("expected 0 <= overlap_tokens < chunk_tokens")
    chunks: list[Chunk] = []
    for section in parsed.sections:
        tokens = _tokens(section.text)
        if not tokens:
            continue
        step = chunk_tokens - overlap_tokens
        for start in range(0, len(tokens), step):
            window = tokens[start : start + chunk_tokens]
            if not window:
                break
            text = " ".join(window)
            char_start = section.start_char + _approx_char_offset(tokens, start)
            digest = hashlib.sha1(
                f"{parsed.document.document_id}:{section.title}:{start}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{digest}",
                    document_id=parsed.document.document_id,
                    section=section.title,
                    text=text,
                    start_char=char_start,
                    end_char=char_start + len(text),
                    page=section.page,
                )
            )
            if start + chunk_tokens >= len(tokens):
                break
    return chunks


def _approx_char_offset(tokens: list[str], start: int) -> int:
    return sum(len(token) + 1 for token in tokens[:start])
