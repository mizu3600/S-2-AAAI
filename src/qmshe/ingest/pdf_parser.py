import hashlib
from pathlib import Path

import fitz

from qmshe.ingest.schemas import Document, ParsedDocument, Section


def _identifier(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"doc_{digest}"


def parse_document(path: str | Path, domain: str = "PSC") -> ParsedDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    sections: list[Section] = []
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            offset = 0
            for page_no, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    sections.append(
                        Section(title=f"Page {page_no}", text=text, page=page_no, start_char=offset)
                    )
                    offset += len(text) + 1
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        sections = _split_text_sections(text)
    else:
        raise ValueError(f"unsupported document type: {suffix}")
    document = Document(
        document_id=_identifier(path), title=path.stem, source_uri=str(path.resolve()), domain=domain
    )
    return ParsedDocument(document=document, sections=sections, metadata={"parser": "pymupdf"})


def _split_text_sections(text: str) -> list[Section]:
    lines = text.splitlines()
    result: list[Section] = []
    title, body, offset = "Document", [], 0
    for line in lines:
        if line.lstrip().startswith("#") and body:
            joined = "\n".join(body).strip()
            result.append(Section(title=title, text=joined, start_char=offset))
            offset += len(joined) + 1
            body = []
        if line.lstrip().startswith("#"):
            title = line.lstrip("# ").strip() or "Untitled"
        else:
            body.append(line)
    joined = "\n".join(body).strip()
    if joined:
        result.append(Section(title=title, text=joined, start_char=offset))
    return result

