from qmshe.ingest.service import ingest_document
from qmshe.pipeline import load_corpus, save_corpus


def test_ingest_markdown_to_corpus(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Results\n"
        "PEAI in inverted PSCs enables surface defect passivation, reduces "
        "non-radiative recombination and improves Voc.",
        encoding="utf-8",
    )

    corpus = ingest_document(source)

    assert corpus.documents
    assert corpus.chunks
    assert corpus.entities
    assert corpus.evidence_hyperedges

    output = tmp_path / "corpus.json"
    save_corpus(corpus, output)
    assert load_corpus(output) == corpus
