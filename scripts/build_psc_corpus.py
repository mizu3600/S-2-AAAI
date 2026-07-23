import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.domain.psc import build_psc_corpus, validate_psc_corpus
from qmshe.domain.quality import audit_corpus
from qmshe.pipeline import save_corpus


def main(
    input_dir: Path = typer.Argument(...),
    output: Path = typer.Option(Path("data/processed/psc_corpus.json")),
    use_llm: bool = typer.Option(True),
) -> None:
    corpus = build_psc_corpus(input_dir, use_llm=use_llm)
    errors = validate_psc_corpus(corpus)
    save_corpus(corpus, output)
    report = asdict(audit_corpus(corpus))
    output.with_suffix(".quality.json").write_text(json.dumps({"errors": errors, **report}, indent=2), encoding="utf-8")
    typer.echo(f"saved PSC corpus to {output}; validation errors={len(errors)}")


if __name__ == "__main__":
    typer.run(main)

