from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd
from graphrag import api
from graphrag.api.index import build_index
from graphrag.cli.initialize import initialize_project_at
from graphrag.config.load_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def document_text(document: dict) -> str:
    return f"{document['title']}: {document['text']}"


def configure(root: Path) -> None:
    initialize_project_at(
        root,
        force=True,
        model="deepseek-chat",
        embedding_model="BAAI/bge-m3",
    )
    path = root / "settings.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "api_key: ${GRAPHRAG_API_KEY}",
        "api_key: ${DEEPSEEK_API_KEY}\n    api_base: https://api.deepseek.com",
        1,
    )
    text = text.replace(
        "api_key: ${GRAPHRAG_API_KEY}",
        "api_key: ${SILICONFLOW_API_KEY}\n    api_base: https://api.siliconflow.cn/v1",
        1,
    )
    text = text.replace("max_gleanings: 1", "max_gleanings: 0", 1)
    text = text.replace("  db_uri: output/lancedb", "  db_uri: output/lancedb\n  vector_size: 1024")
    text = "concurrent_requests: 16\n" + text
    path.write_text(text, encoding="utf-8")


def load_outputs(root: Path) -> dict[str, pd.DataFrame | None]:
    output = root / "output"
    required = ("entities", "communities", "community_reports", "text_units", "relationships")
    result = {name: pd.read_parquet(output / f"{name}.parquet") for name in required}
    covariates = output / "covariates.parquet"
    result["covariates"] = pd.read_parquet(covariates) if covariates.exists() else None
    return result


def source_ranking(context: dict, documents: list[dict]) -> list[str]:
    sources = context.get("sources")
    if sources is None:
        return []
    rows = sources.to_dict("records") if hasattr(sources, "to_dict") else sources
    ranking = []
    for row in rows:
        content = str(row.get("text") or row.get("content") or "")
        document_id = next(
            (
                document["document_id"]
                for document in documents
                if document_text(document) in content or document["text"] in content
            ),
            None,
        )
        if document_id is not None and document_id not in ranking:
            ranking.append(document_id)
    return ranking


async def main() -> None:
    args = parse_args()
    work_root = args.work_dir.resolve()
    examples = json.loads(args.input.read_text(encoding="utf-8"))[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            json.loads(line)["example_id"]
            for line in args.output.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("error") is None
        }

    with args.output.open("a", encoding="utf-8") as output:
        for index, example in enumerate(examples, 1):
            if example["example_id"] in completed:
                continue
            started = time.monotonic()
            record = {
                "example_id": example["example_id"],
                "framework": "graphrag",
                "ranking": [],
                "elapsed_seconds": None,
                "error": None,
            }
            try:
                root = work_root / example["example_id"]
                configure(root)
                config = load_config(root)
                documents = pd.DataFrame(
                    [
                        {
                            "id": document["document_id"],
                            "title": document["title"],
                            "text": document_text(document),
                            "creation_date": "2026-07-22",
                        }
                        for document in example["documents"]
                    ]
                )
                results = await build_index(
                    config=config,
                    method="standard",
                    input_documents=documents,
                )
                errors = [
                    f"{item.workflow}: {item.error}" for item in results if item.error is not None
                ]
                if errors:
                    raise RuntimeError("; ".join(errors))
                frames = load_outputs(root)
                _, context = await api.local_search(
                    config=config,
                    entities=frames["entities"],
                    communities=frames["communities"],
                    community_reports=frames["community_reports"],
                    text_units=frames["text_units"],
                    relationships=frames["relationships"],
                    covariates=frames["covariates"],
                    community_level=2,
                    response_type="Single Paragraph",
                    query=example["question"],
                )
                record["ranking"] = source_ranking(context, example["documents"])
                record["context_tables"] = sorted(context)
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            record["elapsed_seconds"] = time.monotonic() - started
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[{index}/{len(examples)}] {example['example_id']} "
                f"docs={len(record['ranking'])} error={record['error']}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
