import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus, build_suite_corpus
from qmshe.evaluation.splits import fixed_partition
from qmshe.pipeline import verbalize_fact
from qmshe.retrieval.seed_retriever import BM25Retriever
from qmshe.training.lora_reranker import RerankerLoRAConfig, train_reranker_lora


def main(
    base_model: Path = typer.Option(..., help="Local BGE reranker model directory"),
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("data/models/bge_reranker_lora")),
    limit: int = typer.Option(500), epochs: int = typer.Option(3),
    batch_size: int = typer.Option(8), learning_rate: float = typer.Option(1e-5),
    negatives_per_query: int = typer.Option(12), validation_fraction: float = typer.Option(0.15),
    test_fraction: float = typer.Option(0.15), seed: int = typer.Option(42),
    max_memory_fraction: float = typer.Option(0.45),
) -> None:
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    partitions = fixed_partition(suite.examples, validation_fraction, test_fraction)
    built = build_suite_corpus(suite.examples)
    names = {item.entity_id: item.canonical_name for item in built.corpus.entities}
    texts = {
        fact.hyperedge_id: verbalize_fact(fact, names)
        for fact in built.corpus.evidence_hyperedges
    }
    positives = dict(built.training_pairs)
    def prepare(examples):
        output = []
        for example in examples:
            gold = positives.get(example.question, set())
            local = build_example_corpus(example)
            local_ids = sorted(fact.hyperedge_id for fact in local.corpus.evidence_hyperedges)
            retriever = BM25Retriever(local_ids, [texts[item] for item in local_ids])
            candidates = [
                hit.object_id for hit in retriever.search(example.question, len(local_ids))
            ]
            candidates = list(dict.fromkeys([*sorted(gold), *candidates]))
            if gold:
                output.append((example.question, gold, candidates))
        return output

    report = train_reranker_lora(
        prepare(partitions["train"]), prepare(partitions["validation"]), texts,
        RerankerLoRAConfig(
            base_model=str(base_model), output_dir=str(output_dir), epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            negatives_per_query=negatives_per_query, seed=seed,
            max_memory_fraction=max_memory_fraction,
        ),
    )
    typer.echo(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    typer.run(main)
