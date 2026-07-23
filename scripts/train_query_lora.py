import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_suite_corpus
from qmshe.evaluation.splits import fixed_partition
from qmshe.training.lora_query_encoder import LoRATrainingConfig, train_query_lora


def main(
    base_model: Path = typer.Option(..., help="Local BGE-M3 model directory"),
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("data/models/bge_m3_query_lora")),
    limit: int = typer.Option(100), epochs: int = typer.Option(2),
    max_length: int = typer.Option(256), document_batch_size: int = typer.Option(8),
    learning_rate: float = typer.Option(2e-5), max_memory_fraction: float = typer.Option(0.45),
    validation_fraction: float = typer.Option(0.15, min=0.05, max=0.4),
    test_fraction: float = typer.Option(0.15, min=0.05, max=0.4),
    seed: int = typer.Option(42),
) -> None:
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    built = build_suite_corpus(suite.examples)
    names = {entity.entity_id: entity.canonical_name for entity in built.corpus.entities}
    fact_text_by_id = {
        fact.hyperedge_id: _verbalize_fact(fact, names)
        for fact in built.corpus.evidence_hyperedges
    }
    partitions = fixed_partition(suite.examples, validation_fraction, test_fraction)
    pairs_by_question = dict(built.training_pairs)
    train_pairs = [
        (item.question, pairs_by_question[item.question]) for item in partitions["train"]
        if item.question in pairs_by_question
    ]
    evaluation_pairs = [
        (item.question, pairs_by_question[item.question]) for item in partitions["validation"]
        if item.question in pairs_by_question
    ]
    if not train_pairs or not evaluation_pairs:
        raise typer.BadParameter("at least two examples are required for a train/evaluation split")
    report = train_query_lora(
        [question for question, _ in train_pairs],
        [positive_ids for _, positive_ids in train_pairs],
        fact_text_by_id,
        LoRATrainingConfig(
            base_model=str(base_model), output_dir=str(output_dir), epochs=epochs,
            max_length=max_length, document_batch_size=document_batch_size,
            learning_rate=learning_rate, max_memory_fraction=max_memory_fraction, seed=seed,
        ),
        evaluation_questions=[question for question, _ in evaluation_pairs],
        evaluation_positive_ids=[positive_ids for _, positive_ids in evaluation_pairs],
    )
    typer.echo(json.dumps(asdict(report), indent=2))


def _verbalize_fact(fact, names: dict[str, str]) -> str:
    arguments = ", ".join(
        f"{argument.role}={names.get(argument.entity_id, argument.entity_id)}"
        for argument in fact.arguments
    )
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in fact.qualifiers.items() if value is not None
    )
    return f"{fact.predicate}: {arguments}" + (f"; {qualifiers}" if qualifiers else "")


if __name__ == "__main__":
    typer.run(main)
