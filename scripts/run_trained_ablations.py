import json
import random
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_suite_corpus
from qmshe.evaluation.experiment import BenchmarkExperimentRunner, LocalBenchmarkEncoder
from qmshe.evaluation.retrieval_metrics import recall_at_k
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("reports/ablations")),
    limit: int = typer.Option(20), epochs: int = typer.Option(3),
) -> None:
    suite = load_benchmark(dataset, input_path, limit=limit)
    pairs = list(suite.examples)
    random.Random(42).shuffle(pairs)
    split = min(max(1, int(len(pairs) * 0.8)), len(pairs) - 1)
    train_examples = pairs[:split]
    evaluation_examples = pairs[split:]
    merged = build_suite_corpus(suite.examples)
    train_built = build_suite_corpus(train_examples)
    evaluation_built = build_suite_corpus(evaluation_examples)
    encoder = LocalBenchmarkEncoder(128)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    full = _train_hypergraph(
        merged, train_built, encoder, epochs,
        bridge_loss_weight=0.5, use_hard_negatives=True, use_role_aware=True,
    )
    for variant in ("ours-low", "ours-fixed", "ours-no-high", "ours-no-raw", "ours-full"):
        recalls = [
            recall_at_k(full.ablation_search(question, variant, 20), gold, 20)
            for question, gold in evaluation_built.training_pairs
        ]
        results[variant] = _result(recalls, "runtime", full._ablation_training_history)

    configurations = {
        "ours-no-role": dict(bridge_loss_weight=0.5, use_hard_negatives=True, use_role_aware=False),
        "ours-no-bridge-loss": dict(bridge_loss_weight=0.0, use_hard_negatives=True, use_role_aware=True),
        "ours-no-hard-negative": dict(bridge_loss_weight=0.5, use_hard_negatives=False, use_role_aware=True),
    }
    for name, options in configurations.items():
        pipeline = _train_hypergraph(merged, train_built, encoder, epochs, **options)
        recalls = [
            recall_at_k(pipeline.query(question, 20, False).retrieved_hyperedges, gold, 20)
            for question, gold in evaluation_built.training_pairs
        ]
        results[name] = _result(recalls, "retrained", pipeline._ablation_training_history)

    graph = QMSGEGraphPipeline(
        merged.corpus, text_encoder=encoder, profile=GraphProfile.REIFIED_FACT
    )
    graph.generator.client = None
    graph_history = graph.train_stage_a(train_built.training_pairs, epochs=epochs)
    graph_recalls = [
        recall_at_k(graph.query(question, 20, False).retrieved_facts, gold, 20)
        for question, gold in evaluation_built.training_pairs
    ]
    results["ours-graph"] = _result(graph_recalls, "retrained", graph_history)

    baseline_records = BenchmarkExperimentRunner(
        methods=["semantic+lap_pe", "laplacian_eigenmaps"], encoder_dimension=128
    ).run(
        type(suite)(
            name=suite.name, split=suite.split, examples=evaluation_examples,
            source=suite.source, version=suite.version,
        ), output_dir / "structural_baselines",
    )
    mapping = {"semantic+lap_pe": "ours-concat", "laplacian_eigenmaps": "ours-eigen"}
    for method, name in mapping.items():
        recalls = [record.recall_at_20 for record in baseline_records if record.method == method]
        results[name] = _result(recalls, "rebuilt", [])

    payload = {
        "dataset": dataset, "train_examples": len(train_examples),
        "evaluation_examples": len(evaluation_examples), "epochs": epochs,
        "results": results,
        "warning": "Small-sample engineering validation; not a publication result.",
    }
    (output_dir / "trained_ablations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output_dir / "trained_ablations.md").write_text(
        _render(payload), encoding="utf-8"
    )
    typer.echo(json.dumps(payload, indent=2))


def _train_hypergraph(merged, training, encoder, epochs, **options):
    pipeline = QMSHEPipeline(merged.corpus, text_encoder=encoder)
    pipeline.generator.client = None
    pipeline._ablation_training_history = pipeline.train_stage_a(
        training.training_pairs, epochs=epochs,
        bridge_by_question=training.bridge_by_question, **options,
    )
    return pipeline


def _result(recalls, execution, history):
    return {
        "status": "completed", "execution": execution,
        "recall@20": sum(recalls) / max(len(recalls), 1),
        "loss_history": history,
    }


def _render(payload):
    lines = [
        "# Trained ablations", "",
        f"Train examples: {payload['train_examples']}; evaluation examples: {payload['evaluation_examples']}.",
        "", "| Variant | Execution | Recall@20 | Status |", "|---|---|---:|---|",
    ]
    for name, result in payload["results"].items():
        lines.append(
            f"| {name} | {result['execution']} | {result['recall@20']:.4f} | {result['status']} |"
        )
    lines.extend(["", f"> {payload['warning']}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)
