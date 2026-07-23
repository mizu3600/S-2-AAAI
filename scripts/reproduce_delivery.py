import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import typer

from generate_math_report import main as generate_math_report
from run_trained_ablations import main as run_trained_ablations
from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.ablations import run_runtime_ablations
from qmshe.evaluation.dual_mode import DualModeExperimentRunner
from qmshe.evaluation.experiment import BenchmarkExperimentRunner, LocalBenchmarkEncoder
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline, load_corpus
from qmshe.scaling.load_test import run_load_test


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("reports/delivery")),
    limit: int = typer.Option(5),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_math_report(output_dir / "math")
    suite = load_benchmark(dataset, input_path, limit=limit)
    embedding_dir = output_dir / "embedding_baselines"
    records = BenchmarkExperimentRunner().run(suite, embedding_dir)
    DualModeExperimentRunner().run(suite, output_dir / "dual_mode")

    synthetic = load_corpus("data/processed/synthetic.json")
    encoder = LocalBenchmarkEncoder(128)
    hypergraph = QMSHEPipeline(synthetic, text_encoder=encoder)
    hypergraph.generator.client = None
    entity_graph = QMSGEGraphPipeline(
        synthetic, text_encoder=encoder, profile=GraphProfile.ENTITY_RELATION
    )
    reified_graph = QMSGEGraphPipeline(
        synthetic, text_encoder=encoder, profile=GraphProfile.REIFIED_FACT
    )
    for pipeline in (entity_graph, reified_graph):
        pipeline.generator.client = None
    questions = [
        "How does PEAI improve Voc?",
        "What reduces non-radiative recombination?",
        "What affects long-term stability?",
        "Summarize the device mechanisms and conditions.",
    ]
    gate_rows = []
    for question in questions:
        for mode, pipeline in (
            ("hypergraph", hypergraph),
            ("graph_entity_relation", entity_graph),
            ("graph_reified_fact", reified_graph),
        ):
            result = pipeline.query(question, top_k=4, return_debug=False)
            gate_rows.append({"question": question, "mode": mode, **result.band_weights})
    (output_dir / "gate_weights.json").write_text(
        json.dumps(gate_rows, indent=2), encoding="utf-8"
    )
    _write_gate_svg(output_dir / "gate_weights.svg", gate_rows)

    run_runtime_ablations(
        hypergraph,
        [("How does PEAI improve Voc?", {"fact_1", "fact_2", "fact_3"})],
        output_dir / "ablations.json",
    )
    run_trained_ablations(
        dataset=dataset, input_path=input_path,
        output_dir=output_dir / "trained_ablations", limit=limit, epochs=2,
    )
    load_started = perf_counter()
    cold = asyncio.run(run_load_test(
        hypergraph, [f"PEAI Voc request {index}" for index in range(100)], 100, 8
    ))
    warm = asyncio.run(run_load_test(
        hypergraph, questions, 100, 8
    ))
    efficiency = {
        "cold_cache": asdict(cold), "warm_cache": asdict(warm),
        "total_load_test_seconds": perf_counter() - load_started,
    }
    (output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, indent=2), encoding="utf-8"
    )
    _write_cases(output_dir / "cases.md", suite, records)
    _write_delivery_report(output_dir, suite, efficiency)
    typer.echo(f"delivery artifacts written to {output_dir}")


def _write_gate_svg(path: Path, rows: list[dict]) -> None:
    width, row_height, left, bar_width = 920, 30, 300, 560
    height = 45 + row_height * len(rows)
    colors = {"raw": "#666666", "low": "#2f6fed", "mid": "#ef8a17", "high": "#c33c54"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for index, row in enumerate(rows):
        y = 25 + index * row_height
        label = f"{row['mode']}: {row['question'][:27]}"
        lines.append(
            f'<text x="10" y="{y + 15}" font-family="sans-serif" font-size="11">{_escape(label)}</text>'
        )
        x = left
        for name in ("raw", "low", "mid", "high"):
            segment = float(row[name]) * bar_width
            lines.append(
                f'<rect x="{x:.2f}" y="{y}" width="{segment:.2f}" height="18" fill="{colors[name]}"><title>{name}: {row[name]:.4f}</title></rect>'
            )
            x += segment
    for index, name in enumerate(("raw", "low", "mid", "high")):
        lines.append(
            f'<text x="{left + index * 80}" y="15" fill="{colors[name]}" font-family="sans-serif" font-size="12">{name}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cases(path: Path, suite, records) -> None:
    by_example = {
        record.example_id: record for record in records if record.method == "qmshe"
    }
    ordered = sorted(by_example.values(), key=lambda item: (item.recall_at_20, item.mrr))
    example_by_id = {example.example_id: example for example in suite.examples}
    selected = [("Failure", item) for item in ordered[:2]] + [
        ("Success", item) for item in ordered[-2:]
    ]
    lines = ["# Retrieval cases", ""]
    for label, record in selected:
        example = example_by_id[record.example_id]
        lines.extend([
            f"## {label}: {record.example_id}", "",
            f"- Question: {example.question}",
            f"- Hop count: {record.hop_count}",
            f"- Query type: {record.query_type}",
            f"- Recall@20: {record.recall_at_20:.4f}",
            f"- MRR: {record.mrr:.4f}", "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_delivery_report(output: Path, suite, efficiency: dict) -> None:
    cold, warm = efficiency["cold_cache"], efficiency["warm_cache"]
    lines = [
        "# QMSxE-RAG reproducible delivery report", "",
        f"Dataset: {suite.name}; examples: {len(suite.examples)}; source: `{suite.source}`.", "",
        "## Deliverables", "",
        "| Deliverable | Artifact |",
        "|---|---|",
        "| Mathematical correctness | [Spectral report](math/spectral_validation.md) |",
        "| Embedding baseline main table | [Baseline report](embedding_baselines/report.md) |",
        "| Hop and query-type tables | [Grouped baseline report](embedding_baselines/report.md) |",
        "| Graph/Hypergraph comparison | [Dual-mode report](dual_mode/report.md) |",
        "| Ablations | [Ablation results](ablations.json) |",
        "| Rebuilt/retrained ablations | [Trained ablations](trained_ablations/trained_ablations.md) |",
        "| Efficiency | [Efficiency metrics](efficiency.json) |",
        "| Band-gate visualization | [Gate weights](gate_weights.svg) |",
        "| Success and failure cases | [Cases](cases.md) |", "",
        "## Efficiency summary", "",
        "| Run | Requests | Success rate | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|",
        f"| Cold | {cold['requests']} | {cold['success_rate']:.4f} | {cold['p50_ms']:.3f} | {cold['p95_ms']:.3f} |",
        f"| Warm | {warm['requests']} | {warm['success_rate']:.4f} | {warm['p50_ms']:.3f} | {warm['p95_ms']:.3f} |",
        "", "## Reproduction", "",
        "```bash",
        "python scripts/reproduce_delivery.py --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json --limit 5",
        "```", "",
        "System-level LightRAG/PathRAG/GraphRAG results are imported separately from the isolated "
        "Unified-RAG-Evaluation run; smoke results must not be described as publication claims.", "",
    ]
    (output / "DELIVERY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    typer.run(main)
