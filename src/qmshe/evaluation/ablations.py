import json
from dataclasses import dataclass
from pathlib import Path

from qmshe.evaluation.retrieval_metrics import recall_at_k


@dataclass(frozen=True)
class AblationSpec:
    name: str
    execution: str
    change: str


ABLATIONS = {
    "ours-low": AblationSpec("ours-low", "runtime", "retain only low-frequency block"),
    "ours-fixed": AblationSpec("ours-fixed", "runtime", "replace query gate with equal weights"),
    "ours-no-high": AblationSpec("ours-no-high", "runtime", "remove high-frequency block"),
    "ours-no-raw": AblationSpec("ours-no-raw", "runtime", "remove raw semantic residual"),
    "ours-concat": AblationSpec("ours-concat", "rebuild", "semantic plus LapPE concatenation"),
    "ours-graph": AblationSpec("ours-graph", "rebuild", "ordinary binary graph operator"),
    "ours-no-role": AblationSpec("ours-no-role", "rebuild", "single incidence matrix"),
    "ours-no-bridge-loss": AblationSpec("ours-no-bridge-loss", "retrain", "bridge loss weight zero"),
    "ours-no-hard-negative": AblationSpec("ours-no-hard-negative", "retrain", "random negatives only"),
    "ours-eigen": AblationSpec("ours-eigen", "rebuild", "explicit Laplacian eigenvectors"),
    "ours-full": AblationSpec("ours-full", "runtime", "complete model"),
}


def run_runtime_ablations(pipeline, questions: list[tuple[str, set[str]]], output: str | Path) -> dict:
    summary = {}
    for name, spec in ABLATIONS.items():
        if spec.execution != "runtime":
            summary[name] = {"status": f"requires_{spec.execution}", "change": spec.change}
            continue
        recalls = [recall_at_k(pipeline.ablation_search(question, name, 20), gold, 20) for question, gold in questions]
        summary[name] = {"status": "completed", "recall@20": sum(recalls) / max(len(recalls), 1),
                         "change": spec.change}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

