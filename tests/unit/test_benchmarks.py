import json

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.experiment import BenchmarkExperimentRunner


def test_hotpot_adapter_and_experiment(tmp_path):
    sample = [{
        "_id": "sample-1", "question": "What links Alpha and Beta?", "answer": "Bridge",
        "type": "bridge", "level": "easy",
        "context": [
            ["Alpha", ["Alpha mentions Bridge.", "Distractor sentence."]],
            ["Beta", ["Bridge connects to Beta."]],
        ],
        "supporting_facts": [["Alpha", 0], ["Beta", 0]],
    }]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    suite = load_benchmark("hotpotqa", path)
    assert suite.examples[0].hop_count == 2
    built = build_example_corpus(suite.examples[0])
    assert len(built.gold_fact_ids) == 2
    records = BenchmarkExperimentRunner().run(suite, tmp_path / "results")
    assert len(records) == 11
    assert (tmp_path / "results" / "report.md").exists()
