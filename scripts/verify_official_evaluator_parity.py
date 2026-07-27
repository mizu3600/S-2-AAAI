from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from s2rag.evaluation.metrics import answer_scores
from s2rag.evaluation.official_metrics import score_official_metrics


ROOT = Path("third_party/official_evaluators").resolve()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        results = {
            "hotpotqa": _verify_hotpot(temp_dir),
            "musique": _verify_musique(temp_dir),
            "2wikimultihopqa": _verify_2wiki(temp_dir),
        }
    print(json.dumps(results, indent=2))


def _verify_hotpot(temp_dir: Path) -> dict:
    prediction = {"answer": {"q1": "Ada"}, "sp": {"q1": [["Result", 0]]}}
    gold = [
        {
            "_id": "q1",
            "answer": "Ada",
            "supporting_facts": [["Result", 0]],
        }
    ]
    pred_path = _write_json(temp_dir / "hotpot_pred.json", prediction)
    gold_path = _write_json(temp_dir / "hotpot_gold.json", gold)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "hotpotqa" / "hotpot_evaluate_v1.py"),
            str(pred_path),
            str(gold_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    upstream = ast.literal_eval(completed.stdout.strip().splitlines()[-1])
    local = _local_scores("hotpotqa")
    _assert_close(upstream["em"], local["official_answer_em"])
    _assert_close(upstream["f1"], local["official_answer_f1"])
    _assert_close(upstream["sp_em"], local["official_support_em"])
    _assert_close(upstream["sp_f1"], local["official_support_f1"])
    _assert_close(upstream["joint_em"], local["official_joint_em"])
    _assert_close(upstream["joint_f1"], local["official_joint_f1"])
    return {"status": "matched", "script_metrics": upstream}


def _verify_musique(temp_dir: Path) -> dict:
    prediction = [
        {
            "id": "q1",
            "predicted_answer": "Ada Lovelace",
            "predicted_support_idxs": [0],
            "predicted_answerable": True,
        }
    ]
    gold = [
        {
            "id": "q1",
            "answer": "Augusta Ada King",
            "answer_aliases": ["Ada Lovelace"],
            "answerable": True,
            "paragraphs": [{"idx": 0, "is_supporting": True}],
        }
    ]
    pred_path = _write_jsonl(temp_dir / "musique_pred.jsonl", prediction)
    gold_path = _write_jsonl(temp_dir / "musique_gold.jsonl", gold)
    output_path = temp_dir / "musique_metrics.json"
    script = ROOT / "musique" / "evaluate_v1.0.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(pred_path),
            str(gold_path),
            "--output_filepath",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=script.parent,
    )
    upstream = json.loads(output_path.read_text(encoding="utf-8"))
    answer = answer_scores(
        "Ada Lovelace",
        ["Augusta Ada King", "Ada Lovelace"],
        profile="musique_official",
    )
    local = _local_scores("musique", answer=answer)
    _assert_close(upstream["answer_em"], local["official_answer_em"])
    _assert_close(upstream["answer_f1"], local["official_answer_f1"])
    _assert_close(upstream["support_f1"], local["official_support_f1"])
    return {"status": "matched", "script_metrics": upstream}


def _verify_2wiki(temp_dir: Path) -> dict:
    prediction = {
        "answer": {"q1": "Ada Lovelace"},
        "sp": {"q1": [["Result", 0]]},
        "evidence": {"q1": [["Ada", "born in", "London"]]},
    }
    gold = [
        {
            "_id": "q1",
            "answer": "Ada",
            "answer_id": "Q1",
            "supporting_facts": [["Result", 0]],
            "evidences": [["Ada", "born in", "London"]],
            "evidences_id": [],
        }
    ]
    aliases = [{"Q_id": "Q1", "aliases": ["Ada Lovelace"], "demonyms": []}]
    pred_path = _write_json(temp_dir / "2wiki_pred.json", prediction)
    gold_path = _write_json(temp_dir / "2wiki_gold.json", gold)
    alias_path = _write_jsonl(temp_dir / "2wiki_aliases.json", aliases)
    completed = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "2wikimultihopqa"
                / "2wikimultihop_evaluate_v1.1.py"
            ),
            str(pred_path),
            str(gold_path),
            str(alias_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    upstream = json.loads(completed.stdout)
    answer = answer_scores(
        "Ada Lovelace",
        ["Ada", "Ada Lovelace"],
        profile="2wikimultihopqa_official",
    )
    local = _local_scores("2wikimultihopqa", answer=answer)
    _assert_close(upstream["em"] / 100, local["official_answer_em"])
    _assert_close(upstream["f1"] / 100, local["official_answer_f1"])
    _assert_close(upstream["sp_em"] / 100, local["official_support_em"])
    _assert_close(upstream["sp_f1"] / 100, local["official_support_f1"])
    return {
        "status": "matched_answer_and_support",
        "script_metrics": upstream,
        "local_evidence_and_joint": "N/A until official triples are emitted",
    }


def _local_scores(dataset: str, *, answer: dict | None = None) -> dict:
    return score_official_metrics(
        dataset=dataset,
        answer_metric=answer
        or answer_scores("Ada", "Ada", profile="hotpotqa_official"),
        predicted_sentence_ids=["s1"],
        gold_sentence_ids={"s1"},
        predicted_passage_ids=["p1"],
        gold_passage_ids={"p1"},
        answer_available=True,
        sentence_support_available=True,
        passage_support_available=True,
    )


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _assert_close(left, right) -> None:
    if abs(float(left) - float(right)) > 1e-9:
        raise AssertionError(f"official evaluator mismatch: {left} != {right}")


if __name__ == "__main__":
    main()
