from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FRAMEWORKS = (
    "graphrag",
    "lightrag",
    "pathrag",
    "hypergraphrag",
    "hipporag2",
    "cograg",
    "hgrag",
    "hyperrag",
)

REPOSITORIES = {
    "graphrag": "GraphRAG",
    "lightrag": "LightRAG",
    "pathrag": "PathRAG",
    "hypergraphrag": "HyperGraphRAG",
    "hipporag2": "HippoRAG2",
    "cograg": "Cog-RAG",
    "hgrag": "HGRAG",
    "hyperrag": "Hyper-RAG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the complete eight-framework native command manifest."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--framework-venv-root",
        type=Path,
        help="Optional root containing <framework>-linux virtual environments.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument(
        "--runtime-site",
        type=Path,
        help="Optional directory containing official runtime-only dependencies.",
    )
    parser.add_argument(
        "--optional-runtime-site",
        type=Path,
        help="Optional isolated site for import-only dependencies such as ollama.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    wrappers = project / "scripts" / "official_baselines"
    repos = project / "third_party" / "official_baselines"
    suite_sha256 = sha256_file(args.suite)
    common = [
        "--input",
        str(args.suite.resolve()),
        "--dataset",
        args.dataset,
        "--seed",
        str(args.seed),
        "--expected-count",
        str(args.expected_count),
        "--suite-sha256",
        suite_sha256,
        "--llm-base-url",
        "http://127.0.0.1:8020/v1",
        "--embedding-base-url",
        "http://127.0.0.1:18080/v1",
        "--resume",
    ]
    specs = []
    for framework in FRAMEWORKS:
        repo = repos / REPOSITORIES[framework]
        framework_python = args.python
        framework_site_packages = None
        if args.framework_venv_root:
            venv_name = {
                "graphrag": "graphrag-linux",
                "lightrag": "lightrag-linux",
                "pathrag": "pathrag-linux",
                "hypergraphrag": "hypergraphrag-linux",
            }.get(framework)
            candidate = (
                args.framework_venv_root / venv_name / "bin" / "python"
                if venv_name
                else None
            )
            if (
                candidate is not None
                and candidate.exists()
                and framework in {"graphrag", "lightrag"}
            ):
                framework_python = candidate
            elif candidate is not None and candidate.exists():
                site_candidates = sorted(
                    (candidate.parent.parent / "lib").glob(
                        "python*/site-packages"
                    )
                )
                if site_candidates:
                    framework_site_packages = site_candidates[-1]
        output_dir = args.output_root / args.dataset / framework
        output = output_dir / "retrieval.jsonl"
        work_dir = output_dir / "artifacts"
        cache = args.output_root / "cache" / f"{framework}.sqlite3"
        if framework == "graphrag":
            command = [
                str(framework_python),
                str(wrappers / "run_graphrag_official.py"),
                "--official-repo",
                str(repo),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--example-concurrency",
                "1",
                "--internal-concurrency",
                "8",
                *common,
            ]
        elif framework == "lightrag":
            command = [
                str(framework_python),
                str(wrappers / "run_lightrag_official.py"),
                "--official-repo",
                str(repo),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--llm-cache",
                str(cache),
                "--example-concurrency",
                "2",
                "--global-llm-concurrency",
                "16",
                *common,
            ]
        elif framework in {"pathrag", "hypergraphrag"}:
            command = [
                str(framework_python),
                str(wrappers / "run_path_hyper_official.py"),
                "--framework",
                framework,
                "--official-repo",
                str(repo),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--llm-cache",
                str(cache),
                "--example-concurrency",
                "2",
                "--global-llm-concurrency",
                "16",
                *common,
            ]
        else:
            command = [
                str(framework_python),
                str(wrappers / "run_native_official.py"),
                "--framework",
                framework,
                "--official-repo",
                str(repo),
                "--output",
                str(output),
                "--work-dir",
                str(work_dir),
                "--llm-cache",
                str(cache),
                "--example-concurrency",
                "2",
                "--global-llm-concurrency",
                "16",
                *common,
            ]
        python_path = [str(wrappers), str(project / "src"), str(repo)]
        if args.runtime_site:
            python_path.insert(0, str(args.runtime_site.resolve()))
        if (
            args.optional_runtime_site
            and framework in {"pathrag", "hypergraphrag"}
        ):
            python_path.insert(0, str(args.optional_runtime_site.resolve()))
        if framework_site_packages:
            python_path.insert(1, str(framework_site_packages.resolve()))
        specs.append(
            {
                "name": f"{args.dataset}-{framework}",
                "framework": framework,
                "official_repo": str(repo),
                "official_commit": git_commit(repo),
                "command": command,
                "accepts_llm_base_url": True,
                "env": {"PYTHONPATH": ":".join(python_path)},
            }
        )
    args.commands.parent.mkdir(parents=True, exist_ok=True)
    args.commands.write_text(
        json.dumps(specs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(specs)} native commands to {args.commands}", flush=True)


def git_commit(repo: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
