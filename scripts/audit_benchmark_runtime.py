from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the shared DeepSeek, embedding and reranker services."
    )
    parser.add_argument("--gateway", default="http://127.0.0.1:8020")
    parser.add_argument("--embedding", default="http://127.0.0.1:18080")
    parser.add_argument("--reranker", default="http://127.0.0.1:18081")
    parser.add_argument("--embedding-container", default="ureval-bge-m3")
    parser.add_argument(
        "--reranker-container",
        default="ureval-bge-reranker-v2-m3",
    )
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--reranker-model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def get_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"response": payload}


def container_inspect(name: str) -> dict:
    output = subprocess.run(
        ["docker", "inspect", name],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = json.loads(output)
    if len(rows) != 1:
        raise ValueError(f"expected one container named {name}, found {len(rows)}")
    row = rows[0]
    return {
        "id": row["Id"],
        "image": row["Image"],
        "state": row["State"],
        "args": row.get("Args", []),
        "mounts": row.get("Mounts", []),
        "device_requests": row.get("HostConfig", {}).get("DeviceRequests", []),
    }


def model_weight_hash(path: Path | None) -> dict | None:
    if path is None:
        return None
    files = sorted(path.rglob("*.safetensors"))
    if not files:
        files = sorted(path.rglob("pytorch_model*.bin"))
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path)
        digest.update(str(relative).encode("utf-8"))
        with file_path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
                total_bytes += len(block)
    return {
        "path": str(path.resolve()),
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def gpu_processes() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid, process_name, used_memory = [
            item.strip() for item in line.split(",", 3)
        ]
        rows.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    with httpx.Client(timeout=30) as client:
        gateway = get_json(client, f"{args.gateway.rstrip('/')}/health")
        embedding = get_json(client, f"{args.embedding.rstrip('/')}/info")
        reranker = get_json(client, f"{args.reranker.rstrip('/')}/info")
        probe = client.post(
            f"{args.embedding.rstrip('/')}/v1/embeddings",
            headers={"Authorization": "Bearer local-tei"},
            json={"model": "BAAI/bge-m3", "input": ["runtime audit"]},
        )
        probe.raise_for_status()
        embedding_dimension = len(probe.json()["data"][0]["embedding"])
        rerank_probe = client.post(
            f"{args.reranker.rstrip('/')}/rerank",
            json={"query": "runtime audit", "texts": ["runtime audit", "other"]},
        )
        rerank_probe.raise_for_status()

    report = {
        "audited_at_unix": time.time(),
        "gateway": gateway,
        "embedding": {
            "health": embedding,
            "dimension": embedding_dimension,
            "container": container_inspect(args.embedding_container),
            "weights": model_weight_hash(args.embedding_model_path),
        },
        "reranker": {
            "health": reranker,
            "probe": rerank_probe.json(),
            "container": container_inspect(args.reranker_container),
            "weights": model_weight_hash(args.reranker_model_path),
        },
        "gpu_processes": gpu_processes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote runtime audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
