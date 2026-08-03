from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run framework commands concurrently behind one DeepSeek gateway"
    )
    parser.add_argument(
        "--commands",
        type=Path,
        required=True,
        help="JSON array of {name, command:[...], env:{...}} objects",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--gateway-base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--log-dir", type=Path)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    specs = json.loads(args.commands.read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(args.max_parallel)
    status_file = args.status_file or args.commands.with_suffix(".status.json")
    log_dir = args.log_dir or args.commands.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    statuses = {
        str(spec["name"]): {
            "status": "pending",
            "command": [str(item) for item in spec["command"]],
        }
        for spec in specs
    }
    write_status(status_file, statuses)

    with httpx.Client(timeout=10) as client:
        response = client.get(
            args.gateway_base_url.removesuffix("/v1").rstrip("/") + "/health"
        )
        response.raise_for_status()

    async def run(spec):
        name = str(spec["name"])
        command = [str(item) for item in spec["command"]]
        if "--llm-base-url" in command:
            command[command.index("--llm-base-url") + 1] = args.gateway_base_url
        elif spec.get("accepts_llm_base_url", True):
            command.extend(["--llm-base-url", args.gateway_base_url])
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
        env["DEEPSEEK_API_KEY"] = str(spec.get("framework") or name)
        env["DEEPSEEK_BASE_URL"] = args.gateway_base_url
        async with semaphore:
            log_path = log_dir / f"{name}.log"
            statuses[name].update(
                {
                    "status": "running",
                    "started_at_unix": time.time(),
                    "log_path": str(log_path.resolve()),
                    "command": command,
                }
            )
            write_status(status_file, statuses)
            with log_path.open("ab") as log:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    env=env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                return_code = await process.wait()
            statuses[name].update(
                {
                    "status": "success" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "finished_at_unix": time.time(),
                }
            )
            write_status(status_file, statuses)
        return name, return_code

    results = await asyncio.gather(*(run(spec) for spec in specs))
    failures = [name for name, return_code in results if return_code]
    if failures:
        raise RuntimeError(
            f"{len(failures)} framework commands failed: {', '.join(failures)}"
        )


def write_status(path: Path, statuses: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    asyncio.run(main())
