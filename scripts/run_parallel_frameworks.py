from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path


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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    specs = json.loads(args.commands.read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(args.max_parallel)

    async def run(spec):
        name = str(spec["name"])
        command = [str(item) for item in spec["command"]]
        if "--llm-base-url" in command:
            command[command.index("--llm-base-url") + 1] = args.gateway_base_url
        elif spec.get("accepts_llm_base_url", True):
            command.extend(["--llm-base-url", args.gateway_base_url])
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
        env["DEEPSEEK_API_KEY"] = name
        env["DEEPSEEK_BASE_URL"] = args.gateway_base_url
        async with semaphore:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=env,
            )
            return_code = await process.wait()
        if return_code:
            raise RuntimeError(f"{name} exited with status {return_code}")

    await asyncio.gather(*(run(spec) for spec in specs))


if __name__ == "__main__":
    asyncio.run(main())
