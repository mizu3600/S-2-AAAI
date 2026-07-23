import asyncio
import math
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class LoadTestResult:
    requests: int
    concurrency: int
    success_rate: float
    throughput_qps: float
    p50_ms: float
    p95_ms: float
    max_ms: float


async def run_load_test(pipeline, questions: list[str], requests: int = 100, concurrency: int = 8) -> LoadTestResult:
    semaphore = asyncio.Semaphore(concurrency)
    latencies, successes = [], 0

    async def one(question: str):
        nonlocal successes
        async with semaphore:
            start = perf_counter()
            try:
                await asyncio.to_thread(pipeline.query, question, 12, False)
                successes += 1
            finally:
                latencies.append((perf_counter() - start) * 1000)

    started = perf_counter()
    await asyncio.gather(*(one(questions[index % len(questions)]) for index in range(requests)))
    elapsed = perf_counter() - started
    ordered = sorted(latencies)
    return LoadTestResult(
        requests=requests, concurrency=concurrency, success_rate=successes / max(requests, 1),
        throughput_qps=requests / max(elapsed, 1e-9), p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95), max_ms=max(ordered, default=0),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, max(0, math.ceil(len(values) * quantile) - 1))]

