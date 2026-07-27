import math
from collections import defaultdict
from threading import Lock


class RuntimeMetrics:
    def __init__(self):
        self._lock = Lock()
        self.queries = 0
        self.no_evidence = 0
        self.latencies_ms: list[float] = []
        self.citations = 0
        self.band_totals = defaultdict(float)

    def observe(self, result, latency_ms: float) -> None:
        with self._lock:
            self.queries += 1
            self.no_evidence += int(not result.retrieved_facts)
            self.latencies_ms.append(latency_ms)
            self.citations += len(result.citations)
            for name, value in result.band_weights.items():
                self.band_totals[name] += value

    def snapshot(self) -> dict:
        with self._lock:
            ordered = sorted(self.latencies_ms)
            queries = max(self.queries, 1)
            return {
                "queries": self.queries,
                "no_evidence_rate": self.no_evidence / queries,
                "average_citations": self.citations / queries,
                "p50_latency_ms": _percentile(ordered, 0.50),
                "p95_latency_ms": _percentile(ordered, 0.95),
                "average_band_weights": {name: value / queries for name, value in self.band_totals.items()},
            }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, math.ceil(len(values) * quantile) - 1)]
