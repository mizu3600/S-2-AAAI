from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class TimerResult:
    seconds: float = 0.0


@contextmanager
def timed():
    result = TimerResult()
    start = perf_counter()
    yield result
    result.seconds = perf_counter() - start

