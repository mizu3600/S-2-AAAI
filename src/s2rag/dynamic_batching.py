from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
import threading
from time import monotonic
from typing import Generic, TypeVar, cast


Payload = TypeVar("Payload")
Result = TypeVar("Result")


@dataclass(slots=True)
class _BatchRequest(Generic[Payload, Result]):
    payload: Payload
    item_count: int
    future: Future[Result]


class DynamicBatcher(Generic[Payload, Result]):
    """Merge concurrent requests while keeping one model owner thread."""

    def __init__(
        self,
        handler: Callable[[list[Payload]], list[Result]],
        *,
        max_items: int,
        wait_seconds: float,
        name: str,
    ):
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if wait_seconds < 0:
            raise ValueError("wait_seconds cannot be negative")
        self._handler = handler
        self._max_items = max_items
        self._wait_seconds = wait_seconds
        self._queue: Queue[_BatchRequest[Payload, Result] | object] = Queue()
        self._stop = object()
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Payload, *, item_count: int) -> Result:
        if item_count < 1:
            raise ValueError("item_count must be positive")
        request = _BatchRequest(
            payload=payload,
            item_count=item_count,
            future=Future(),
        )
        with self._close_lock:
            if self._closed:
                raise RuntimeError("dynamic batcher is closed")
            self._queue.put(request)
        return request.future.result()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(self._stop)
        self._thread.join()

    def _run(self) -> None:
        carry: _BatchRequest[Payload, Result] | None = None
        stopping = False
        while not stopping:
            if carry is None:
                queued = self._queue.get()
                if queued is self._stop:
                    break
                first = cast(_BatchRequest[Payload, Result], queued)
            else:
                first = carry
                carry = None

            requests = [first]
            item_count = first.item_count
            deadline = monotonic() + self._wait_seconds
            while item_count < self._max_items:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    queued = self._queue.get(timeout=remaining)
                except Empty:
                    break
                if queued is self._stop:
                    stopping = True
                    break
                request = cast(_BatchRequest[Payload, Result], queued)
                if (
                    requests
                    and item_count + request.item_count > self._max_items
                ):
                    carry = request
                    break
                requests.append(request)
                item_count += request.item_count

            self._dispatch(requests)

        if carry is not None:
            carry.future.set_exception(RuntimeError("dynamic batcher is closed"))
        while True:
            try:
                queued = self._queue.get_nowait()
            except Empty:
                break
            if queued is not self._stop:
                cast(_BatchRequest[Payload, Result], queued).future.set_exception(
                    RuntimeError("dynamic batcher is closed")
                )

    def _dispatch(
        self,
        requests: list[_BatchRequest[Payload, Result]],
    ) -> None:
        try:
            results = self._handler([request.payload for request in requests])
            if len(results) != len(requests):
                raise RuntimeError(
                    "dynamic batch handler returned a different result count"
                )
        except BaseException as exc:
            for request in requests:
                request.future.set_exception(exc)
            return
        for request, result in zip(requests, results, strict=True):
            request.future.set_result(result)
