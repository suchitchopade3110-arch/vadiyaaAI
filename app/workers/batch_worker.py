"""Threaded batch worker utilities."""

from queue import Queue
from threading import Thread
from typing import Callable, Iterable


def run_batch(items: Iterable, handler: Callable, num_workers: int = 4) -> list:
    """Process items with a small Queue + Thread pool, preserving input order."""
    items = list(items)
    queue: Queue = Queue()
    results = [None] * len(items)
    errors = []

    for index, item in enumerate(items):
        queue.put((index, item))

    def worker():
        while True:
            try:
                index, item = queue.get_nowait()
            except Exception:
                return
            try:
                results[index] = handler(item)
            except Exception as exc:
                errors.append((index, exc))
            finally:
                queue.task_done()

    threads = [Thread(target=worker, daemon=True) for _ in range(max(1, num_workers))]
    for thread in threads:
        thread.start()
    queue.join()
    for thread in threads:
        thread.join(timeout=0.1)

    if errors:
        index, exc = errors[0]
        raise RuntimeError(f"Batch item {index} failed") from exc
    return results


__all__ = ["Queue", "Thread", "run_batch"]
