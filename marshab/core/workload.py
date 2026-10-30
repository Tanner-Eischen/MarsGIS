"""One bounded scientific workload per application process, without a queue."""

import threading

_slot = threading.BoundedSemaphore(1)


class WorkloadBusyError(RuntimeError):
    """Another scientific job already owns the process's worker slot."""


def acquire_workload() -> bool:
    return _slot.acquire(blocking=False)


def release_workload() -> None:
    _slot.release()
