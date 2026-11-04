"""Owns the model worker process.

v1 does not restart a dead worker — the next ticket adds crash detection,
`recovering` and restart. What this class must get right now is narrower and
non-negotiable: the gateway stays alive when the child dies, and an inference
waits with a deadline instead of blocking forever.
"""

from __future__ import annotations

import multiprocessing as mp
import threading

from .worker import worker_main


class WorkerError(RuntimeError):
    """The worker is unavailable or returned an error."""


class WorkerTimeout(WorkerError):
    """The worker did not answer within the request deadline."""


class WorkerSupervisor:
    def __init__(self, backend: str, config: dict, timeout_ms: float = 5000):
        self._backend = backend
        self._config = dict(config)
        self._timeout_s = float(timeout_ms) / 1000.0
        self._lock = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._next_request_id = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=worker_main,
            args=(child, self._backend, self._config),
            daemon=True,
            name=f"model_worker_{self._backend}",
        )
        process.start()
        child.close()
        self._process, self._connection = process, parent

        if not parent.poll(30):
            self._terminate()
            raise WorkerError("the model worker did not start within 30s")
        try:
            message = parent.recv()
        except EOFError as error:
            self._terminate()
            raise WorkerError(f"the model worker exited before hello: {error}") from error
        if message[0] != "hello" or not message[1].get("ok"):
            detail = message[1].get("error", "no hello") if message[0] == "hello" else "bad hello"
            self._terminate()
            raise WorkerError(f"the model worker failed to start: {detail}")

    def stop(self) -> None:
        with self._lock:
            connection, process = self._connection, self._process
            self._connection = self._process = None
            if connection is not None:
                try:
                    connection.send(("close",))
                except Exception:       # noqa: BLE001 — teardown is best effort
                    pass
                connection.close()
            if process is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

    def _terminate(self) -> None:
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            connection.close()
        if process is not None:
            process.terminate()
            process.join(timeout=5)

    # ── requests ─────────────────────────────────────────────────────────────

    def request(self, payload: dict):
        with self._lock:
            if not self.alive:
                raise WorkerError(f"model worker is {self.status()['state']}")
            self._next_request_id += 1
            request_id = self._next_request_id
            try:
                self._connection.send(("infer", request_id, payload))
                if not self._connection.poll(self._timeout_s):
                    raise WorkerTimeout(
                        f"model worker did not answer within {self._timeout_s:g}s")
                kind, reply_id, body = self._connection.recv()
            except WorkerTimeout:
                # The reply may still arrive later and would be read as the
                # answer to the next request, so a timed-out worker is not
                # reusable. Hard cancel/restart is the next ticket's job.
                self._terminate()
                raise
            except (EOFError, OSError) as error:
                raise WorkerError(f"model worker exited: {error}") from error
            if reply_id != request_id:
                raise WorkerError(f"worker reply {reply_id} does not match request {request_id}")
            if kind != "result":
                raise WorkerError(str(body))
            return body

    @property
    def start_method(self) -> str:
        return self._ctx.get_start_method()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def status(self) -> dict:
        if self._process is None:
            state = "stopped"
        elif self.alive:
            state = "ready"
        else:
            state = "dead"
        return {"state": state, "pid": self.pid, "restart_count": 0}
