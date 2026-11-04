"""The model worker process.

Started with `spawn`, never `fork`: a forked child inherits ActuCore's already
loaded CUDA and vendor runtime state, which is exactly what the process boundary
is meant to prevent. The child owns one model instance and speaks a small
command protocol over a pipe; it has no ROS or HTTP responsibilities.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from multiprocessing.connection import Connection

from .backends import DelayBackend, MockBackend, SmolVLABackend

log = logging.getLogger(__name__)


def build_backend(name: str, config: dict) -> MockBackend | SmolVLABackend:
    capabilities = dict(config.get("capabilities") or {})
    if name == "mock":
        return MockBackend(capabilities, period_s=float(config.get("period_s", 8.0)))
    if name == "delay":
        return DelayBackend(capabilities,
                            delay_ms=float(config.get("delay_ms", 0)),
                            period_s=float(config.get("period_s", 8.0)))
    if name == "smolvla":
        return SmolVLABackend(config)
    raise ValueError(f"unknown runtime backend {name!r}")


def _watch_health(backend: MockBackend | SmolVLABackend,
                  connection: Connection) -> None:
    """Publish `health()` changes to the parent while the model loads.

    A separate pipe rather than a request: the card asks `/healthz` every tick,
    and queueing a health probe behind an inference would make a loading model
    look like a hung gateway. The worker owns the transition, so it pushes it.
    """
    last = None
    while True:
        try:
            healthy = bool(backend.health())
        except Exception:       # noqa: BLE001 — a probe must not kill the watcher
            healthy = False
        if healthy != last:
            try:
                connection.send(("health", healthy))
            except (BrokenPipeError, EOFError, OSError):
                return
            last = healthy
        time.sleep(0.1)


def worker_main(connection: Connection, backend_name: str, config: dict,
                health_connection: Connection | None = None) -> None:
    """Entry point for the spawned child. Never returns until closed or killed."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [model-worker] %(levelname)s %(message)s")
    try:
        backend = build_backend(backend_name, config)
    except Exception as error:      # noqa: BLE001 — the parent must hear why
        connection.send(("hello", {"ok": False,
                                   "error": f"{type(error).__name__}: {error}"}))
        return

    if health_connection is not None:
        threading.Thread(target=_watch_health, args=(backend, health_connection),
                         name="model_worker_health", daemon=True).start()

    connection.send(("hello", {"ok": True, "pid": os.getpid(),
                               "capabilities": backend.capabilities()}))
    try:
        while True:
            message = connection.recv()
            if message[0] == "infer":
                _, request_id, request = message
                try:
                    actions = backend.infer(request)
                except Exception as error:      # noqa: BLE001
                    connection.send(("error", request_id,
                                     f"{type(error).__name__}: {error}"))
                else:
                    connection.send(("result", request_id, actions))
            elif message[0] == "close":
                break
    except (EOFError, ConnectionResetError):
        pass
    finally:
        backend.close()
