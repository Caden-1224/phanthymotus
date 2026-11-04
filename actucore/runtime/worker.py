"""The model worker process.

Started with `spawn`, never `fork`: a forked child inherits ActuCore's already
loaded CUDA and vendor runtime state, which is exactly what the process boundary
is meant to prevent. The child owns one model instance and speaks a small
command protocol over a pipe; it has no ROS or HTTP responsibilities.
"""

from __future__ import annotations

import logging
import os

from .backends import DelayBackend, MockBackend

log = logging.getLogger(__name__)


def build_backend(name: str, config: dict) -> MockBackend:
    capabilities = dict(config.get("capabilities") or {})
    if name == "mock":
        return MockBackend(capabilities, period_s=float(config.get("period_s", 8.0)))
    if name == "delay":
        return DelayBackend(capabilities,
                            delay_ms=float(config.get("delay_ms", 0)),
                            period_s=float(config.get("period_s", 8.0)))
    raise ValueError(f"unknown runtime backend {name!r}")


def worker_main(connection, backend_name: str, config: dict) -> None:
    """Entry point for the spawned child. Never returns until closed or killed."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [model-worker] %(levelname)s %(message)s")
    try:
        backend = build_backend(backend_name, config)
    except Exception as error:      # noqa: BLE001 — the parent must hear why
        connection.send(("hello", {"ok": False,
                                   "error": f"{type(error).__name__}: {error}"}))
        return

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
