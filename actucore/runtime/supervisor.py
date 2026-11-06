"""Owns the model worker process and its recovery loop.

The gateway must survive the child: a crash marks the worker `recovering`, a
monitor notices a dead process and spawns a replacement, and sessions are only
returned to `ready` once that replacement's model is healthy. Idle workers are
reaped on a configured timer so a stopped card does not keep a GPU context.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time

from .worker import worker_main

log = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """The worker is unavailable or returned an error."""


class WorkerTimeout(WorkerError):
    """The worker did not answer within the request deadline."""


class WorkerRecovering(WorkerError):
    """The worker is being replaced; the caller should retry after recovery."""


class WorkerLoading(WorkerError):
    """The worker process is up but its model is still loading."""


class BackendError(WorkerError):
    """The worker answered, but the model itself failed."""


class WorkerSupervisor:
    def __init__(self, backend: str, config: dict, timeout_ms: float = 5000,
                 idle_reap_ms: float = 60000, monitor_interval_ms: float = 100,
                 on_recovering=None, on_ready=None):
        self._backend = backend
        self._config = dict(config)
        self._timeout_s = float(timeout_ms) / 1000.0
        self._idle_reap_ms = max(0.0, float(idle_reap_ms))
        self._monitor_interval_s = max(0.02, float(monitor_interval_ms) / 1000.0)
        self._on_recovering = on_recovering
        self._on_ready = on_ready

        self._lock = threading.Lock()          # process/state lifecycle only
        self._request_lock = threading.Lock()  # one in-flight child request
        self._queue_lock = threading.Lock()    # placement queue-depth counter
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._health_connection = None
        self._healthy = threading.Event()
        self._capabilities = {}
        self._next_request_id = 0

        self._state = "stopped"
        self._restart_count = 0
        self._recovery_started = 0.0
        self._recovery_time_ms = None
        self._ready_notified = True
        self._last_error = ""
        self._last_activity = time.monotonic()
        self._in_flight = False
        self._outstanding = 0       # requests queued or executing on this worker
        self._reaping = False
        self._stop_event = threading.Event()
        self._closed = False
        self._monitor = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            self._closed = False
            self._stop_event.clear()
            self._spawn_locked()
            self._state = "ready"
            self._last_activity = time.monotonic()
            self._ready_notified = self.healthy
        self._monitor = threading.Thread(target=self._monitor_loop,
                                         name="model_worker_monitor", daemon=True)
        self._monitor.start()

    def stop(self) -> None:
        self._closed = True
        self._stop_event.set()
        monitor, self._monitor = self._monitor, None
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=2)
        with self._lock:
            self._terminate_locked()
            self._state = "stopped"

    def kill(self, reason: str) -> None:
        """Hard cancel: terminate the child now; the monitor starts its replacement."""
        with self._lock:
            self._reaping = False
            self._enter_recovering_locked(reason)
            self._terminate_locked(graceful=False)

    def reap(self) -> None:
        """Release the child but keep the supervisor able to start another."""
        with self._lock:
            self._reaping = True
            self._terminate_locked()
            self._state = "stopped"
            self._last_error = "worker closed"

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self._monitor_interval_s):
            with self._lock:
                if self._closed:
                    return
                self._drain_health_locked()
                if self._state == "recovering":
                    self._retry_recovery_locked()
                elif self._state == "ready":
                    if self._process is None or not self._process.is_alive():
                        self._enter_recovering_locked("model worker exited")
                        self._terminate_locked()
                    elif not self._ready_notified:
                        # A replacement can answer hello before its weights are
                        # loaded. Sessions stay `recovering` until health is
                        # true, which is what stops a card acting on a worker
                        # that cannot infer yet.
                        self._mark_ready_locked()
                    elif self._idle_expired_locked():
                        self._terminate_locked()
                        self._state = "stopped"
                        self._last_error = "idle worker reaped"
                        log.info("[runtime] idle model worker reaped")

    def _retry_recovery_locked(self) -> None:
        self._terminate_locked()
        try:
            self._spawn_locked()
        except WorkerError as error:
            self._last_error = str(error)
            return
        self._state = "ready"
        self._ready_notified = False
        self._last_activity = time.monotonic()
        self._mark_ready_locked()

    def _mark_ready_locked(self) -> None:
        """Publish readiness once the model, not just the process, is up."""
        if self._ready_notified or not self.healthy:
            return
        self._ready_notified = True
        self._last_activity = time.monotonic()
        if self._recovery_started:
            self._recovery_time_ms = int(
                (time.monotonic() - self._recovery_started) * 1000)
            log.info("[runtime] model worker ready after %dms restart #%d "
                     "(rss=%sMB)", self._recovery_time_ms, self._restart_count,
                     self.rss_mb)
        else:
            log.info("[runtime] model worker ready (rss=%sMB)", self.rss_mb)
        self._notify(self._on_ready)

    def _enter_recovering_locked(self, reason: str) -> None:
        if self._state == "recovering":
            self._last_error = reason
            return
        self._state = "recovering"
        self._restart_count += 1
        self._recovery_started = time.monotonic()
        self._recovery_time_ms = None
        self._ready_notified = False
        self._last_error = reason
        log.warning("[runtime] model worker recovering (restart #%d): %s",
                    self._restart_count, reason)
        self._notify(self._on_recovering, reason)

    def _fail_locked(self, reason: str) -> None:
        if self._reaping:
            self._terminate_locked()
            self._state = "stopped"
            return
        self._enter_recovering_locked(reason)
        self._terminate_locked()

    def _idle_expired_locked(self) -> bool:
        return (self._idle_reap_ms > 0 and not self._in_flight
                and self._ready_notified
                and time.monotonic() - self._last_activity > self._idle_reap_ms / 1000.0)

    def _spawn_locked(self) -> None:
        parent, child = self._ctx.Pipe(duplex=True)
        health_parent, health_child = self._ctx.Pipe(duplex=False)
        process = self._ctx.Process(
            target=worker_main,
            args=(child, self._backend, self._config, health_child),
            daemon=True,
            name=f"model_worker_{self._backend}",
        )
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            health_parent.close()
            health_child.close()
            raise
        child.close()
        health_child.close()

        if not parent.poll(30):
            health_parent.close()
            self._kill_process(process, parent)
            raise WorkerError("the model worker did not start within 30s")
        try:
            message = parent.recv()
        except EOFError as error:
            health_parent.close()
            self._kill_process(process, parent)
            raise WorkerError(f"the model worker exited before hello: {error}") from error
        if message[0] != "hello" or not message[1].get("ok"):
            detail = message[1].get("error", "no hello") if message[0] == "hello" else "bad hello"
            health_parent.close()
            self._kill_process(process, parent)
            raise WorkerError(f"the model worker failed to start: {detail}")

        self._process, self._connection = process, parent
        self._health_connection = health_parent
        self._healthy = threading.Event()
        self._capabilities = dict(message[1].get("capabilities") or {})
        # The watcher sends its first state before hello, so a mock worker is
        # healthy by the time `start()` returns and a loading SmolVLA worker is
        # recorded as not healthy immediately.
        self._drain_health_locked(0.5)

    def _drain_health_locked(self, wait_s: float = 0) -> None:
        connection = self._health_connection
        if connection is None:
            return
        try:
            if wait_s and not connection.poll(wait_s):
                return
            while connection.poll(0):
                message = connection.recv()
                if message and message[0] == "health":
                    if message[1]:
                        self._healthy.set()
                    else:
                        self._healthy.clear()
        except (EOFError, OSError):
            return

    def _terminate_locked(self, graceful: bool = True) -> None:
        process, connection = self._process, self._connection
        health, self._health_connection = self._health_connection, None
        self._process = self._connection = None
        self._healthy.clear()
        if health is not None:
            try:
                health.close()
            except Exception:       # noqa: BLE001 — teardown is best effort
                pass
        self._kill_process(process, connection, graceful=graceful)

    @staticmethod
    def _kill_process(process, connection, graceful: bool = True) -> None:
        if process is not None and not graceful and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if connection is not None:
            try:
                connection.close()
            except Exception:       # noqa: BLE001 — teardown is best effort
                pass
        if process is None:
            return
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)

    @staticmethod
    def _notify(callback, *args) -> None:
        if callback is not None:
            try:
                callback(*args)
            except Exception:       # noqa: BLE001 — a callback must not kill the worker
                pass

    # ── requests ─────────────────────────────────────────────────────────────

    def request(self, payload: dict) -> list:
        with self._queue_lock:
            self._outstanding += 1
        try:
            with self._request_lock:
                with self._lock:
                    if self._closed:
                        raise WorkerError("the model worker is closed")
                    if self._state in ("stopped", "dead"):
                        self._reaping = False
                        try:
                            self._spawn_locked()
                        except WorkerError as error:
                            self._state = "dead"
                            self._last_error = str(error)
                            raise
                        self._state = "ready"
                        self._last_activity = time.monotonic()
                        self._ready_notified = self.healthy
                        self._recovery_started = 0.0
                    if self._state == "recovering":
                        raise WorkerRecovering("the model worker is recovering")
                    if not self.alive:
                        self._fail_locked("model worker is not running")
                        raise WorkerRecovering("the model worker is recovering")
                    if not self.healthy:
                        raise WorkerLoading("the model worker is still loading")
                    self._next_request_id += 1
                    request_id = self._next_request_id
                    connection = self._connection
                    self._in_flight = True

                try:
                    try:
                        connection.send(("infer", request_id, payload))
                        if not connection.poll(self._timeout_s):
                            raise WorkerTimeout(
                                f"model worker did not answer within {self._timeout_s:g}s")
                        kind, reply_id, body = connection.recv()
                    except WorkerTimeout as error:
                        with self._lock:
                            self._fail_locked(str(error))
                        raise
                    except (EOFError, OSError) as error:
                        with self._lock:
                            self._fail_locked(f"model worker exited: {error}")
                        raise WorkerError(f"model worker exited: {error}") from error

                    if reply_id != request_id:
                        with self._lock:
                            self._fail_locked(
                                f"worker reply {reply_id} does not match request {request_id}")
                        raise WorkerError("model worker returned an out-of-order reply")
                    if kind != "result":
                        raise BackendError(str(body))
                    return body
                finally:
                    with self._lock:
                        self._in_flight = False
                        self._last_activity = time.monotonic()
        finally:
            with self._queue_lock:
                self._outstanding -= 1

    @property
    def start_method(self) -> str:
        return self._ctx.get_start_method()

    @property
    def alive(self) -> bool:
        # No lock: /healthz must answer while an inference owns the request path.
        process = self._process
        return self._state == "ready" and process is not None and process.is_alive()

    @property
    def healthy(self) -> bool:
        """Process up *and* the model behind it able to infer."""
        event = self._healthy
        return self.alive and event is not None and event.is_set()

    @property
    def capabilities(self) -> dict:
        return dict(self._capabilities)

    @property
    def queue_depth(self) -> int:
        """Requests queued or executing on this worker, for placement."""
        with self._queue_lock:
            return self._outstanding

    @property
    def state(self) -> str:
        """Public worker phase, including the dead-process correction."""
        process = self._process
        state = self._state
        if state == "ready" and (process is None or not process.is_alive()):
            return "recovering"
        return state

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None else None

    @property
    def rss_mb(self) -> float | None:
        """Resident memory of the worker process, or None when unreadable."""
        pid = self.pid
        if pid is None:
            return None
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return round(int(line.split()[1]) / 1024.0, 1)
        except OSError:
            pass
        return None

    def status(self) -> dict:
        process = self._process
        return {
            "state": self.state,
            "healthy": self.healthy,
            "pid": process.pid if process is not None else None,
            "rss_mb": self.rss_mb,
            "restart_count": self._restart_count,
            "recovery_time_ms": self._recovery_time_ms,
            "recovery_time_s": (None if self._recovery_time_ms is None
                                else round(self._recovery_time_ms / 1000.0, 3)),
            "last_error": self._last_error,
            "in_flight": self._in_flight,
            "queue_depth": self.queue_depth,
        }
