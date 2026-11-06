"""The runtime gateway: `motus.vla/1` over loopback HTTP.

The card keeps using its existing `vla_cloud` provider; this process is simply
the endpoint it points at. Sessions, cancellation and worker ownership live
behind that unchanged wire protocol.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlsplit

from . import config as runtime_config
from .remote import RemoteError, RemoteHTTPBackend
from .session import (BusyError, ClosedError, LOCAL_TARGET, REMOTE_TARGET,
                      SessionRegistry, TARGETS)
from .supervisor import (BackendError, WorkerError, WorkerLoading,
                         WorkerSupervisor, WorkerTimeout)

log = logging.getLogger(__name__)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _error(code: str, message: str, **extra) -> dict:
    return {"schema": runtime_config.SCHEMA,
            "error": {"code": code, "message": message, **extra}}


class RuntimeGateway:
    def __init__(self, config: dict):
        self._config = dict(config or {})
        self._capabilities = runtime_config.capabilities(self._config)
        self._sessions = SessionRegistry()
        worker_config = dict(self._config)
        worker_config["capabilities"] = self._capabilities
        self._worker = WorkerSupervisor(
            backend=str(self._config.get("backend") or "mock"),
            config=worker_config,
            timeout_ms=float(self._config.get("infer_timeout_ms", 5000)),
            idle_reap_ms=float(self._config.get(
                "idle_reap_ms", self._config.get("idle_timeout_ms", 60000))),
            monitor_interval_ms=float(self._config.get("monitor_interval_ms", 100)),
            on_recovering=self._sessions.mark_all_recovering,
            on_ready=self._sessions.mark_all_ready,
        )
        remote_config = runtime_config.remote(self._config)
        self._remote = (RemoteHTTPBackend(remote_config, self.capabilities)
                        if remote_config else None)
        self._server = None
        self._thread = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._worker.start()
        try:
            self._server = _ThreadingHTTPServer(
                (runtime_config.bind_host(self._config),
                 runtime_config.port(self._config)),
                _make_handler(self),
            )
        except Exception:
            self._worker.stop()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="runtime_gateway", daemon=True)
        self._thread.start()
        log.info("[runtime] gateway listening on %s (worker pid=%s)",
                 self.url, self._worker.pid)

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._sessions.close_all()
        if self._remote is not None:
            self._remote.close()
        self._worker.stop()

    @property
    def url(self) -> str:
        if self._server is None:
            return runtime_config.base_url(self._config)
        host, port = self._server.server_address[:2]
        # A server bound to every interface is reached through loopback
        # locally; printing 0.0.0.0 as a client URL would be a dead address.
        if host in ("", "0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    @property
    def worker(self) -> WorkerSupervisor:
        return self._worker

    # ── HTTP surface ─────────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        """What the worker actually loaded, not what the config guessed.

        A real checkpoint owns `action_dim`, `chunk_size`, `image_size` and the
        rest; copying them into runtime config would be a second source of
        truth that drifts from the weights. The worker answers hello from the
        checkpoint's own config, before its weights load, so this is available
        as soon as the gateway starts.
        """
        capabilities = dict(self._capabilities)
        capabilities.update(self._worker.capabilities)
        capabilities["schema"] = runtime_config.SCHEMA
        if "ready" in capabilities:
            capabilities["ready"] = self._worker.healthy
        return capabilities

    def health(self) -> tuple[int, dict]:
        state = self._worker.state
        if self._worker.healthy:
            return 200, {"schema": runtime_config.SCHEMA,
                         "status": "ok", "worker": "ready"}
        if state == "ready":
            # The process is up but the checkpoint's weights are still loading.
            # This is the state the card must show as `loading`, not `running`.
            return 503, {"schema": runtime_config.SCHEMA,
                         "status": "loading", "worker": "ready", "healthy": False}
        return 503, {"schema": runtime_config.SCHEMA,
                     "status": "unavailable", "worker": state, "healthy": False}

    def status(self) -> dict:
        result = {"schema": runtime_config.SCHEMA,
                  "worker": self._worker.status(),
                  "sessions": self._sessions.snapshot()}
        if self._remote is not None:
            result["remote"] = self._remote.status()
        return result

    def infer(self, body: dict) -> tuple[int, dict]:
        if body.get("schema") != runtime_config.SCHEMA:
            return 400, _error("schema_mismatch",
                               f"this gateway speaks {runtime_config.SCHEMA!r}, "
                               f"request says {body.get('schema')!r}")
        session_id = str(body.get("session_id") or "")
        if not session_id:
            return 400, _error("missing_session", "infer needs a session_id")

        requested = body.get("target")
        if requested is not None and requested not in TARGETS:
            return 400, _error(
                "bad_target",
                f"target must be one of {TARGETS}, not {requested!r}",
                session_id=session_id)

        session = self._sessions.get(session_id)
        created = session is None
        if created:
            target = str(requested or LOCAL_TARGET)
            if target == REMOTE_TARGET and self._remote is None:
                return 400, _error(
                    "remote_not_configured",
                    "this gateway has no `runtime.remote.endpoint`; a session "
                    "cannot be placed on a remote target",
                    session_id=session_id)
            session = self._sessions.get_or_create(session_id, target)
        if requested and requested != session.target:
            # A session is pinned to one machine for its whole life. Mixing
            # chunks from two targets is not a retry, it is a different policy.
            return 409, _error(
                "target_conflict",
                f"session {session_id!r} is pinned to {session.target!r}, "
                f"request asks for {requested!r}", session_id=session_id)

        if session.target == LOCAL_TARGET:
            worker_state = self._worker.state
            if worker_state == "recovering":
                return 503, _error(
                    "recovering", "the model worker is recovering",
                    session_id=session_id)
            if worker_state == "ready" and not self._worker.healthy:
                return 503, _error(
                    "loading", "model weights are still loading",
                    session_id=session_id)

        if created and session.target == REMOTE_TARGET:
            # A new session revalidates the remote identity: the laptop server
            # may have restarted with different weights since the last one.
            try:
                self._remote.capabilities(refresh=True)
            except RemoteError as error:
                return error.status, _error(
                    error.code, str(error), session_id=session_id)

        try:
            session.begin_infer()
        except BusyError as error:
            return 409, _error("busy", str(error), session_id=session_id)
        except ClosedError as error:
            return 409, _error("closed", str(error), session_id=session_id)

        try:
            if session.target == REMOTE_TARGET:
                actions = self._remote.infer(body, session_id)
            else:
                actions = self._worker.request(body)
        except RemoteError as error:
            session.finish_infer(str(error))
            if session.closed:
                return 409, _error("cancelled", "the session was closed",
                                   session_id=session_id)
            return error.status, _error(
                error.code, str(error), session_id=session_id)
        except WorkerTimeout as error:
            if session.closed:
                session.finish_infer(str(error))
                return 409, _error("cancelled", "the session was closed", session_id=session_id)
            session.mark_recovering(str(error))
            return 504, _error("timeout", str(error), session_id=session_id)
        except WorkerLoading as error:
            session.finish_infer(str(error))
            return 503, _error("loading", str(error), session_id=session_id)
        except BackendError as error:
            session.finish_infer(str(error))
            return 502, _error("model_error", str(error), session_id=session_id)
        except WorkerError as error:
            if session.closed:
                session.finish_infer(str(error))
                return 409, _error("cancelled", "the session was closed", session_id=session_id)
            session.mark_recovering(str(error))
            return 503, _error("recovering", str(error), session_id=session_id)

        if session.cancelled:
            session.finish_infer()
            return 409, _error("cancelled", "the session was cancelled", session_id=session_id)
        if not isinstance(actions, list) or not actions:
            session.finish_infer("empty action chunk")
            return 502, _error(
                "empty_chunk",
                "the execution target returned no actions",
                session_id=session_id)

        session.finish_infer()
        return 200, {"schema": runtime_config.SCHEMA,
                     "seq": body.get("seq"), "actions": actions}

    def close(self, body: dict) -> tuple[int, dict]:
        session_id = str(body.get("session_id") or "")
        if session_id:
            session = self._sessions.close(session_id)
            if session is None:
                return 404, _error("not_found", f"no session {session_id!r}",
                                   session_id=session_id)
            if session.target == REMOTE_TARGET and self._remote is not None:
                try:
                    self._remote.close_session(session_id)
                except RemoteError:
                    # The local session is already closed; a remote server that
                    # is gone has nothing left to clean up that matters.
                    pass
            result = {"schema": runtime_config.SCHEMA,
                      "closed": session_id,
                      "state": session.snapshot()["state"]}
        else:
            closed = self._sessions.close_all()
            if self._remote is not None:
                self._remote.close()
            result = {"schema": runtime_config.SCHEMA, "closed": closed}
        self._worker.reap()
        return 200, result

    def cancel(self, body: dict) -> tuple[int, dict]:
        session_id = str(body.get("session_id") or "")
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            return 404, _error("not_found", f"no session {session_id!r}",
                               session_id=session_id)
        session.cancel()
        if bool(body.get("hard")) and session.in_flight:
            if session.target == REMOTE_TARGET:
                # There is no local process to kill, and a remote forward
                # cannot be preempted. The reply is discarded when it arrives,
                # which is the same contract as a soft cancel.
                return 202, {"schema": runtime_config.SCHEMA,
                             "cancelled": session_id, "state": "cancelling"}
            self._worker.kill("hard cancel requested")
            session.mark_recovering("hard cancel requested")
            return 202, {"schema": runtime_config.SCHEMA,
                         "cancelled": session_id, "state": "recovering"}
        return 200, {"schema": runtime_config.SCHEMA,
                     "cancelled": session_id,
                     "state": session.snapshot()["state"]}


def _make_handler(gateway: RuntimeGateway) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):      # noqa: A002
            log.debug("[runtime] %s - %s", self.address_string(), format % args)

        def do_GET(self):                          # noqa: N802
            path = urlsplit(self.path).path
            if path == "/capabilities":
                self._json(200, gateway.capabilities())
            elif path == "/healthz":
                self._json(*gateway.health())
            elif path == "/status":
                self._json(200, gateway.status())
            else:
                self._json(404, _error("not_found", f"no route {path!r}"))

        def do_POST(self):                         # noqa: N802
            path = urlsplit(self.path).path
            try:
                body = self._body()
            except ValueError as error:
                self._json(400, _error("bad_request", str(error)))
                return
            if path == "/infer":
                self._json(*gateway.infer(body))
            elif path == "/cancel":
                self._json(*gateway.cancel(body))
            elif path == "/close":
                self._json(*gateway.close(body))
            else:
                self._json(404, _error("not_found", f"no route {path!r}"))

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                value = json.loads(self.rfile.read(length).decode())
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"body is not JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError("body must be a JSON object")
            return value

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
