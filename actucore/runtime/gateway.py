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
from .session import BusyError, ClosedError, SessionRegistry
from .supervisor import BackendError, WorkerError, WorkerSupervisor, WorkerTimeout

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
        self._server = None
        self._thread = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._worker.start()
        try:
            self._server = _ThreadingHTTPServer(
                (runtime_config.host(self._config), runtime_config.port(self._config)),
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
        self._worker.stop()

    @property
    def url(self) -> str:
        if self._server is None:
            return runtime_config.base_url(self._config)
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def worker(self) -> WorkerSupervisor:
        return self._worker

    # ── HTTP surface ─────────────────────────────────────────────────────────

    def capabilities(self) -> dict:
        return dict(self._capabilities)

    def health(self) -> tuple[int, dict]:
        if self._worker.alive:
            return 200, {"schema": runtime_config.SCHEMA,
                         "status": "ok", "worker": "ready"}
        return 503, {"schema": runtime_config.SCHEMA,
                     "status": "unavailable", "worker": self._worker.status()["state"]}

    def status(self) -> dict:
        return {"schema": runtime_config.SCHEMA,
                "worker": self._worker.status(),
                "sessions": self._sessions.snapshot()}

    def infer(self, body: dict) -> tuple[int, dict]:
        if body.get("schema") != runtime_config.SCHEMA:
            return 400, _error("schema_mismatch",
                               f"this gateway speaks {runtime_config.SCHEMA!r}, "
                               f"request says {body.get('schema')!r}")
        session_id = str(body.get("session_id") or "")
        if not session_id:
            return 400, _error("missing_session", "infer needs a session_id")

        session = self._sessions.get_or_create(session_id)
        try:
            session.begin_infer()
        except BusyError as error:
            return 409, _error("busy", str(error), session_id=session_id)
        except ClosedError as error:
            return 409, _error("closed", str(error), session_id=session_id)

        try:
            actions = self._worker.request(body)
        except WorkerTimeout as error:
            if session.closed:
                session.finish_infer(str(error))
                return 409, _error("cancelled", "the session was closed", session_id=session_id)
            session.mark_recovering(str(error))
            return 504, _error("timeout", str(error), session_id=session_id)
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
            return 502, _error("empty_chunk", "the model worker returned no actions",
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
            result = {"schema": runtime_config.SCHEMA,
                      "closed": session_id,
                      "state": session.snapshot()["state"]}
        else:
            closed = self._sessions.close_all()
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
