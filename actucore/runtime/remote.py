"""The `remote_http` backend: a `motus.vla/1` client for a remote target.

A remote target is another machine's runtime gateway, not a worker in another
process, so this file is the backend that implements that target. It must not
accept a remote whose checkpoint differs from the local worker's, and it must
not report a latency that fits when the driver's observation-age or watchdog
budget has already expired.

The latency sample is the whole request round trip, including the remote
inference. That is the number the driver's `max_obs_age_ms` actually spends,
so it is the conservative one to compare against the budget.
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

# The fields a remote target must agree with the local worker on before it may
# serve a session. Anything else (ready, n_cameras, ...) describes local state
# rather than the checkpoint and is not part of the identity check.
REQUIRED_FIELDS = ("schema", "model", "action_dim", "control_mode",
                   "chunk_size", "control_hz")

# Bounded: /status must stay cheap on a long-running session, and a p95 over
# the last hundred chunks is enough to judge the link.
SAMPLE_WINDOW = 100

BACKEND_NAME = "remote_http"


class RemoteError(RuntimeError):
    """A remote target could not serve the request."""

    def __init__(self, message: str, code: str = "remote_error",
                 status: int = 502):
        super().__init__(message)
        self.code = code
        self.status = status


class RemoteCapabilitiesMismatch(RemoteError):
    def __init__(self, message: str):
        super().__init__(message, "remote_capabilities_mismatch", 502)


class RemoteHTTPBackend:
    """The `remote_http` backend, pointed at one `motus.vla/1` endpoint.

    Unlike `mock`/`delay`/`smolvla` it does not run inside a worker process:
    the worker is on the other side of the endpoint. The gateway selects it as
    the backend of a remote execution target (`runtime.remote.backend`).
    """

    def __init__(self, config: dict,
                 local_capabilities: Callable[[], dict]):
        self._endpoint = str(config.get("endpoint") or "").rstrip("/")
        if not self._endpoint:
            raise ValueError("remote target needs an `endpoint`")
        self._key = str(config.get("api_key") or "")
        self._timeout = float(config.get("timeout_ms", 2000)) / 1000.0
        self._local_capabilities = local_capabilities
        # The budget is a driver property. Without both numbers configured,
        # there is nothing honest to compare the p95 against, so the verdict
        # stays undecided instead of using a guessed default.
        self._budget_configured = all(
            config.get(field) is not None
            for field in ("max_obs_age_ms", "watchdog_ms"))
        self._budget = {
            "max_obs_age_ms": float(config.get("max_obs_age_ms", 300)),
            "watchdog_ms": float(config.get("watchdog_ms", 200)),
        }
        self._lock = threading.Lock()
        self._samples_lock = threading.Lock()
        self._attempts = collections.deque(maxlen=SAMPLE_WINDOW)
        self._capabilities = None
        self._compatible = None
        self._error = ""

    # ── backend protocol ─────────────────────────────────────────────────────

    def capabilities(self, refresh: bool = False) -> dict:
        with self._lock:
            if self._capabilities is not None and not refresh:
                return dict(self._capabilities)
            try:
                capabilities = self._get("/capabilities")
                self._check_capabilities(capabilities)
            except RemoteCapabilitiesMismatch as error:
                self._capabilities = None
                self._compatible = False
                self._error = str(error)
                raise
            except RemoteError as error:
                self._capabilities = None
                self._compatible = None
                self._error = str(error)
                raise
            self._capabilities = dict(capabilities)
            self._compatible = True
            self._error = ""
            return dict(self._capabilities)

    def _check_capabilities(self, remote: dict) -> None:
        """Refuse a target that is not the same model as the local worker.

        The six fields are the model's identity: schema, checkpoint name, the
        action space and its timing. A remote with a different checkpoint is
        not a slower copy of the same policy — it is a different policy, and
        letting it serve the session would move the arm differently.
        """
        local = dict(self._local_capabilities() or {})
        differences = []
        for field in REQUIRED_FIELDS:
            if field not in remote:
                differences.append(
                    f"{field} missing on remote "
                    f"(local {local.get(field)!r})")
                continue
            if field not in local:
                differences.append(
                    f"{field} missing on local "
                    f"(remote {remote.get(field)!r})")
                continue
            remote_value = remote[field]
            local_value = local[field]
            try:
                same = self._same(field, remote_value, local_value)
            except (TypeError, ValueError):
                same = False
            if not same:
                differences.append(
                    f"{field} remote {remote_value!r} "
                    f"!= local {local_value!r}")
        if differences:
            raise RemoteCapabilitiesMismatch(
                f"remote target {self._endpoint} does not match the local "
                f"model: " + "; ".join(differences))

    @staticmethod
    def _same(field: str, remote_value: object,
              local_value: object) -> bool:
        if field in ("action_dim", "chunk_size"):
            return int(remote_value) == int(local_value)
        if field == "control_hz":
            return float(remote_value) == float(local_value)
        return str(remote_value) == str(local_value)

    def infer(self, request: dict, session_id: str) -> list:
        self.capabilities()
        payload = {key: value for key, value in request.items()
                   if key != "target"}
        payload["session_id"] = session_id
        started = time.monotonic()
        try:
            reply = self._post("/infer", payload)
        except RemoteError as error:
            self._record((time.monotonic() - started) * 1000.0, ok=False)
            self._set_error(str(error))
            raise
        self._record((time.monotonic() - started) * 1000.0, ok=True)
        if reply.get("seq") not in (None, payload.get("seq")):
            message = (f"remote reply seq {reply.get('seq')!r} != "
                       f"request {payload.get('seq')!r}")
            self._set_error(message)
            raise RemoteError(message, "out_of_order", 502)
        actions = reply.get("actions")
        if not isinstance(actions, list) or not actions:
            message = "remote target returned no actions"
            self._set_error(message)
            raise RemoteError(message, "empty_chunk", 502)
        self._set_error("")
        return actions

    def close(self) -> None:
        try:
            self._post("/close", {})
        except RemoteError:
            pass

    def close_session(self, session_id: str) -> None:
        self._post("/close", {"session_id": session_id})

    # ── observability ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "backend": BACKEND_NAME,
            "endpoint": self._endpoint,
            "compatible": self._compatible,
            "last_error": self._error,
            "latency": self.latency(),
        }

    def latency(self) -> dict:
        with self._samples_lock:
            attempts = list(self._attempts)
        successes = [elapsed for elapsed, ok in attempts if ok]
        failures = len(attempts) - len(successes)
        result = {
            "samples": len(successes),
            "attempts": len(attempts),
            "failures": failures,
            "last_rtt_ms": (round(successes[-1], 1) if successes else None),
            "p95_rtt_ms": None,
            "budget": dict(self._budget),
            "budget_configured": self._budget_configured,
            "within_max_obs_age": None,
            "within_watchdog": None,
            "control_suitable": None,
            "note": "",
        }
        if not attempts:
            result["note"] = "no remote samples yet"
            return result
        if not successes:
            result["control_suitable"] = False
            result["note"] = (
                f"all {len(attempts)} remote requests failed; use this "
                f"target for offline or non-control validation only")
            return result

        p95 = _p95(successes)
        result["p95_rtt_ms"] = round(p95, 1)
        if not self._budget_configured:
            result["note"] = (
                "driver budget not configured; set "
                "runtime.remote.max_obs_age_ms and watchdog_ms to the "
                "driver descriptor before judging control suitability")
            return result

        within_age = p95 <= self._budget["max_obs_age_ms"]
        within_watchdog = p95 <= self._budget["watchdog_ms"]
        result["within_max_obs_age"] = within_age
        result["within_watchdog"] = within_watchdog
        problems = []
        if failures:
            problems.append(
                f"{failures}/{len(attempts)} remote requests failed")
        if not within_age:
            problems.append(
                f"p95 {p95:.1f}ms exceeds max_obs_age_ms "
                f"{self._budget['max_obs_age_ms']:g}")
        if not within_watchdog:
            problems.append(
                f"p95 {p95:.1f}ms exceeds watchdog_ms "
                f"{self._budget['watchdog_ms']:g}")
        suitable = not problems
        result["control_suitable"] = suitable
        if problems:
            result["note"] = (
                "; ".join(problems)
                + "; use this target for offline or non-control validation "
                  "only")
        return result

    def _record(self, elapsed_ms: float, ok: bool) -> None:
        with self._samples_lock:
            self._attempts.append((float(elapsed_ms), ok))

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    # ── transport ────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(
            self._endpoint + path, headers=self._headers(), method="GET")
        return self._send(request)

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self._endpoint + path, data=body, headers=self._headers(),
            method="POST")
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(
                    request, timeout=self._timeout) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = self._remote_detail(error)
            raise RemoteError(
                f"{request.get_method()} {request.full_url} → "
                f"{error.code}: {detail}", "remote_error", 502) from error
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            raise RemoteError(
                f"{request.full_url}: {error}",
                "remote_unavailable", 503) from error
        except UnicodeDecodeError as error:
            raise RemoteError(
                f"{request.full_url}: response is not UTF-8: {error}",
                "remote_bad_reply", 502) from error
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RemoteError(
                f"{request.full_url}: response is not JSON: {error}",
                "remote_bad_reply", 502) from error

    @staticmethod
    def _remote_detail(error: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode())
            detail = (payload.get("error") or {}).get("message")
            if detail:
                return str(detail)
        except Exception:       # noqa: BLE001 — a body is optional decoration
            pass
        return str(error.reason)


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
