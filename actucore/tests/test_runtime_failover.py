"""Session-scoped mid-session failover, exercised over the gateway HTTP API.

A session is created on the local target, the local worker fails, and the same
session must continue on the remote target from the next chunk. The seam is the
same one the card uses: `/infer` and `/status`. No test reads session internals.

What these tests pin down:

- a crash or a local inference error moves that session to remote on its next
  chunk, and the chunk that was interrupted is never returned;
- the switch happens once at a chunk boundary, never per chunk, and never back;
- local recovery only affects *new* sessions (`failback`);
- the counters and reasons are visible in `/status.placement` and the session
  snapshot.

Real RTX 4060 acceptance stays skipped unless a laptop server and the local
checkpoint are both provided; mock results are never reported as hardware.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_runtime_failover.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.vla_cloud import SCHEMA  # noqa: E402
from runtime import RuntimeGateway  # noqa: E402
from runtime.supervisor import BackendError  # noqa: E402

MODEL = "mock-sine@8s"
CAPABILITIES = {
    "model": MODEL,
    "control_mode": "joint_position",
    "action_dim": 7,
    "chunk_size": 5,
    "control_hz": 30,
    "needs_state": False,
    "n_cameras": 0,
    "image_size": 0,
    "supports_rtc": False,
}
PLACEMENT = {
    "max_queue_depth": 10,
    "max_p95_ms": 10_000,
    "sample_window": 10,
    "ewma_alpha": 1.0,
}


def make_server(**overrides):
    config = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "mock",
        "capabilities": dict(CAPABILITIES),
        "idle_reap_ms": 0,
    }
    config.update(overrides)
    server = RuntimeGateway(config)
    server.start()
    return server


def make_gateway(remote_endpoint=None, **overrides):
    config = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "mock",
        "capabilities": dict(CAPABILITIES),
        "idle_reap_ms": 0,
        "monitor_interval_ms": 50,
        "placement": dict(PLACEMENT),
    }
    if remote_endpoint:
        config["remote"] = {
            "backend": "remote_http",
            "endpoint": remote_endpoint,
            "timeout_ms": 3000,
            "max_obs_age_ms": 300,
            "watchdog_ms": 200,
        }
    config.update(overrides)
    gateway = RuntimeGateway(config)
    gateway.start()
    return gateway


def _request(url, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, method="POST" if body else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def infer_payload(session_id, seq=1, **overrides):
    payload = {"schema": SCHEMA, "session_id": session_id, "seq": seq,
               "t_capture_ms": 1_700_000_000_000, "prompt": "pick it up",
               "images": {}, "inference_delay": 0}
    payload.update(overrides)
    return payload


def status(gateway):
    return _request(gateway.url + "/status")[1]


def session_snapshot(gateway, session_id):
    matches = [session for session in status(gateway)["sessions"]
               if session["session_id"] == session_id]
    assert matches, f"no session {session_id!r}"
    return matches[-1]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def kill_local(gateway):
    pid = gateway.worker.pid
    assert pid is not None
    os.kill(pid, signal.SIGKILL)
    assert wait_for(lambda: status(gateway)["worker"]["state"] == "recovering")


# ── failover on a local health failure ───────────────────────────────────────

def test_a_crashed_local_session_fails_over_and_keeps_serving():
    server = make_server()
    # A slow monitor keeps the worker in `recovering` long enough to observe
    # the failover decision deterministically; the crash itself is real.
    gateway = make_gateway(server.url, monitor_interval_ms=5000)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200
        assert session_snapshot(gateway, "s1")["target"] == "local"

        kill_local(gateway)

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=2))
        assert code == 200
        assert len(body["actions"]) == CAPABILITIES["chunk_size"]

        session = session_snapshot(gateway, "s1")
        assert session["target"] == "remote"
        assert session["failover_count"] == 1
        assert session["switch_count"] == 1
        assert session["failover_reason"] == "local_unhealthy"

        placement = status(gateway)["placement"]
        assert placement["failover_count"] == 1
        assert placement["switch_count"] == 1
        assert placement["switches"][-1] == {
            "session_id": "s1",
            "kind": "failover",
            "placement_decision": "remote",
            "from": "local",
            "to": "remote",
            "reason": "local_unhealthy",
        }
        assert status(gateway)["remote"]["latency"]["attempts"] == 1

        # The next chunk continues on the same remote session.
        assert _request(gateway.url + "/infer",
                        infer_payload("s1", seq=3))[0] == 200
        assert session_snapshot(gateway, "s1")["failover_count"] == 1
        assert status(gateway)["remote"]["latency"]["attempts"] == 2
    finally:
        gateway.stop()
        server.stop()


def test_a_local_inference_error_fails_over_on_the_next_chunk(monkeypatch):
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200

        calls = {"count": 0}
        original = gateway._worker.request

        def failing_request(payload):
            calls["count"] += 1
            if calls["count"] == 1:
                raise BackendError("model failed on this chunk")
            return original(payload)

        monkeypatch.setattr(gateway._worker, "request", failing_request)

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=2))
        assert code == 502
        assert body["error"]["code"] == "model_error"
        assert "actions" not in body

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=3))
        assert code == 200 and body["actions"]
        session = session_snapshot(gateway, "s1")
        assert session["target"] == "remote"
        assert session["failover_reason"] == "local_inference_error"
        assert status(gateway)["placement"]["failover_count"] == 1

        # The worker stayed healthy, so local never degraded: a new session is
        # ordinary local-first, not a failback.
        assert _request(gateway.url + "/infer",
                        infer_payload("s2"))[0] == 200
        placement = status(gateway)["placement"]
        assert placement["failback_count"] == 0
        assert placement["switch_count"] == 1
        assert [d for d in placement["decisions"]
                if d["session_id"] == "s2"][-1]["reason"] == "local_first"
    finally:
        gateway.stop()
        server.stop()


def test_an_empty_local_chunk_fails_over_on_the_next_chunk(monkeypatch):
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200

        calls = {"count": 0}
        original = gateway._worker.request

        def empty_request(payload):
            calls["count"] += 1
            if calls["count"] == 1:
                return []
            return original(payload)

        monkeypatch.setattr(gateway._worker, "request", empty_request)

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=2))
        assert code == 502
        assert body["error"]["code"] == "empty_chunk"

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=3))
        assert code == 200 and body["actions"]
        session = session_snapshot(gateway, "s1")
        assert session["target"] == "remote"
        assert session["failover_reason"] == "local_empty_chunk"
    finally:
        gateway.stop()
        server.stop()


def test_a_timed_out_local_chunk_fails_over_on_the_next_chunk():
    server = make_server()
    gateway = make_gateway(server.url, backend="delay", delay_ms=5,
                           infer_timeout_ms=100)
    try:
        code, body = _request(
            gateway.url + "/infer",
            infer_payload("s1", delay_ms=500))
        assert code == 504
        assert body["error"]["code"] == "timeout"
        assert "actions" not in body

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=2))
        assert code == 200 and body["actions"]
        session = session_snapshot(gateway, "s1")
        assert session["target"] == "remote"
        assert session["failover_reason"] == "local_timeout"
        assert status(gateway)["placement"]["failover_count"] == 1
    finally:
        gateway.stop()
        server.stop()


def test_an_in_flight_local_chunk_is_discarded_not_mixed_with_remote():
    server = make_server()
    gateway = make_gateway(server.url, backend="delay", delay_ms=500)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200

        reply = {}

        def run_infer():
            reply["value"] = _request(gateway.url + "/infer",
                                      infer_payload("s1", seq=2))

        thread = threading.Thread(target=run_infer)
        thread.start()
        assert wait_for(lambda: session_snapshot(gateway, "s1")["in_flight"])
        old_pid = gateway.worker.pid

        kill_local(gateway)
        thread.join(timeout=10)

        code, body = reply["value"]
        assert code in (503, 504)
        assert "actions" not in body, "the interrupted local chunk leaked"
        # The failed chunk must not have been silently retried on remote.
        assert status(gateway)["remote"]["latency"]["attempts"] == 0
        # The old worker process is replaced; its pipe and GPU context are gone.
        assert wait_for(lambda: gateway.worker.pid not in (None, old_pid),
                        timeout=10)

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=3))
        assert code == 200 and body["actions"]
        assert session_snapshot(gateway, "s1")["target"] == "remote"
        assert status(gateway)["remote"]["latency"]["attempts"] == 1
    finally:
        gateway.stop()
        server.stop()


def test_a_crash_does_not_fail_over_when_no_remote_is_configured():
    gateway = make_gateway()
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200
        kill_local(gateway)

        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=2))
        assert code == 503
        assert body["error"]["code"] == "recovering"
        session = session_snapshot(gateway, "s1")
        assert session["target"] == "local"
        assert session["failover_count"] == 0
        assert status(gateway)["placement"]["failover_count"] == 0
    finally:
        gateway.stop()


def test_a_crash_does_not_fail_over_to_an_incompatible_remote():
    server = make_server(capabilities={**CAPABILITIES, "model": "other",
                                       "action_dim": 6})
    gateway = make_gateway(server.url, monitor_interval_ms=5000)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200
        kill_local(gateway)

        code, _ = _request(gateway.url + "/infer",
                           infer_payload("s1", seq=2))
        assert code == 503
        session = session_snapshot(gateway, "s1")
        assert session["target"] == "local"
        assert session["failover_count"] == 0
    finally:
        gateway.stop()
        server.stop()


def test_explicit_local_target_still_fails_over_after_a_crash():
    server = make_server()
    gateway = make_gateway(server.url, monitor_interval_ms=5000)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1", target="local"))[0] == 200
        kill_local(gateway)

        # The explicit local pin is session placement, not a per-chunk promise:
        # a local health failure still moves this session.
        assert _request(gateway.url + "/infer",
                        infer_payload("s1", seq=2, target="local"))[0] == 200
        assert session_snapshot(gateway, "s1")["target"] == "remote"

        # An explicit request for the old target is a conflict, not a switch.
        code, body = _request(gateway.url + "/infer",
                              infer_payload("s1", seq=3, target="local"))
        assert code == 409
        assert body["error"]["code"] == "target_conflict"
    finally:
        gateway.stop()
        server.stop()


# ── one switch, no failback of the existing session ─────────────────────────

def test_a_hard_cancel_is_not_a_health_failure_and_does_not_fail_over():
    server = make_server()
    gateway = make_gateway(server.url, backend="delay", delay_ms=5)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("idle"))[0] == 200
        assert _request(gateway.url + "/infer",
                        infer_payload("stuck"))[0] == 200

        reply = {}

        def run_stuck():
            reply["value"] = _request(
                gateway.url + "/infer",
                infer_payload("stuck", seq=2, delay_ms=500))

        thread = threading.Thread(target=run_stuck)
        thread.start()
        assert wait_for(lambda: session_snapshot(gateway, "stuck")["in_flight"])

        code, body = _request(gateway.url + "/cancel",
                              {"session_id": "stuck", "hard": True})
        assert code == 202
        assert body["state"] == "recovering"
        thread.join(timeout=10)
        assert reply["value"][0] in (409, 503)

        assert wait_for(lambda: status(gateway)["worker"]["state"] == "ready"
                        and status(gateway)["worker"]["restart_count"] == 1,
                        timeout=10)

        # The other local session stays local: an operator's hard cancel is
        # not an involuntary local health failure.
        assert _request(gateway.url + "/infer",
                        infer_payload("idle", seq=2))[0] == 200
        assert session_snapshot(gateway, "idle")["target"] == "local"
        assert status(gateway)["placement"]["failover_count"] == 0
    finally:
        gateway.stop()
        server.stop()


def test_a_failed_over_session_stays_remote_while_new_sessions_return_local():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200
        kill_local(gateway)
        assert _request(gateway.url + "/infer",
                        infer_payload("s1", seq=2))[0] == 200
        assert session_snapshot(gateway, "s1")["target"] == "remote"

        # Let the local worker recover and stabilise.
        assert wait_for(lambda: status(gateway)["worker"]["state"] == "ready"
                        and status(gateway)["worker"]["restart_count"] == 1,
                        timeout=10)

        # A new session is a failback: local again, counted once.
        assert _request(gateway.url + "/infer",
                        infer_payload("s2"))[0] == 200
        decision = [d for d in status(gateway)["placement"]["decisions"]
                    if d["session_id"] == "s2"][-1]
        assert decision["target"] == "local"
        assert decision["reason"] == "local_recovered"

        placement = status(gateway)["placement"]
        assert placement["failover_count"] == 1
        assert placement["failback_count"] == 1
        assert placement["switch_count"] == 2

        # The failed-over session never comes back, even though local is
        # healthy again.
        for seq in (3, 4):
            assert _request(gateway.url + "/infer",
                            infer_payload("s1", seq=seq))[0] == 200
        assert session_snapshot(gateway, "s1")["target"] == "remote"
        assert status(gateway)["placement"]["failover_count"] == 1

        # A second local crash must not cause a second switch for that session.
        kill_local(gateway)
        assert wait_for(lambda: status(gateway)["worker"]["state"] == "recovering")
        assert _request(gateway.url + "/infer",
                        infer_payload("s2", seq=2))[0] == 200
        assert session_snapshot(gateway, "s2")["target"] == "remote"
        assert session_snapshot(gateway, "s1")["failover_count"] == 1
        assert status(gateway)["placement"]["failover_count"] == 2
    finally:
        gateway.stop()
        server.stop()


# ── real hardware acceptance (skipped by default) ──────────────────────────

REAL_ENDPOINT = os.environ.get("ACTUCORE_REMOTE_VLA_ENDPOINT", "")
REAL_CHECKPOINT = os.environ.get("ACTUCORE_SMOLVLA_CHECKPOINT", "")


@pytest.mark.skipif(not (REAL_ENDPOINT and REAL_CHECKPOINT),
                    reason=("hardware acceptance: set "
                            "ACTUCORE_REMOTE_VLA_ENDPOINT to a laptop "
                            "RTX 4060 motus.vla/1 server and "
                            "ACTUCORE_SMOLVLA_CHECKPOINT to the same "
                            "checkpoint here"))
def test_real_rtx4060_mid_session_failover():
    """Real RTX 4060 acceptance. The loopback tests above are not this test."""
    gateway = RuntimeGateway({
        "enabled": True, "host": "127.0.0.1", "port": 0, "backend": "smolvla",
        "smolvla": {"model_dir": REAL_CHECKPOINT,
                    "device": os.environ.get("ACTUCORE_SMOLVLA_DEVICE", "cuda")},
        "idle_reap_ms": 0,
        "monitor_interval_ms": 50,
        "remote": {"endpoint": REAL_ENDPOINT, "timeout_ms": 5000,
                   "max_obs_age_ms": 300, "watchdog_ms": 200},
    })
    gateway.start()
    try:
        assert wait_for(lambda: status(gateway)["worker"]["healthy"], timeout=120)
        assert _request(gateway.url + "/infer",
                        infer_payload("rtx"))[0] == 200
        os.kill(gateway.worker.pid, signal.SIGKILL)
        assert wait_for(lambda: status(gateway)["worker"]["state"] == "recovering")
        assert _request(gateway.url + "/infer",
                        infer_payload("rtx", seq=2))[0] == 200
        assert session_snapshot(gateway, "rtx")["target"] == "remote"
        assert status(gateway)["placement"]["failover_count"] == 1
    finally:
        gateway.stop()
