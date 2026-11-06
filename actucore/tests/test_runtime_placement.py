"""Local-first placement of new sessions, exercised over the gateway HTTP API.

Placement is a property of session creation, not of a chunk: the tests here
start gateways the way the card does and read `/status`, never the policy
object. A synthetic-load CLI at `tools/runtime_load.py` covers the same
decisions end to end and is asserted separately.

Real RTX 4060 acceptance stays skipped unless a laptop server and the local
checkpoint are both provided; mock decisions are never reported as hardware
measurements.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_runtime_placement.py -q
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
    "max_queue_depth": 0,
    "max_p95_ms": 30,
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


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ── default local and decision records ──────────────────────────────────────

def test_new_sessions_are_local_by_default_and_record_a_placement_decision():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        for index in range(2):
            assert _request(gateway.url + "/infer",
                            infer_payload(f"s{index}"))[0] == 200

        body = status(gateway)
        sessions = body["sessions"]
        assert [session["target"] for session in sessions] == ["local", "local"]
        assert [session["placement_decision"] for session in sessions] == \
            ["local", "local"]
        assert [session["placement_reason"] for session in sessions] == \
            ["local_first", "local_first"]

        placement = body["placement"]
        assert placement["policy"] == "local_first"
        assert placement["thresholds"]["max_queue_depth"] == 0
        assert placement["thresholds"]["max_p95_ms"] == 30
        assert len(placement["decisions"]) == 2
        assert placement["counts"] == {"local": 2, "remote": 0}
        decision = placement["decisions"][0]
        assert decision["session_id"] == "s0"
        assert decision["placement_decision"] == "local"
        assert decision["target"] == "local"
        assert decision["reason"] == "local_first"
        assert decision["queue_depth"] == 0
        assert "p95_ms" in decision
    finally:
        gateway.stop()
        server.stop()


def _decision_for(gateway, session_id):
    decisions = status(gateway)["placement"]["decisions"]
    matches = [decision for decision in decisions
               if decision["session_id"] == session_id]
    assert matches, f"no placement decision for {session_id!r}"
    return matches[-1]


# ── overflow conditions ─────────────────────────────────────────────────────

def test_queue_depth_over_threshold_places_a_new_session_on_remote():
    server = make_server()
    gateway = make_gateway(server.url, backend="delay", delay_ms=300)
    try:
        blocker = {}

        def run_blocker():
            blocker["reply"] = _request(
                gateway.url + "/infer",
                infer_payload("blocker", target="local", delay_ms=800))

        thread = threading.Thread(target=run_blocker)
        thread.start()
        assert wait_for(lambda: status(gateway)["worker"]["queue_depth"] >= 1)

        code, body = _request(gateway.url + "/infer", infer_payload("overflow"))
        assert code == 200 and body["actions"]

        decision = _decision_for(gateway, "overflow")
        assert decision["placement_decision"] == "remote"
        assert decision["target"] == "remote"
        assert decision["reason"] == "queue_depth"
        assert decision["queue_depth"] >= 1

        thread.join(timeout=10)
        assert blocker["reply"][0] == 200
        sessions = {session["session_id"]: session
                    for session in status(gateway)["sessions"]}
        assert sessions["blocker"]["target"] == "local"
    finally:
        gateway.stop()
        server.stop()


def test_p95_over_threshold_places_a_new_session_on_remote():
    server = make_server()
    gateway = make_gateway(
        server.url, backend="delay", delay_ms=60,
        placement={**PLACEMENT, "max_p95_ms": 20, "sample_window": 5})
    try:
        # One session is pinned local before the latency rises; it must stay
        # local for its next chunk even while new sessions overflow.
        assert _request(gateway.url + "/infer",
                        infer_payload("sticky", target="local"))[0] == 200
        for index in range(3):
            assert _request(gateway.url + "/infer",
                            infer_payload(f"slow-{index}", target="local"))[0] == 200

        assert _request(gateway.url + "/infer",
                        infer_payload("overflow"))[0] == 200
        decision = _decision_for(gateway, "overflow")
        assert decision["placement_decision"] == "remote"
        assert decision["reason"] == "p95"
        assert decision["p95_ms"] > 20

        assert _request(gateway.url + "/infer",
                        infer_payload("sticky", seq=2))[0] == 200
        sessions = {session["session_id"]: session
                    for session in status(gateway)["sessions"]}
        assert sessions["sticky"]["target"] == "local"
        assert len([d for d in status(gateway)["placement"]["decisions"]
                    if d["session_id"] == "sticky"]) == 1
    finally:
        gateway.stop()
        server.stop()


def test_unhealthy_local_routes_new_sessions_remote_then_local_recovers():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        pid = gateway.worker.pid
        os.kill(pid, signal.SIGKILL)
        assert wait_for(
            lambda: status(gateway)["worker"]["state"] == "recovering")

        assert _request(gateway.url + "/infer",
                        infer_payload("during-recovery"))[0] == 200
        decision = _decision_for(gateway, "during-recovery")
        assert decision["placement_decision"] == "remote"
        assert decision["reason"] == "local_unhealthy"

        assert wait_for(lambda: status(gateway)["worker"]["state"] == "ready"
                        and status(gateway)["worker"]["restart_count"] == 1)

        assert _request(gateway.url + "/infer",
                        infer_payload("after-recovery"))[0] == 200
        decision = _decision_for(gateway, "after-recovery")
        assert decision["placement_decision"] == "local"
        assert decision["reason"] == "local_recovered"

        # The remote session stays remote: recovery is for new sessions, not
        # an excuse to move an existing one mid-life.
        assert _request(gateway.url + "/infer",
                        infer_payload("during-recovery", seq=2))[0] == 200
        sessions = {session["session_id"]: session
                    for session in status(gateway)["sessions"]}
        assert sessions["during-recovery"]["target"] == "remote"
    finally:
        gateway.stop()
        server.stop()


def test_explicit_remote_target_is_recorded_as_explicit():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("manual", target="remote"))[0] == 200
        decision = _decision_for(gateway, "manual")
        assert decision["placement_decision"] == "remote"
        assert decision["reason"] == "explicit"
    finally:
        gateway.stop()
        server.stop()


def test_remote_with_different_capabilities_is_not_used_for_overflow():
    server = make_server(capabilities={
        **CAPABILITIES, "model": "smolvla_other", "action_dim": 6})
    gateway = make_gateway(
        server.url, backend="delay", delay_ms=60,
        placement={**PLACEMENT, "max_p95_ms": 20, "sample_window": 5})
    try:
        for index in range(3):
            assert _request(gateway.url + "/infer",
                            infer_payload(f"slow-{index}", target="local"))[0] == 200

        assert _request(gateway.url + "/infer",
                        infer_payload("stays-local"))[0] == 200
        decision = _decision_for(gateway, "stays-local")
        assert decision["target"] == "local"
        assert decision["reason"] == "remote_unavailable"
        assert decision["remote_healthy"] is False

        remote = status(gateway)["remote"]
        assert remote["healthy"] is False
        assert remote["compatible"] is False
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
def test_real_rtx4060_placement_overflows_to_remote():
    """Real RTX 4060 acceptance. The loopback tests above are not this test."""
    gateway = RuntimeGateway({
        "enabled": True, "host": "127.0.0.1", "port": 0, "backend": "smolvla",
        "smolvla": {"model_dir": REAL_CHECKPOINT,
                    "device": os.environ.get("ACTUCORE_SMOLVLA_DEVICE", "cuda")},
        "idle_reap_ms": 0,
        "monitor_interval_ms": 50,
        # Any measured local latency triggers overflow, so the second session
        # exercises placement rather than explicit target routing.
        "placement": {"enabled": True, "max_queue_depth": 100,
                      "max_p95_ms": 0.0, "sample_window": 5,
                      "ewma_alpha": 1.0, "p95_ttl_ms": 0,
                      "decision_log_size": 100},
        "remote": {"endpoint": REAL_ENDPOINT, "timeout_ms": 5000,
                   "max_obs_age_ms": 300, "watchdog_ms": 200},
    })
    gateway.start()
    try:
        assert wait_for(lambda: status(gateway)["worker"]["healthy"], timeout=120)
        assert _request(gateway.url + "/infer",
                        infer_payload("rtx-local"))[0] == 200
        assert _decision_for(gateway, "rtx-local")["target"] == "local"

        assert _request(gateway.url + "/infer",
                        infer_payload("rtx-overflow"))[0] == 200
        decision = _decision_for(gateway, "rtx-overflow")
        assert decision["target"] == "remote"
        assert decision["reason"] == "p95"

        latency = status(gateway)["remote"]["latency"]
        assert latency["samples"] == 1
        assert latency["last_rtt_ms"] > 0
        assert status(gateway)["remote"]["healthy"] is True
    finally:
        gateway.stop()


def test_an_idle_reaped_worker_is_woken_instead_of_overflowing():
    server = make_server()
    gateway = make_gateway(server.url, idle_reap_ms=100)
    try:
        assert wait_for(lambda: status(gateway)["worker"]["state"] == "stopped")
        assert _request(gateway.url + "/infer",
                        infer_payload("after-idle"))[0] == 200
        decision = _decision_for(gateway, "after-idle")
        assert decision["target"] == "local"
        assert decision["reason"] == "local_first"
    finally:
        gateway.stop()
        server.stop()


def test_placement_can_be_disabled_explicitly():
    server = make_server()
    gateway = make_gateway(
        server.url, placement={**PLACEMENT, "enabled": False})
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("s1"))[0] == 200
        decision = _decision_for(gateway, "s1")
        assert decision["target"] == "local"
        assert decision["reason"] == "placement_disabled"
    finally:
        gateway.stop()
        server.stop()


def test_a_remote_with_a_dead_worker_is_not_overflow_target():
    server = make_server(monitor_interval_ms=5000)
    gateway = make_gateway(
        server.url, backend="delay", delay_ms=60,
        placement={**PLACEMENT, "max_p95_ms": 20, "sample_window": 5})
    try:
        os.kill(server.worker.pid, signal.SIGKILL)
        assert wait_for(
            lambda: _request(server.url + "/status")[1]["worker"]["state"]
            == "recovering")

        for index in range(3):
            assert _request(gateway.url + "/infer",
                            infer_payload(f"slow-{index}", target="local"))[0] == 200

        assert _request(gateway.url + "/infer",
                        infer_payload("stays-local"))[0] == 200
        decision = _decision_for(gateway, "stays-local")
        assert decision["target"] == "local"
        assert decision["reason"] == "remote_unavailable"
        assert decision["remote_healthy"] is False
    finally:
        gateway.stop()
        server.stop()


def test_a_remote_over_the_driver_budget_is_not_overflow_target():
    server = make_server(backend="delay", delay_ms=60)
    gateway = make_gateway(
        server.url, backend="delay", delay_ms=60,
        placement={**PLACEMENT, "max_p95_ms": 20, "sample_window": 5},
        remote={"endpoint": server.url, "timeout_ms": 3000,
                "max_obs_age_ms": 20, "watchdog_ms": 30})
    try:
        for seq in range(1, 4):
            assert _request(gateway.url + "/infer",
                            infer_payload("warm-remote", seq=seq,
                                          target="remote"))[0] == 200
        assert status(gateway)["remote"]["latency"]["control_suitable"] is False

        for index in range(3):
            assert _request(gateway.url + "/infer",
                            infer_payload(f"slow-{index}", target="local"))[0] == 200

        assert _request(gateway.url + "/infer",
                        infer_payload("stays-local"))[0] == 200
        decision = _decision_for(gateway, "stays-local")
        assert decision["target"] == "local"
        assert decision["reason"] == "remote_unsuitable"
        assert decision["remote_control_suitable"] is False
    finally:
        gateway.stop()
        server.stop()


def test_local_worker_crash_does_not_touch_a_remote_session():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", target="remote"))[0] == 200

        os.kill(gateway.worker.pid, signal.SIGKILL)
        assert wait_for(lambda: status(gateway)["worker"]["state"] == "recovering")

        session = {s["session_id"]: s for s in status(gateway)["sessions"]}["remote-1"]
        assert session["target"] == "remote"
        assert session["state"] == "ready"
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", seq=2))[0] == 200
    finally:
        gateway.stop()
        server.stop()


def test_a_rejected_explicit_remote_is_not_logged_as_a_decision():
    gateway = make_gateway()
    try:
        code, body = _request(gateway.url + "/infer",
                              infer_payload("remote-1", target="remote"))
        assert code == 400
        assert body["error"]["code"] == "remote_not_configured"
        assert status(gateway)["placement"]["decisions"] == []
    finally:
        gateway.stop()
