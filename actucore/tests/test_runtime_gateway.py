"""The model runtime gateway, exercised over `motus.vla/1`.

Every test here talks to the gateway the way the card does — through
`VLACloudProvider` or a plain HTTP request — because that is the only contract
the rest of the system knows about. Worker and session internals are not
imported unless the test is about killing them.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_runtime_gateway.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import sys
import threading
import time
import types
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.mock import MockProvider  # noqa: E402
from plugins.vla.providers.vla_cloud import SCHEMA, VLACloudProvider  # noqa: E402
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
DESCRIPTOR = {
    "control_interface": "motus.control/1",
    "mode": "joint_position",
    "dof": 7,
    "joint_names": [f"joint{i}" for i in range(1, 8)],
    "units": {"angle": "rad"},
    "limits": {"lower": [-2.0] * 7, "upper": [2.0] * 7},
    "rate": {"max_hz": 100, "expected_hz": 30, "watchdog_ms": 200},
    "force_torque": None,
}


def make_gateway(**overrides):
    config = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "mock",
        "capabilities": dict(CAPABILITIES),
    }
    config.update(overrides)
    gateway = RuntimeGateway(config)
    gateway.start()
    return gateway


def observation(**overrides):
    base = dict(images={"main": b"jpegbytes"}, state=[0.0] * 7,
                prompt="pick it up", t_capture_ms=1_700_000_000_000)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _request(url, payload=None):
    """One raw HTTP call, for the cases where the client must see the status."""
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method="POST" if body else "GET",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def infer_payload(session_id, seq=1):
    return {"schema": SCHEMA, "session_id": session_id, "seq": seq,
            "t_capture_ms": 1_700_000_000_000, "prompt": "pick it up",
            "images": {}, "inference_delay": 0}


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ── handshake and the action chunk ───────────────────────────────────────────

def test_capabilities_match_the_direct_provider():
    gateway = make_gateway()
    try:
        direct = MockProvider(DESCRIPTOR, chunk_size=5).capabilities()
        through_gateway = VLACloudProvider({}, {"endpoint": gateway.url}).capabilities()

        assert through_gateway["schema"] == SCHEMA
        assert {key: through_gateway[key] for key in direct} == direct
    finally:
        gateway.stop()


def test_infer_returns_a_mock_action_chunk():
    gateway = make_gateway()
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        actions = provider.infer(observation())

        assert len(actions) == CAPABILITIES["chunk_size"]
        assert all(len(step) == CAPABILITIES["action_dim"] for step in actions)

        status, body = _request(gateway.url + "/status")
        assert status == 200
        assert body["sessions"][0]["state"] == "ready"
        assert body["sessions"][0]["inference_count"] == 1
    finally:
        gateway.stop()


def test_an_empty_chunk_is_an_error_not_a_pause(monkeypatch):
    gateway = make_gateway()
    try:
        monkeypatch.setattr(gateway._worker, "request", lambda _payload: [])
        status, body = _request(gateway.url + "/infer", infer_payload("s1"))
        assert status == 502
        assert body["error"]["code"] == "empty_chunk"
    finally:
        gateway.stop()


def test_a_different_schema_is_refused_before_inference():
    gateway = make_gateway()
    try:
        status, body = _request(gateway.url + "/infer",
                                {**infer_payload("s1"), "schema": "motus.vla/2"})
        assert status == 400
        assert body["error"]["code"] == "schema_mismatch"
    finally:
        gateway.stop()


# ── one inference at a time ──────────────────────────────────────────────────

def test_concurrent_inference_on_one_session_is_busy():
    gateway = make_gateway(backend="delay", delay_ms=400)
    try:
        first = {}

        def run_first():
            first["reply"] = _request(gateway.url + "/infer", infer_payload("s1"))

        thread = threading.Thread(target=run_first)
        thread.start()
        assert wait_for(lambda: any(s["in_flight"] for s in
                                    _request(gateway.url + "/status")[1]["sessions"]))

        status, body = _request(gateway.url + "/infer", infer_payload("s1", seq=2))
        assert status == 409
        assert body["error"]["code"] == "busy"

        thread.join(timeout=5)
        assert first["reply"][0] == 200
    finally:
        gateway.stop()


def test_health_answers_while_an_inference_is_in_flight():
    gateway = make_gateway(backend="delay", delay_ms=500)
    try:
        def run_infer():
            _request(gateway.url + "/infer", infer_payload("s1"))

        thread = threading.Thread(target=run_infer)
        thread.start()
        assert wait_for(lambda: any(s["in_flight"] for s in
                                    _request(gateway.url + "/status")[1]["sessions"]))

        started = time.monotonic()
        status, body = _request(gateway.url + "/healthz")
        elapsed = time.monotonic() - started

        assert status == 200 and body["status"] == "ok"
        assert elapsed < 0.2, f"health waited behind inference: {elapsed:.3f}s"
        thread.join(timeout=5)
    finally:
        gateway.stop()


def test_cancel_does_not_deliver_the_old_chunk():
    gateway = make_gateway(backend="delay", delay_ms=500)
    try:
        reply = {}

        def run_infer():
            reply["value"] = _request(gateway.url + "/infer", infer_payload("s1"))

        thread = threading.Thread(target=run_infer)
        thread.start()
        assert wait_for(lambda: any(s["in_flight"] for s in
                                    _request(gateway.url + "/status")[1]["sessions"]))

        status, body = _request(gateway.url + "/close", {"session_id": "s1"})
        assert status == 200
        assert body["state"] in ("cancelling", "terminated")

        thread.join(timeout=5)
        status, body = reply["value"]
        assert status == 409
        assert body["error"]["code"] == "cancelled"
        assert "actions" not in body
    finally:
        gateway.stop()


# ── process isolation ────────────────────────────────────────────────────────

def test_the_worker_is_spawned_not_forked():
    """A forked child would inherit ActuCore's CUDA and vendor runtime state."""
    gateway = make_gateway()
    try:
        assert gateway.worker.start_method == "spawn"
    finally:
        gateway.stop()


def test_killing_the_worker_leaves_the_gateway_alive():
    gateway = make_gateway()
    try:
        pid = gateway.worker.pid
        assert pid and gateway.worker.status()["state"] == "ready"

        os.kill(pid, signal.SIGKILL)
        assert wait_for(lambda: gateway.worker.status()["state"] == "dead")

        status, body = _request(gateway.url + "/healthz")
        assert status == 503 and body["status"] == "unavailable"

        status, body = _request(gateway.url + "/status")
        assert status == 200
        assert body["worker"]["state"] == "dead"
        assert body["worker"]["restart_count"] == 0
        assert gateway.worker.pid == pid          # no automatic restart

        # The card is a plain object in this process; a dead worker must make
        # it hold (health false) rather than raise or take the process down.
        from plugins.vla import VLAPlugin
        card = VLAPlugin({"provider": "vla_cloud", "endpoint": gateway.url},
                         executor=None)
        card._provider = VLACloudProvider({}, {"endpoint": gateway.url})
        card._running = True

        class _Publisher:
            published = 0

            def publish(self, _message):
                self.published += 1

        card._publisher = _Publisher()
        assert card._state() == "loading"
        card._tick()                          # no exception, publishes nothing
        assert card._publisher.published == 0
    finally:
        gateway.stop()


# ── the card's existing path, unchanged ──────────────────────────────────────

def test_the_unchanged_card_starts_and_runs_on_the_runtime(monkeypatch):
    """The card's real start path negotiates against the gateway's caps."""
    from plugins.vla import VLAPlugin

    gateway = make_gateway()
    try:
        card = VLAPlugin({"provider": "vla_cloud", "endpoint": gateway.url,
                          "cloud_model_name": MODEL}, executor=None)
        # ROS only: the provider, capabilities read and negotiation are the
        # parts under test, and all three happen before any node exists.
        monkeypatch.setattr(card, "_open_publisher", lambda mode: None)
        monkeypatch.setattr(card, "_bind_inputs", lambda node, inputs, caps: ({}, ""))
        monkeypatch.setattr(card, "_attach_node", lambda: None)

        started = card._start({"control_interface": DESCRIPTOR, "task": "pick it up"})
        assert started["state"] == "running"

        command = card.next_command()

        assert command["schema"] == "motus.control/1"
        assert command["dof"] == 7
        assert len(command["values"]) == 7
        assert command["chunk"] == {"index": 0, "size": 5}
    finally:
        gateway.stop()


# ── the rollback switch ─────────────────────────────────────────────────────

def test_runtime_disabled_leaves_the_card_on_its_existing_provider():
    from runtime import vla_config

    original = {"provider": "mock", "period_s": 4.0}
    assert vla_config(original, {"enabled": False}) == original


def test_runtime_enabled_points_the_card_at_its_local_gateway():
    from runtime import vla_config

    routed = vla_config({"provider": "mock", "period_s": 4.0},
                        {"enabled": True, "host": "127.0.0.1", "port": 15732,
                         "capabilities": {"model": MODEL}})

    assert routed["provider"] == "vla_cloud"
    assert routed["endpoint"] == "http://127.0.0.1:15732"
    assert routed["cloud_model_name"] == MODEL
    assert routed["period_s"] == 4.0


def test_cancel_keeps_the_session_usable():
    """Cancel is about the in-flight chunk, not the card's whole lifetime."""
    gateway = make_gateway(backend="delay", delay_ms=400)
    try:
        reply = {}

        def run_infer():
            reply["value"] = _request(gateway.url + "/infer", infer_payload("s1"))

        thread = threading.Thread(target=run_infer)
        thread.start()
        assert wait_for(lambda: any(s["in_flight"] for s in
                                    _request(gateway.url + "/status")[1]["sessions"]))

        status, body = _request(gateway.url + "/cancel", {"session_id": "s1"})
        assert status == 200 and body["state"] == "cancelling"

        thread.join(timeout=5)
        assert reply["value"][0] == 409
        assert reply["value"][1]["error"]["code"] == "cancelled"

        assert wait_for(lambda: _request(gateway.url + "/status")[1]["sessions"][0]["state"]
                        == "ready")
        status, body = _request(gateway.url + "/infer", infer_payload("s1", seq=2))
        assert status == 200 and body["actions"]
    finally:
        gateway.stop()
