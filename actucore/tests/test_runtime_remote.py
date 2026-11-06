"""The `remote_http` backend, exercised over `motus.vla/1`.

A remote target is another `motus.vla/1` server — the laptop RTX 4060 tool uses
the same contract — so every test here starts a real loopback gateway and
points the local gateway at it. Nothing reaches into the HTTP client: what is
asserted is what an operator or the card can see through `/infer` and
`/status`.

The real-hardware acceptance test stays skipped unless a laptop server and a
local checkpoint are both provided; a mock measurement is never reported as a
real RTX 4060 result.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_runtime_remote.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.vla_cloud import (  # noqa: E402
    SCHEMA, VLACloudProvider)
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


def make_server(**overrides):
    """A loopback `motus.vla/1` server, the same process the laptop tool
    runs."""
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


def observation(**overrides):
    base = dict(images={"main": b"jpegbytes"}, state=[0.0] * 7,
                prompt="pick it up", t_capture_ms=1_700_000_000_000)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, method="POST" if body else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def infer_payload(session_id, seq=1, **overrides):
    payload = {"schema": SCHEMA, "session_id": session_id, "seq": seq,
               "t_capture_ms": 1_700_000_000_000, "prompt": "pick it up",
               "images": {}, "inference_delay": 0}
    payload.update(overrides)
    return payload


def remote_status(gateway):
    return _request(gateway.url + "/status")[1]["remote"]


# ── routing ──────────────────────────────────────────────────────────────────

def test_an_explicit_session_runs_on_the_remote_target():
    """`target: remote` on the first infer pins that session to the remote."""
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))

        assert status == 200
        assert len(body["actions"]) == CAPABILITIES["chunk_size"]
        assert all(len(step) == CAPABILITIES["action_dim"]
                   for step in body["actions"])

        session = _request(gateway.url + "/status")[1]["sessions"][0]
        assert session["target"] == "remote"
        remote = remote_status(gateway)
        assert remote["backend"] == "remote_http"
        assert remote["latency"]["samples"] == 1
    finally:
        gateway.stop()
        server.stop()


def test_a_session_never_switches_target_between_chunks():
    """The first infer pins the target; the next chunks must not move."""
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", target="remote"))[0] == 200
        # No target field on the next request: the existing session keeps its
        # placement rather than quietly falling back to local.
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", seq=2))[0] == 200

        status, body = _request(gateway.url + "/infer",
                                infer_payload(
                                    "remote-1", seq=3, target="local"))
        assert status == 409
        assert body["error"]["code"] == "target_conflict"
        assert "remote" in body["error"]["message"]

        session = _request(gateway.url + "/status")[1]["sessions"][0]
        assert session["target"] == "remote"
        assert remote_status(gateway)["latency"]["samples"] == 2
    finally:
        gateway.stop()
        server.stop()


def test_a_session_without_a_target_stays_local():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("local-1"))
        assert status == 200
        session = _request(gateway.url + "/status")[1]["sessions"][0]
        assert session["target"] == "local"
        assert remote_status(gateway)["latency"]["samples"] == 0
    finally:
        gateway.stop()
        server.stop()


def test_remote_target_must_be_configured():
    gateway = make_gateway()
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))
        assert status == 400
        assert body["error"]["code"] == "remote_not_configured"
    finally:
        gateway.stop()


# ── capability identity ──────────────────────────────────────────────────────

def test_a_remote_target_with_different_capabilities_is_refused():
    """A different checkpoint must be refused, in words an operator can
    act on."""
    server = make_server(capabilities={
        **CAPABILITIES, "model": "smolvla_other", "action_dim": 6,
        "chunk_size": 4, "control_mode": "eef_pose", "control_hz": 15,
    })
    gateway = make_gateway(server.url)
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))

        assert status == 502
        assert body["error"]["code"] == "remote_capabilities_mismatch"
        message = body["error"]["message"]
        for expected in ("model", "action_dim", "chunk_size", "control_mode",
                         "control_hz", "smolvla_other", "mock-sine@8s",
                         "6", "7", "15"):
            assert expected in message, expected

        remote = remote_status(gateway)
        assert remote["compatible"] is False
        assert "action_dim" in remote["last_error"]

        # The local target is unaffected: refusing remote is not a downgrade.
        status, body = _request(gateway.url + "/infer",
                                infer_payload("local-1"))
        assert status == 200 and body["actions"]
    finally:
        gateway.stop()
        server.stop()


# ── RTT and the driver budget ────────────────────────────────────────────────

def test_remote_rtt_and_p95_are_recorded_and_fit_the_driver_budget():
    server = make_server(backend="delay", delay_ms=40)
    gateway = make_gateway(server.url)
    try:
        for seq in range(1, 6):
            status, body = _request(gateway.url + "/infer",
                                    infer_payload(
                                        "remote-1", seq=seq, target="remote"))
            assert status == 200 and body["actions"]

        latency = remote_status(gateway)["latency"]
        assert latency["samples"] == 5
        assert latency["attempts"] == 5
        assert latency["failures"] == 0
        assert latency["last_rtt_ms"] >= 35
        assert latency["p95_rtt_ms"] >= 35
        assert latency["budget"] == {"max_obs_age_ms": 300.0,
                                     "watchdog_ms": 200.0}
        assert latency["budget_configured"] is True
        assert latency["within_max_obs_age"] is True
        assert latency["within_watchdog"] is True
        assert latency["control_suitable"] is True
        assert latency["note"] == ""
    finally:
        gateway.stop()
        server.stop()


def test_remote_p95_over_the_driver_budget_is_offline_only():
    """A slow laptop target is still useful, but not for closed-loop
    control."""
    server = make_server(backend="delay", delay_ms=60)
    gateway = make_gateway(server.url, remote={
        "endpoint": server.url,
        "timeout_ms": 3000,
        "max_obs_age_ms": 20,
        "watchdog_ms": 30,
    })
    try:
        for seq in range(1, 4):
            status, body = _request(gateway.url + "/infer",
                                    infer_payload(
                                        "remote-1", seq=seq, target="remote"))
            assert status == 200 and body["actions"]

        latency = remote_status(gateway)["latency"]
        assert latency["samples"] == 3
        assert latency["p95_rtt_ms"] > 50
        assert latency["within_max_obs_age"] is False
        assert latency["within_watchdog"] is False
        assert latency["control_suitable"] is False
        assert "offline" in latency["note"]
        assert "max_obs_age_ms" in latency["note"]
        assert "watchdog_ms" in latency["note"]
    finally:
        gateway.stop()
        server.stop()

# ── the laptop test server ───────────────────────────────────────────────────

def test_the_laptop_server_uses_the_same_checkpoint_settings_as_the_robot():
    """The tool is the same gateway and config — not a second model."""
    from tools.remote_vla_server import server_config

    card = {
        "model_name": "smolvla_base",
        "models": {"smolvla_base": {"model_dir": "/models/vla/smolvla_base",
                                    "unnorm_key": "so100"}},
        "feature_map": {"main": "observation.images.camera1"},
        "device": "cuda",
    }
    config = server_config({
        "runtime": {"backend": "mock", "port": 15732,
                    "remote": {"endpoint": "http://somewhere:15732"}},
        "plugins": {"vla": card},
    }, host="0.0.0.0", port=15733)

    assert config["backend"] == "smolvla"
    assert config["host"] == "0.0.0.0"
    assert config["port"] == 15733
    assert "remote" not in config        # a server does not become a client
    assert config["smolvla"]["model_name"] == "smolvla_base"
    assert config["smolvla"]["models"] == card["models"]
    assert config["smolvla"]["models"]["smolvla_base"]["unnorm_key"] == "so100"
    assert config["smolvla"]["feature_map"] == card["feature_map"]


def test_the_laptop_server_lets_an_explicit_checkpoint_override_the_card():
    from tools.remote_vla_server import server_config

    config = server_config({"plugins": {"vla": {
        "model_name": "smolvla_base",
        "models": {"smolvla_base": {"model_dir": "/models/vla/smolvla_base"},
                   "smolvla_tianyi": {"model_dir": "/old/laptop/path"}}}}},
        model_name="smolvla_tianyi",
        model_dir="/models/vla/smolvla_tianyi",
        device="cuda:0")

    assert config["smolvla"]["model_name"] == "smolvla_tianyi"
    assert config["smolvla"]["model_dir"] == "/models/vla/smolvla_tianyi"
    assert config["smolvla"]["models"]["smolvla_tianyi"]["model_dir"] == \
           "/models/vla/smolvla_tianyi"
    assert config["smolvla"]["device"] == "cuda:0"


def test_remote_configuration_does_not_change_the_local_handshake():
    """The card still negotiates against the local worker, not the remote."""
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        capabilities = VLACloudProvider(
            {}, {"endpoint": gateway.url}).capabilities()
        assert capabilities["schema"] == SCHEMA
        assert capabilities["model"] == MODEL
        assert capabilities["action_dim"] == CAPABILITIES["action_dim"]
    finally:
        gateway.stop()
        server.stop()


# ── real hardware acceptance (skipped by default) ───────────────────────────

REAL_ENDPOINT = os.environ.get("ACTUCORE_REMOTE_VLA_ENDPOINT", "")
REAL_CHECKPOINT = os.environ.get("ACTUCORE_SMOLVLA_CHECKPOINT", "")


@pytest.mark.skipif(not (REAL_ENDPOINT and REAL_CHECKPOINT),
                    reason=("hardware acceptance: set "
                            "ACTUCORE_REMOTE_VLA_ENDPOINT to a laptop RTX "
                            "4060 motus.vla/1 server and "
                            "ACTUCORE_SMOLVLA_CHECKPOINT to the same "
                            "checkpoint here"))
def test_real_laptop_rtx4060_rtt_is_recorded():
    """Real RTX 4060 acceptance. The loopback tests above are not this test."""
    gateway = RuntimeGateway({
        "enabled": True, "host": "127.0.0.1", "port": 0, "backend": "smolvla",
        "smolvla": {"model_dir": REAL_CHECKPOINT,
                    "device": os.environ.get(
                        "ACTUCORE_SMOLVLA_DEVICE", "cuda")},
        "idle_reap_ms": 0,
        "remote": {"endpoint": REAL_ENDPOINT, "timeout_ms": 5000,
                   "max_obs_age_ms": 300, "watchdog_ms": 200},
    })
    gateway.start()
    try:
        for seq in range(1, 11):
            status, body = _request(gateway.url + "/infer",
                                    infer_payload(
                                        "rtx", seq=seq, target="remote"))
            assert status == 200 and body["actions"]

        latency = remote_status(gateway)["latency"]
        assert latency["samples"] == 10
        assert latency["p95_rtt_ms"] > 0
        # Record the verdict rather than forcing a pass: a slow laptop target
        # is allowed to be explicitly offline-only.
        if latency["control_suitable"]:
            assert latency["note"] == ""
        else:
            assert "offline" in latency["note"]
    finally:
        gateway.stop()


def test_hard_cancel_of_a_remote_session_does_not_kill_the_local_worker():
    """A remote forward cannot be preempted; the local worker must survive."""
    server = make_server(backend="delay", delay_ms=300)
    gateway = make_gateway(server.url)
    try:
        reply = {}

        def run():
            reply["value"] = _request(
                gateway.url + "/infer",
                infer_payload("remote-1", target="remote"))

        thread = threading.Thread(target=run)
        thread.start()

        def in_flight():
            sessions = _request(gateway.url + "/status")[1]["sessions"]
            return any(session["in_flight"] for session in sessions)

        deadline = time.monotonic() + 5
        while not in_flight() and time.monotonic() < deadline:
            time.sleep(0.01)
        local_pid = gateway.worker.pid

        status, body = _request(gateway.url + "/cancel",
                                {"session_id": "remote-1", "hard": True})
        assert status == 202
        assert body["state"] == "cancelling"

        thread.join(timeout=5)
        assert reply["value"][0] == 409
        assert reply["value"][1]["error"]["code"] == "cancelled"
        assert gateway.worker.pid == local_pid
        assert gateway.worker.status()["restart_count"] == 0
    finally:
        gateway.stop()
        server.stop()


def test_an_unreachable_remote_target_returns_a_readable_error():
    gateway = make_gateway("http://127.0.0.1:1", remote={
        "endpoint": "http://127.0.0.1:1", "timeout_ms": 100})
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))
        assert status == 503
        assert body["error"]["code"] == "remote_unavailable"
        assert "127.0.0.1:1" in body["error"]["message"]

        remote = remote_status(gateway)
        assert remote["compatible"] is None
        assert "127.0.0.1:1" in remote["last_error"]
        assert remote["latency"]["attempts"] == 0
    finally:
        gateway.stop()


def test_a_remote_speaking_another_schema_is_refused():
    """Schema is part of the identity check, not a version-negotiation
    detail."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps(
                {**CAPABILITIES, "schema": "motus.vla/2"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gateway = make_gateway(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))
        assert status == 502
        assert body["error"]["code"] == "remote_capabilities_mismatch"
        assert "motus.vla/2" in body["error"]["message"]
        assert "schema" in body["error"]["message"]
    finally:
        gateway.stop()
        server.shutdown()
        server.server_close()


def test_closing_a_remote_session_closes_it_on_the_server_too():
    server = make_server()
    gateway = make_gateway(server.url)
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", target="remote"))[0] == 200

        status, body = _request(
            gateway.url + "/close", {"session_id": "remote-1"})
        assert status == 200
        assert body["state"] == "terminated"

        remote_session = _request(server.url + "/status")[1]["sessions"][0]
        assert remote_session["session_id"] == "remote-1"
        assert remote_session["state"] == "terminated"
    finally:
        gateway.stop()
        server.stop()


def test_a_new_session_revalidates_remote_capabilities():
    """A laptop restart with different weights must not keep the old check."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    capabilities = {**CAPABILITIES, "schema": SCHEMA}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._json(capabilities)

        def do_POST(self):
            self._json({"seq": 1, "actions": [[0.0] * 7]})

        def _json(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gateway = make_gateway(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("s1", target="remote"))
        assert status == 200 and body["actions"]

        capabilities["model"] = "smolvla_after_restart"
        status, body = _request(gateway.url + "/infer",
                                infer_payload("s2", target="remote"))
        assert status == 502
        assert body["error"]["code"] == "remote_capabilities_mismatch"
        assert "smolvla_after_restart" in body["error"]["message"]
    finally:
        gateway.stop()
        server.shutdown()
        server.server_close()


def test_a_failed_remote_inference_counts_as_a_failed_attempt():
    """A dead laptop must not look suitable because old samples are fast."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    capabilities = {**CAPABILITIES, "schema": SCHEMA}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps(capabilities).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.connection.close()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gateway = make_gateway(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        status, body = _request(gateway.url + "/infer",
                                infer_payload("remote-1", target="remote"))
        assert status == 503
        assert body["error"]["code"] == "remote_unavailable"

        latency = remote_status(gateway)["latency"]
        assert latency["attempts"] == 1
        assert latency["samples"] == 0
        assert latency["failures"] == 1
        assert latency["control_suitable"] is False
        assert "offline" in latency["note"]
    finally:
        gateway.stop()
        server.shutdown()
        server.server_close()


def test_remote_without_a_configured_budget_does_not_claim_suitability():
    """No configured driver budget means no honest suitability verdict."""
    server = make_server()
    gateway = make_gateway(server.url, remote={
        "endpoint": server.url, "timeout_ms": 3000})
    try:
        assert _request(gateway.url + "/infer",
                        infer_payload("remote-1", target="remote"))[0] == 200
        latency = remote_status(gateway)["latency"]
        assert latency["budget_configured"] is False
        assert latency["control_suitable"] is None
        assert latency["within_max_obs_age"] is None
        assert "not configured" in latency["note"]
    finally:
        gateway.stop()
        server.stop()


def test_the_remote_backend_name_is_validated():
    """The remote target declares its backend; a typo fails at startup."""
    with pytest.raises(ValueError) as excinfo:
        RuntimeGateway({"enabled": True, "backend": "mock",
                        "remote": {"backend": "not-a-backend",
                                   "endpoint": "http://127.0.0.1:1"}})
    assert "remote_http" in str(excinfo.value)
