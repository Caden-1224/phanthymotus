#!/usr/bin/env python3
"""Run a repeatable multi-session synthetic load against local-first placement.

Two loopback `motus.vla/1` gateways are started: a `delay` local worker and a
`mock` remote target. Nothing here needs a GPU, weights, or an external
network, so CI can assert the same decision table on a laptop.

The ten checks are the placement contract, not a latency benchmark:

    baseline   three new sessions on a healthy local worker      -> local
    queue      two new sessions while one local chunk is running -> remote (queue_depth)
    p95        two new sessions after a slow local chunk         -> remote (p95)
    p95 back   one new session after the p95 metric expires      -> local (local_recovered)
    unhealthy  one new session while the worker restarts         -> remote (local_unhealthy)
    recovered  one new session after the worker is ready again   -> local (local_recovered)

Run:

    python3 actucore/tools/runtime_load.py --json

Exit code is 0 only when every decision matches its expected target and reason.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.vla_cloud import SCHEMA  # noqa: E402
from runtime import RuntimeGateway  # noqa: E402

CAPABILITIES = {
    "model": "mock-sine@8s",
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
    # A single outstanding local chunk is already over capacity for this
    # synthetic run; real deployments should leave room for one in-flight
    # chunk and use the queue threshold for the backlog behind it.
    "max_queue_depth": 0,
    "max_p95_ms": 150,
    "sample_window": 10,
    "ewma_alpha": 0.5,
    # Short enough that the p95-recovered check does not wait half a minute;
    # production uses the config.yaml default.
    "p95_ttl_ms": 800,
    "decision_log_size": 100,
}


def _server_config() -> dict:
    return {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "mock",
        "capabilities": dict(CAPABILITIES),
        "idle_reap_ms": 0,
    }


def _gateway_config(remote_endpoint: str) -> dict:
    return {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "delay",
        "delay_ms": 5,
        "capabilities": dict(CAPABILITIES),
        "idle_reap_ms": 0,
        "monitor_interval_ms": 50,
        "placement": dict(PLACEMENT),
        "remote": {
            "backend": "remote_http",
            "endpoint": remote_endpoint,
            "timeout_ms": 3000,
            "max_obs_age_ms": 300,
            "watchdog_ms": 200,
        },
    }


def _request(url: str, payload: dict | None = None, timeout: float = 10):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, method="POST" if body else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def _status(gateway) -> dict:
    return _request(gateway.url + "/status")[1]


def _infer_payload(session_id: str, seq: int = 1, target: str | None = None,
                   delay_ms: float | None = None) -> dict:
    payload = {
        "schema": SCHEMA,
        "session_id": session_id,
        "seq": seq,
        "t_capture_ms": 1_700_000_000_000,
        "prompt": "synthetic placement load",
        "images": {},
        "inference_delay": 0,
    }
    if target is not None:
        payload["target"] = target
    if delay_ms is not None:
        payload["delay_ms"] = delay_ms
    return payload


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _decision_for(gateway, session_id: str) -> dict | None:
    decisions = _status(gateway).get("placement", {}).get("decisions") or []
    matches = [decision for decision in decisions
               if decision.get("session_id") == session_id]
    return matches[-1] if matches else None


def _check(gateway, session_id: str, expected_target: str,
           expected_reason: str, **payload) -> dict:
    code, body = _request(gateway.url + "/infer",
                          _infer_payload(session_id, **payload))
    decision = _decision_for(gateway, session_id) or {}
    p95 = decision.get("p95_ms")
    ok = (code == 200
          and bool(body.get("actions"))
          and decision.get("placement_decision") == expected_target
          and decision.get("target") == expected_target
          and decision.get("reason") == expected_reason
          and isinstance(decision.get("queue_depth"), int)
          and "p95" in decision
          and "p95_ms" in decision)
    if expected_reason == "queue_depth":
        ok = ok and decision["queue_depth"] >= 1
    if expected_reason == "p95":
        ok = ok and isinstance(p95, (int, float)) and p95 > PLACEMENT["max_p95_ms"]
    return {
        "session_id": session_id,
        "expected_target": expected_target,
        "expected_reason": expected_reason,
        "target": decision.get("target"),
        "placement_decision": decision.get("placement_decision"),
        "reason": decision.get("reason"),
        "queue_depth": decision.get("queue_depth"),
        "p95": p95,
        "p95_ms": p95,
        "http_status": code,
        "ok": ok,
    }


def run() -> dict:
    """Run the scenario once and return the machine-readable report."""
    server = RuntimeGateway(_server_config())
    server.start()
    gateway = None
    decisions = []
    try:
        # The endpoint is only known after the server binds its ephemeral port.
        gateway = RuntimeGateway(_gateway_config(server.url))
        gateway.start()

        # Three clean local sessions establish a low p95 baseline.
        for index in range(3):
            decisions.append(_check(gateway, f"baseline-{index}", "local",
                                    "local_first"))

        # One slow local chunk occupies the worker; the two sessions created
        # while it runs see queue_depth > max_queue_depth.
        blocker = {}

        def run_blocker():
            blocker["reply"] = _request(
                gateway.url + "/infer",
                _infer_payload("queue-blocker", target="local", delay_ms=400))

        thread = threading.Thread(target=run_blocker, daemon=True)
        thread.start()
        if not _wait(lambda: _status(gateway)["worker"]["queue_depth"] >= 1):
            raise RuntimeError("local queue depth never rose above zero")
        decisions.append(_check(gateway, "queue-0", "remote", "queue_depth"))
        decisions.append(_check(gateway, "queue-1", "remote", "queue_depth"))
        thread.join(timeout=15)
        if blocker.get("reply", (0,))[0] != 200:
            raise RuntimeError(f"queue blocker failed: {blocker.get('reply')}")

        # The blocker's 400ms chunk is now in the EWMA, so a new session sees
        # p95 over the 150ms threshold.
        if not _wait(lambda: (_status(gateway)["placement"]["local"]["p95_ms"]
                              or 0) > PLACEMENT["max_p95_ms"]):
            raise RuntimeError("local EWMA p95 did not cross the threshold")
        decisions.append(_check(gateway, "p95-0", "remote", "p95"))
        decisions.append(_check(gateway, "p95-1", "remote", "p95"))

        # Once the p95 sample expires, the next new session probes local again
        # instead of staying remote forever with no fresh local samples.
        if not _wait(lambda: _status(gateway)["placement"]["local"]["p95_ms"]
                     is None, timeout=5):
            raise RuntimeError("local p95 never expired after its TTL")
        decisions.append(_check(gateway, "p95-recovered", "local",
                                "local_recovered"))

        # Kill the local worker: new sessions must use the remote until the
        # replacement is ready, then go back to local.
        pid = gateway.worker.pid
        os.kill(pid, signal.SIGKILL)
        if not _wait(lambda: _status(gateway)["worker"]["state"] == "recovering"):
            raise RuntimeError("local worker never entered recovering")
        decisions.append(_check(gateway, "unhealthy", "remote",
                                "local_unhealthy"))

        if not _wait(lambda: _status(gateway)["worker"]["state"] == "ready"
                     and _status(gateway)["worker"]["restart_count"] == 1,
                     timeout=15):
            raise RuntimeError("local worker did not become ready again")
        decisions.append(_check(gateway, "recovered", "local", "local_recovered"))
    finally:
        if gateway is not None:
            gateway.stop()
        server.stop()

    passed = sum(1 for decision in decisions if decision["ok"])
    return {
        "schema": "motus.runtime-load/1",
        "scenario": "local-first-placement",
        "thresholds": dict(PLACEMENT),
        "decisions": decisions,
        "summary": {
            "total": len(decisions),
            "passed": passed,
            "failed": len(decisions) - passed,
            "accuracy": round(passed / len(decisions), 3) if decisions else 0.0,
            "ok": bool(decisions) and passed == len(decisions),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="print the full machine-readable report")
    parser.add_argument("--output", default="",
                        help="also write the machine-readable report here")
    args = parser.parse_args(argv)

    report = run()
    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for decision in report["decisions"]:
            mark = "PASS" if decision["ok"] else "FAIL"
            print(f"{mark} {decision['session_id']}: expected "
                  f"{decision['expected_target']}/{decision['expected_reason']}, "
                  f"got {decision['target']}/{decision['reason']}")
        summary = report["summary"]
        print(f"placement decisions: {summary['passed']}/{summary['total']} "
              f"({summary['accuracy']:.0%}) — "
              f"{'PASS' if summary['ok'] else 'FAIL'}")
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
