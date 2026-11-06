#!/usr/bin/env python3
"""Run repeatable synthetic load against the model runtime.

Two loopback `motus.vla/1` gateways are started: a `delay` local worker and a
`mock` remote target. Nothing here needs a GPU, weights, or an external
network, so CI can assert the same decision table on a laptop.

Scenarios:

`placement` (default) — the #8 local-first placement contract, 10 checks:

    baseline   three new sessions on a healthy local worker      -> local
    queue      two new sessions while one local chunk is running -> remote (queue_depth)
    p95        two new sessions after a slow local chunk         -> remote (p95)
    p95 back   one new session after the p95 metric expires      -> local (local_recovered)
    unhealthy  one new session while the worker restarts         -> remote (local_unhealthy)
    recovered  one new session after the worker is ready again   -> local (local_recovered)

`failover` — the #9 mid-session failover contract:

    five local sessions are created, the local worker is killed once, and each
    session's next chunk must run on remote; after local recovery a new
    session returns to local (failback) while the old sessions stay remote.
    Chunks are never mixed: the crash is a real worker failure, and remote
    latency counts only the chunks remote actually served.

Run:

    python3 actucore/tools/runtime_load.py --json
    python3 actucore/tools/runtime_load.py --json --scenario failover

Exit code is 0 only when every check matches its expectation.
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
# Failover must be caused by a worker crash, not by the queue or p95 overflow
# thresholds: keep both high so the only reason left is local health.
FAILOVER_PLACEMENT = {
    "max_queue_depth": 100,
    "max_p95_ms": 10_000,
    "sample_window": 10,
    "ewma_alpha": 1.0,
    "p95_ttl_ms": 30_000,
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


def _gateway_config(remote_endpoint: str, placement: dict | None = None) -> dict:
    return {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "delay",
        "delay_ms": 5,
        "capabilities": dict(CAPABILITIES),
        "idle_reap_ms": 0,
        "monitor_interval_ms": 50,
        "placement": dict(placement or PLACEMENT),
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


def _session_for(gateway, session_id: str) -> dict | None:
    sessions = _status(gateway).get("sessions") or []
    matches = [session for session in sessions
               if session.get("session_id") == session_id]
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
    """Run the #8 placement scenario once and return its report."""
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


def _failover_check(gateway, session_id: str,
                    expected_remote_attempts: int) -> dict:
    """Verify one pre-existing local session's next chunk runs on remote."""
    second_code, second_body = _request(
        gateway.url + "/infer", _infer_payload(session_id, seq=2))
    after = _session_for(gateway, session_id) or {}
    remote_attempts = _status(gateway)["remote"]["latency"]["attempts"]
    ok = (second_code == 200 and bool(second_body.get("actions"))
          and after.get("target") == "remote"
          and after.get("failover_count") == 1
          and after.get("switch_count") == 1
          and after.get("failover_reason") == "local_unhealthy"
          and remote_attempts == expected_remote_attempts)
    return {
        "session_id": session_id,
        "expected_target": "remote",
        "target": after.get("target"),
        "failover_count": after.get("failover_count"),
        "switch_count": after.get("switch_count"),
        "failover_reason": after.get("failover_reason"),
        "remote_attempts": remote_attempts,
        "http_status": second_code,
        "actions": len(second_body.get("actions") or []),
        "ok": ok,
    }


def run_failover() -> dict:
    """Run the #9 mid-session failover scenario once and return its report."""
    server = RuntimeGateway(_server_config())
    server.start()
    gateway = None
    failovers = []
    sticky = {}
    failback = {}
    counters = {}
    try:
        gateway = RuntimeGateway(
            _gateway_config(server.url, placement=FAILOVER_PLACEMENT))
        gateway.start()
        if not _wait(lambda: _status(gateway)["worker"]["healthy"]):
            raise RuntimeError("local worker never became healthy")

        # Five independent sessions are pinned local before the crash. One
        # crash is enough: every one of them must fail over on its next chunk,
        # which is the "failover 5/5" gate without five restart cycles.
        session_ids = [f"failover-{index}" for index in range(1, 6)]
        for session_id in session_ids:
            code, body = _request(
                gateway.url + "/infer", _infer_payload(session_id))
            session = _session_for(gateway, session_id) or {}
            if code != 200 or not body.get("actions") \
                    or session.get("target") != "local":
                raise RuntimeError(
                    f"{session_id} did not start local: {code} {body}")
        initial_remote_attempts = _status(gateway)["remote"]["latency"]["attempts"]

        pid = gateway.worker.pid
        if pid is None:
            raise RuntimeError("local worker has no pid to crash")
        os.kill(pid, signal.SIGKILL)
        if not _wait(lambda: _status(gateway)["worker"]["state"] == "recovering"):
            raise RuntimeError("local worker never entered recovering")

        for index, session_id in enumerate(session_ids, start=1):
            failovers.append(_failover_check(gateway, session_id, index))

        recovered = _wait(
            lambda: _status(gateway)["worker"]["state"] == "ready"
            and _status(gateway)["worker"]["restart_count"] == 1, timeout=20)

        # A failed-over session must keep using remote after local recovers.
        sticky_code, sticky_body = _request(
            gateway.url + "/infer", _infer_payload("failover-1", seq=3))
        sticky_session = _session_for(gateway, "failover-1") or {}
        old_targets = {session.get("session_id"): session.get("target")
                       for session in _status(gateway)["sessions"]
                       if session.get("session_id", "").startswith("failover-")}
        sticky = {
            "session_id": "failover-1",
            "target": sticky_session.get("target"),
            "failover_count": sticky_session.get("failover_count"),
            "switch_count": sticky_session.get("switch_count"),
            "http_status": sticky_code,
            "old_session_targets": old_targets,
            "ok": (sticky_code == 200 and bool(sticky_body.get("actions"))
                   and sticky_session.get("target") == "remote"
                   and sticky_session.get("failover_count") == 1
                   and sticky_session.get("switch_count") == 1
                   and set(old_targets.values()) == {"remote"}),
        }

        # After local recovery, a new session returns to local: that is the
        # ticket's failback counter, not a forced switch of the old session.
        failback_code, failback_body = _request(
            gateway.url + "/infer", _infer_payload("failback-1"))
        failback_decision = _decision_for(gateway, "failback-1") or {}
        placement = _status(gateway)["placement"]
        counters = {
            "failover_count": placement["failover_count"],
            "failback_count": placement["failback_count"],
            "switch_count": placement["switch_count"],
        }
        failback = {
            "session_id": "failback-1",
            "target": failback_decision.get("target"),
            "reason": failback_decision.get("reason"),
            "failback_count": placement["failback_count"],
            "http_status": failback_code,
            "actions": len(failback_body.get("actions") or []),
            "ok": (failback_code == 200 and bool(failback_body.get("actions"))
                   and failback_decision.get("target") == "local"
                   and failback_decision.get("reason") == "local_recovered"
                   and placement["failback_count"] == 1),
        }
        if not recovered:
            sticky["ok"] = False
            sticky["worker_recovered"] = False
    finally:
        if gateway is not None:
            gateway.stop()
        server.stop()

    passed = sum(1 for failover in failovers if failover["ok"])
    failover_total = 5
    counters_ok = (counters.get("failover_count") == 5
                   and counters.get("failback_count") == 1
                   and counters.get("switch_count") == 6)
    return {
        "schema": "motus.runtime-load/1",
        "scenario": "mid-session-failover",
        "thresholds": dict(FAILOVER_PLACEMENT),
        "failovers": failovers,
        "sticky": sticky,
        "failback": failback,
        "counters": counters,
        "summary": {
            "total": failover_total,
            "passed": passed,
            "failed": failover_total - passed,
            "accuracy": round(passed / failover_total, 3),
            "ok": (passed == failover_total and bool(sticky.get("ok"))
                   and bool(failback.get("ok")) and counters_ok
                   and initial_remote_attempts == 0),
            "postconditions_ok": bool(sticky.get("ok"))
            and bool(failback.get("ok")) and counters_ok,
            "remote_clean_before": initial_remote_attempts == 0,
        },
    }


def run_all() -> dict:
    placement = run()
    failover = run_failover()
    total = placement["summary"]["total"] + failover["summary"]["total"]
    passed = placement["summary"]["passed"] + failover["summary"]["passed"]
    return {
        "schema": "motus.runtime-load/1",
        "scenario": "all",
        "placement": placement,
        "failover": failover,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": round(passed / total, 3) if total else 0.0,
            "ok": placement["summary"]["ok"] and failover["summary"]["ok"],
        },
    }


def _print_report(report: dict) -> None:
    if report["scenario"] == "all":
        _print_report(report["placement"])
        _print_report(report["failover"])
    elif report["scenario"] == "mid-session-failover":
        for failover in report["failovers"]:
            mark = "PASS" if failover["ok"] else "FAIL"
            print(f"{mark} {failover['session_id']}: expected remote, "
                  f"got {failover['target']} after "
                  f"{failover['failover_reason']}")
        print(f"sticky: {'PASS' if report['sticky']['ok'] else 'FAIL'}; "
              f"failback: {'PASS' if report['failback']['ok'] else 'FAIL'}; "
              f"counters: {report['counters']}")
    else:
        for decision in report["decisions"]:
            mark = "PASS" if decision["ok"] else "FAIL"
            print(f"{mark} {decision['session_id']}: expected "
                  f"{decision['expected_target']}/{decision['expected_reason']}, "
                  f"got {decision['target']}/{decision['reason']}")
    summary = report["summary"]
    label = ("failover checks" if report["scenario"] == "mid-session-failover"
             else "placement decisions")
    print(f"{label}: {summary['passed']}/{summary['total']} "
          f"({summary['accuracy']:.0%}) — "
          f"{'PASS' if summary['ok'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="print the full machine-readable report")
    parser.add_argument("--output", default="",
                        help="also write the machine-readable report here")
    parser.add_argument("--scenario", choices=("placement", "failover", "all"),
                        default="placement",
                        help="which synthetic load scenario to run")
    args = parser.parse_args(argv)

    if args.scenario == "failover":
        report = run_failover()
    elif args.scenario == "all":
        report = run_all()
    else:
        report = run()
    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
