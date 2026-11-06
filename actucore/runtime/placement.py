"""Local-first placement for new sessions.

The policy is consulted exactly once, when a session is created. After that the
session owns its target for life: this module is never asked where a later
chunk should run, which is what makes per-chunk switching impossible.

A new session stays local unless the local worker is unavailable, the local
queue is already over the configured depth, or the local EWMA p95 is over the
configured latency. Even then it only moves to a remote target that is
configured, reachable, serving the same checkpoint identity and not already
flagged unsuitable for closed-loop control by the driver budget.
"""

from __future__ import annotations

import collections
import threading
import time

from .metrics import p95 as _p95
from .session import LOCAL_TARGET, REMOTE_TARGET

POLICY = "local_first"


class LocalLatency:
    """EWMA of the rolling p95 of local inference latency.

    A stale value is treated as unknown: after a p95-triggered overflow no new
    local samples arrive, and without expiry the policy would never try the
    recovered local worker again. The first new sample after the TTL starts a
    fresh window instead of blending in old trouble.
    """

    def __init__(self, window: int, alpha: float, ttl_ms: float):
        if int(window) < 1:
            raise ValueError("placement.sample_window must be at least 1")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("placement.ewma_alpha must be between 0 and 1")
        self._samples = collections.deque(maxlen=int(window))
        self._alpha = float(alpha)
        self._ttl_s = max(0.0, float(ttl_ms)) / 1000.0
        self._last_sample = None
        self._ewma = None
        self._lock = threading.Lock()

    def _stale_locked(self) -> bool:
        return (self._ttl_s > 0 and self._last_sample is not None
                and time.monotonic() - self._last_sample > self._ttl_s)

    def record(self, elapsed_ms: float) -> None:
        with self._lock:
            if self._stale_locked():
                self._samples.clear()
                self._ewma = None
            self._samples.append(float(elapsed_ms))
            sample = _p95(self._samples)
            self._ewma = (sample if self._ewma is None
                          else self._alpha * sample + (1 - self._alpha) * self._ewma)
            self._last_sample = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._ewma = None
            self._last_sample = None

    @property
    def p95_ms(self) -> float | None:
        with self._lock:
            if self._ewma is None or self._stale_locked():
                return None
            return round(self._ewma, 1)

    @property
    def ttl_ms(self) -> float:
        return self._ttl_s * 1000.0


class PlacementPolicy:
    def __init__(self, config: dict):
        # `runtime_config.placement()` owns the defaults; duplicating them
        # here would give the same knob two values that can drift.
        config = dict(config)
        self.enabled = bool(config["enabled"])
        self.max_queue_depth = max(0, int(config["max_queue_depth"]))
        self.max_p95_ms = float(config["max_p95_ms"])
        self.latency = LocalLatency(config["sample_window"],
                                    config["ewma_alpha"],
                                    config["p95_ttl_ms"])
        self._decisions = collections.deque(
            maxlen=max(1, int(config["decision_log_size"])))
        # True after a default session had to go remote for a local problem.
        # The first later clean decision is the recovery evidence operators
        # need to see; all subsequent ones are ordinary local-first decisions.
        self._degraded = False
        self._lock = threading.Lock()

    def record_latency(self, elapsed_ms: float) -> None:
        self.latency.record(elapsed_ms)

    def reset_latency(self) -> None:
        self.latency.reset()

    def decide(self, session_id: str, local_healthy: bool,
               local_available: bool, queue_depth: int, remote,
               requested: str | None = None) -> dict:
        """Choose a target for a *new* session and log the reason."""
        p95 = self.latency.p95_ms
        queue_depth = max(0, int(queue_depth))
        remote_configured = remote is not None
        remote_healthy = None
        remote_control_suitable = None
        local_issues = []

        if requested is not None:
            target, reason = requested, "explicit"
        elif not self.enabled:
            target, reason = LOCAL_TARGET, "placement_disabled"
        else:
            if not local_available:
                local_issues.append("local_unhealthy")
            if queue_depth > self.max_queue_depth:
                local_issues.append("queue_depth")
            if p95 is not None and p95 > self.max_p95_ms:
                local_issues.append("p95")

            if not local_issues:
                with self._lock:
                    reason = "local_recovered" if self._degraded else "local_first"
                    self._degraded = False
                target = LOCAL_TARGET
            else:
                if remote_configured:
                    remote_healthy = bool(remote.healthy())
                    if remote_healthy:
                        remote_control_suitable = remote.control_suitable()
                if remote_healthy and remote_control_suitable is not False:
                    with self._lock:
                        self._degraded = True
                    target, reason = REMOTE_TARGET, local_issues[0]
                else:
                    target = LOCAL_TARGET
                    if remote_healthy and remote_control_suitable is False:
                        reason = "remote_unsuitable"
                    elif remote_configured:
                        reason = "remote_unavailable"
                    else:
                        reason = "remote_not_configured"

        decision = {
            "session_id": session_id,
            "placement_decision": target,
            "target": target,
            "reason": reason,
            "queue_depth": queue_depth,
            "p95": p95,
            "p95_ms": p95,
            "local_healthy": bool(local_healthy),
            "local_available": bool(local_available),
            "remote_configured": remote_configured,
            "remote_healthy": remote_healthy,
            "remote_control_suitable": remote_control_suitable,
            "local_issues": local_issues,
        }
        with self._lock:
            self._decisions.append(decision)
        return decision

    def snapshot(self, local_healthy: bool, local_available: bool,
                 queue_depth: int) -> dict:
        with self._lock:
            decisions = [dict(decision) for decision in self._decisions]
            counts = {LOCAL_TARGET: 0, REMOTE_TARGET: 0}
            for decision in decisions:
                counts[decision["target"]] += 1
        p95 = self.latency.p95_ms
        return {
            "policy": POLICY,
            "enabled": self.enabled,
            "thresholds": {
                "max_queue_depth": self.max_queue_depth,
                "max_p95_ms": self.max_p95_ms,
                "p95_ttl_ms": self.latency.ttl_ms,
            },
            "local": {
                "healthy": bool(local_healthy),
                "available": bool(local_available),
                "queue_depth": max(0, int(queue_depth)),
                "p95": p95,
                "p95_ms": p95,
            },
            "counts": counts,
            "decisions": decisions,
        }
