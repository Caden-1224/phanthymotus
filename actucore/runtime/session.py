"""Session state for one VLA card lifetime.

The state machine is the contract from the proposal:

    new → ready → busy → ready
                 ↘ cancelling → ready
    any state → terminated

Only one inference may be in flight per session. That is enforced here rather
than in the HTTP handler so every caller sees the same rule.
"""

from __future__ import annotations

import threading
from enum import Enum

LOCAL_TARGET = "local"
REMOTE_TARGET = "remote"
TARGETS = (LOCAL_TARGET, REMOTE_TARGET)


class SessionState(str, Enum):
    NEW = "new"
    READY = "ready"
    BUSY = "busy"
    CANCELLING = "cancelling"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


class BusyError(RuntimeError):
    """Another inference already owns this session."""


class ClosedError(RuntimeError):
    """The session has been closed; it cannot accept more work."""


class Session:
    def __init__(self, session_id: str, target: str = LOCAL_TARGET,
                 placement: dict | None = None):
        self.session_id = session_id
        # A session is pinned to one machine per chunk for its whole life.
        # The only permitted move is local→remote after a local health failure,
        # once, and it never moves back: switching per chunk or mixing action
        # chunks from two models would break the arm's continuity.
        self.target = target
        self._placement = dict(placement or {})
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state = SessionState.NEW
        self._closed = False
        self._in_flight = False
        self._inference_count = 0
        self._last_error = ""
        # One local→remote move at most, and never back: these counters are
        # the session's own evidence that it stayed on one machine per chunk.
        self._failover_reason = ""
        self._failover_count = 0
        self._switch_count = 0
        self._target_history = [target]

    def begin_infer(self) -> None:
        with self._lock:
            if self._closed:
                raise ClosedError(f"session {self.session_id} is closed")
            if self._in_flight:
                raise BusyError(f"session {self.session_id} already has an inference")
            if self._state is SessionState.NEW:
                self._state = SessionState.READY
            # A cancel applies to the chunk it interrupted, not to the session's
            # whole life: the next inference starts with a clean token.
            self._cancel.clear()
            self._in_flight = True
            self._state = SessionState.BUSY

    def finish_infer(self, error: str = "") -> None:
        with self._lock:
            if self._in_flight:
                self._inference_count += 1
            self._in_flight = False
            if error:
                self._last_error = error
            if self._closed:
                self._state = SessionState.TERMINATED
            elif self._state in (SessionState.BUSY, SessionState.CANCELLING):
                self._state = SessionState.READY

    def mark_recovering(self, error: str) -> None:
        """The worker is gone; hold the session until a replacement is ready."""
        with self._lock:
            if self._in_flight:
                self._inference_count += 1
            self._in_flight = False
            self._last_error = error
            if self._closed:
                self._state = SessionState.TERMINATED
            else:
                self._state = SessionState.RECOVERING

    def mark_ready(self) -> None:
        with self._lock:
            if self._state is SessionState.RECOVERING:
                self._state = SessionState.READY

    def mark_failover_pending(self, reason: str) -> None:
        """Remember a local chunk failure so the *next* chunk can move.

        The target is not changed here: the failed inference still owns the
        chunk it interrupted, and moving it while that reply is in flight is
        exactly the cross-target mix this state machine forbids.
        """
        with self._lock:
            if (self.target == LOCAL_TARGET and not self._failover_count
                    and not self._closed):
                self._failover_reason = reason

    def failover(self, reason: str) -> bool:
        """Move this session to remote, once, at a chunk boundary."""
        with self._lock:
            if (self.target != LOCAL_TARGET or self._failover_count
                    or self._closed or self._in_flight):
                return False
            self.target = REMOTE_TARGET
            self._target_history.append(REMOTE_TARGET)
            self._failover_count = 1
            self._switch_count += 1
            self._failover_reason = reason or "local_unhealthy"
            if self._state is SessionState.RECOVERING:
                self._state = SessionState.READY
            return True

    @property
    def failover_reason(self) -> str:
        with self._lock:
            return self._failover_reason

    @property
    def failover_count(self) -> int:
        with self._lock:
            return self._failover_count

    @property
    def switch_count(self) -> int:
        with self._lock:
            return self._switch_count

    def cancel(self) -> None:
        """Cooperative cancel: the handler discards the reply when it arrives."""
        with self._lock:
            self._cancel.set()
            if self._in_flight:
                self._state = SessionState.CANCELLING
            elif not self._closed:
                self._state = SessionState.READY

    def close(self) -> None:
        """Close now, or after the in-flight inference acknowledges cancel."""
        with self._lock:
            self._closed = True
            self._cancel.set()
            if self._in_flight:
                self._state = SessionState.CANCELLING
            else:
                self._state = SessionState.TERMINATED

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._in_flight

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def snapshot(self) -> dict:
        with self._lock:
            if self._failover_count:
                placement_decision = self.target
                placement_reason = self._failover_reason
            else:
                placement_decision = self._placement.get(
                    "placement_decision", self.target)
                placement_reason = self._placement.get(
                    "reason", "explicit" if self.target != LOCAL_TARGET
                    else "local_first")
            return {
                "session_id": self.session_id,
                "state": self._state.value,
                "target": self.target,
                "placement_decision": placement_decision,
                "placement_reason": placement_reason,
                "queue_depth": self._placement.get("queue_depth", 0),
                "p95": self._placement.get("p95_ms"),
                "p95_ms": self._placement.get("p95_ms"),
                "in_flight": self._in_flight,
                "inference_count": self._inference_count,
                "last_error": self._last_error,
                "failover_count": self._failover_count,
                "switch_count": self._switch_count,
                "failover_reason": self._failover_reason,
                "target_history": list(self._target_history),
            }


class SessionRegistry:
    """Every session the gateway has seen, keyed by the provider's session id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str, target: str = LOCAL_TARGET,
                      placement: dict | None = None) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id, target, placement)
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.close()
        return session

    def close_all(self) -> list[str]:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.close()
        return [session.session_id for session in sessions]

    def snapshot(self) -> list:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.snapshot() for session in sessions]

    def mark_all_recovering(self, error: str,
                            target: str | None = None,
                            failover_reason: str = "") -> None:
        for session in list(self._sessions.values()):
            if target is None or session.target == target:
                if failover_reason:
                    session.mark_failover_pending(failover_reason)
                session.mark_recovering(error)

    def mark_all_ready(self, target: str | None = None) -> None:
        for session in list(self._sessions.values()):
            if target is None or session.target == target:
                session.mark_ready()
