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
        # A session is pinned to its target for its whole life: switching
        # between machines mid-session would mix action chunks computed from
        # different models and clocks. Placement (#8) may choose remote for a
        # new session, never per chunk.
        self.target = target
        self._placement = dict(placement or {})
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._state = SessionState.NEW
        self._closed = False
        self._in_flight = False
        self._inference_count = 0
        self._last_error = ""

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
            return {
                "session_id": self.session_id,
                "state": self._state.value,
                "target": self.target,
                "placement_decision": self._placement.get(
                    "placement_decision", self.target),
                "placement_reason": self._placement.get(
                    "reason", "explicit" if self.target != LOCAL_TARGET
                    else "local_first"),
                "queue_depth": self._placement.get("queue_depth", 0),
                "p95": self._placement.get("p95_ms"),
                "p95_ms": self._placement.get("p95_ms"),
                "in_flight": self._in_flight,
                "inference_count": self._inference_count,
                "last_error": self._last_error,
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
                            target: str | None = None) -> None:
        for session in list(self._sessions.values()):
            if target is None or session.target == target:
                session.mark_recovering(error)

    def mark_all_ready(self, target: str | None = None) -> None:
        for session in list(self._sessions.values()):
            if target is None or session.target == target:
                session.mark_ready()
