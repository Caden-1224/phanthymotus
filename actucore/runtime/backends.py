"""Backends that run inside the model worker process.

They are deliberately tiny: the point of this ticket is the process boundary and
the session contract, not another mock policy. `MockProvider` already owns the
sine signal and its safety properties, so the backends reuse it and only add
what a worker needs — the configured capabilities and, for `delay`, latency.
"""

from __future__ import annotations

import time

from plugins.vla.providers.mock import MockProvider


class MockBackend:
    def __init__(self, capabilities: dict, period_s: float = 8.0):
        self._capabilities = dict(capabilities)
        dof = int(self._capabilities["action_dim"])
        descriptor = {
            "dof": dof,
            "mode": self._capabilities.get("control_mode", "joint_position"),
            "limits": {"lower": [-1.0] * dof, "upper": [1.0] * dof},
        }
        self._provider = MockProvider(
            descriptor,
            period_s=float(period_s),
            chunk_size=int(self._capabilities["chunk_size"]),
            control_hz=float(self._capabilities["control_hz"]),
        )

    def capabilities(self) -> dict:
        return dict(self._capabilities)

    def infer(self, request: dict) -> list:
        return self._provider.infer(None, int(request.get("inference_delay") or 0))

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None


class DelayBackend(MockBackend):
    """Mock trajectory plus a fixed delay, for noticing blocking where it hurts."""

    def __init__(self, capabilities: dict, delay_ms: float = 0, period_s: float = 8.0):
        super().__init__(capabilities, period_s=period_s)
        self._delay_s = max(0.0, float(delay_ms)) / 1000.0

    def infer(self, request: dict) -> list:
        if self._delay_s:
            time.sleep(self._delay_s)
        return super().infer(request)
