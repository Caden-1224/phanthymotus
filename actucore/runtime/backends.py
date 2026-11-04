"""Backends that run inside the model worker process.

`MockBackend`/`DelayBackend` are deliberately tiny: the point of the runtime
skeleton is the process boundary and the session contract, not another mock
policy. `MockProvider` already owns the sine signal and its safety properties,
so they reuse it and only add what a worker needs — the configured capabilities
and, for `delay`, latency.

`SmolVLABackend` is the real one. It wraps the card's existing `SmolVLAProvider`
instead of reimplementing weights, feature mapping or un-normalisation, so the
native provider and the runtime worker execute the same checkpoint the same way.
"""

from __future__ import annotations

import base64
import time
import types

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


class SmolVLABackend:
    """The card's real SmolVLA provider, inside the model worker process.

    The provider reads the checkpoint config in its constructor, then loads the
    weight tensors on a background thread. On a robot whose checkpoint is
    already staged, hello answers capabilities immediately while `/healthz`
    stays false until the weights are in. A cold download still happens before
    hello (the provider owns that path); see `shortcut:` below.
    """

    def __init__(self, config: dict):
        from plugins.vla.providers.smolvla import SmolVLAProvider

        # shortcut: a first-boot download runs before hello and can exceed the
        # supervisor's 30s start timeout; checkpoint staging is expected to
        # happen before the runtime takes over. Upgrade by moving the provider's
        # staging step behind a readiness protocol if cold boot must be supported.
        self._provider = SmolVLAProvider({}, dict(config.get("smolvla") or {}))

    def capabilities(self) -> dict:
        return self._provider.capabilities()

    def infer(self, request: dict) -> list:
        return self._provider.infer(
            _observation(request), int(request.get("inference_delay") or 0))

    def health(self) -> bool:
        return self._provider.health()

    def close(self) -> None:
        self._provider.close()


def _observation(request: dict) -> types.SimpleNamespace:
    """The `motus.vla/1` request as the observation object providers already take.

    Images arrive base64-encoded because the card's `vla_cloud` provider is the
    client; decode them back to the same bytes the native provider receives on
    the robot. Everything else is already in the provider's field names.
    """
    def decoded(value: str | bytes) -> bytes:
        if isinstance(value, str):
            return base64.b64decode(value)
        return bytes(value)

    return types.SimpleNamespace(
        images={name: decoded(value)
                for name, value in (request.get("images") or {}).items()},
        state=request.get("state"),
        eef_state=request.get("eef_state"),
        prompt=request.get("prompt") or "",
        t_capture_ms=int(request.get("t_capture_ms") or 0),
    )
