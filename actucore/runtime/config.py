"""Runtime configuration and the card-facing switch.

Enabling the runtime changes only the card's provider configuration: it points
`vla_cloud` at this process. Disabling it leaves the card exactly as it was, so
rollback is one YAML value rather than a redeploy or a card change.
"""

from __future__ import annotations

SCHEMA = "motus.vla/1"
DEFAULT_PORT = 15732

DEFAULT_CAPABILITIES = {
    "model": "mock-sine@8s",
    "control_mode": "joint_position",
    "action_dim": 7,
    "chunk_size": 10,
    "control_hz": 30,
    "needs_state": False,
    "n_cameras": 0,
    "image_size": 0,
    "supports_rtc": False,
}


def enabled(config: dict) -> bool:
    return bool((config or {}).get("enabled", False))


def capabilities(config: dict) -> dict:
    merged = dict(DEFAULT_CAPABILITIES)
    merged.update(dict((config or {}).get("capabilities") or {}))
    merged["schema"] = SCHEMA
    return merged


def client_host(config: dict) -> str:
    """The address the card should use to reach this gateway."""
    value = str((config or {}).get("host") or "127.0.0.1")
    return "127.0.0.1" if value in ("", "0.0.0.0") else value


def bind_host(config: dict) -> str:
    """The address the gateway binds. A test server may deliberately bind all
    interfaces; the local gateway's default stays loopback."""
    return str((config or {}).get("host") or "127.0.0.1")


def port(config: dict) -> int:
    value = (config or {}).get("port")
    return DEFAULT_PORT if value in (None, "") else int(value)


def base_url(config: dict) -> str:
    return f"http://{client_host(config)}:{port(config)}"


PLACEMENT_DEFAULTS = {
    # Local-first is the product default; thresholds only decide when a *new*
    # session overflows to remote. A session's target never changes after that.
    "enabled": True,
    "max_queue_depth": 1,
    "max_p95_ms": 250.0,
    "sample_window": 50,
    "ewma_alpha": 0.3,
    # A p95 older than this is treated as unknown so a recovered local worker
    # gets a new session instead of being pinned behind an old overload.
    "p95_ttl_ms": 30000.0,
    "decision_log_size": 1000,
}


def placement(config: dict) -> dict:
    merged = dict(PLACEMENT_DEFAULTS)
    merged.update(dict((config or {}).get("placement") or {}))
    return merged


def remote(config: dict) -> dict:
    """The remote execution target, or {} when none is configured."""
    value = dict((config or {}).get("remote") or {})
    endpoint = str(value.get("endpoint") or "").strip()
    if not endpoint:
        return {}
    backend = str(value.get("backend") or "remote_http").strip()
    if backend != "remote_http":
        raise ValueError(
            f"runtime.remote.backend must be 'remote_http', not {backend!r}")
    value["endpoint"] = endpoint
    value["backend"] = backend
    return value


def vla_config(vla: dict, runtime: dict) -> dict:
    """The VLA card config, routed through the local gateway when runtime is on."""
    result = dict(vla or {})
    if not enabled(runtime):
        return result
    result["provider"] = "vla_cloud"
    result["endpoint"] = base_url(runtime)
    result["api_key"] = str((runtime or {}).get("api_key") or "")
    model = str(capabilities(runtime).get("model") or "")
    if model:
        result["cloud_model_name"] = model
    return result


# The checkpoint settings the card's `smolvla` provider reads. The worker must
# load the same ones, or "native vs runtime" compares two different models.
SMOLVLA_SETTINGS = ("model_name", "models", "weights", "model_dir", "feature_map",
                    "unnorm_key", "vlm_dir", "vlm_weights", "device")


def worker_config(runtime: dict, vla: dict) -> dict:
    """Runtime config with the card's SmolVLA checkpoint settings by default.

    The card config is the one operators already set up and verify. Copying
    those keys into the worker section removes the second, hand-maintained
    checkpoint identity — the thing that could silently make the runtime load
    different weights from the native provider. Anything explicitly set under
    `runtime.smolvla` still wins, so a deliberate A/B can override it.
    """
    result = dict(runtime or {})
    if str(result.get("backend") or "") != "smolvla":
        return result
    settings = dict(result.get("smolvla") or {})
    for key in SMOLVLA_SETTINGS:
        if key not in settings and key in (vla or {}):
            settings[key] = vla[key]
    result["smolvla"] = settings
    return result
