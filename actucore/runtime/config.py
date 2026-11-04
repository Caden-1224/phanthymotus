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


def host(config: dict) -> str:
    value = str((config or {}).get("host") or "127.0.0.1")
    return "127.0.0.1" if value in ("", "0.0.0.0") else value


def port(config: dict) -> int:
    return int((config or {}).get("port") or DEFAULT_PORT)


def base_url(config: dict) -> str:
    return f"http://{host(config)}:{port(config)}"


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
