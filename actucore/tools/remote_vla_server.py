#!/usr/bin/env python3
"""Run the laptop's `motus.vla/1` test server with the robot's checkpoint.

This is the server end of the `remote_http` backend: the same `RuntimeGateway`
the robot runs, on a laptop RTX 4060, loading the checkpoint settings from the
same ActuCore config. It is a test/overflow target, not a production serving
service.

On the laptop:

    python actucore/tools/remote_vla_server.py --config actucore/config.yaml \
        --host 0.0.0.0 --port 15732

On the robot, point the runtime at it:

    runtime:
      remote:
        endpoint: http://<laptop-ip>:15732
        max_obs_age_ms: 300    # mirror the driver descriptor
        watchdog_ms: 200       # mirror the driver descriptor

The laptop needs PyTorch + LeRobot and the same checkpoint files as the robot.
`--model-dir` overrides the config when the laptop stages them elsewhere.
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import sys
import threading

import yaml

sys.path.insert(0, __file__.rsplit("/", 2)[0])
# A repo checkout has the downloader in perception/utils; the ActuCore image
# copies those two modules beside main.py instead.
_perception_utils = (pathlib.Path(__file__).resolve().parents[2]
                     / "perception" / "utils")
if _perception_utils.is_dir():
    sys.path.insert(0, str(_perception_utils))

from runtime import RuntimeGateway, worker_config  # noqa: E402


def server_config(config: dict, host: str = "0.0.0.0", port: int = 15732,
                  model_name: str = "", model_dir: str = "",
                  device: str = "") -> dict:
    """The gateway config for a standalone SmolVLA server.

    The checkpoint keys come through `worker_config`, which is the same path
    the robot's local worker uses. That keeps one source of truth: the server
    and the local worker load the same `model_name` / `models` / `feature_map`
    unless the operator explicitly overrides one here.
    """
    runtime = dict(config.get("runtime") or {})
    vla = dict((config.get("plugins") or {}).get("vla") or {})
    runtime.pop("remote", None)      # a server does not become a client
    runtime["backend"] = "smolvla"
    result = worker_config(runtime, vla)
    smolvla = dict(result.get("smolvla") or {})
    if model_name:
        smolvla["model_name"] = model_name
    if model_dir:
        smolvla["model_dir"] = model_dir
        # A `models:` entry wins over the top-level model_dir in the provider,
        # so an explicit path must replace it there too or it would be ignored.
        models = dict(smolvla.get("models") or {})
        selected = smolvla.get("model_name") or ""
        if selected in models:
            entry = dict(models[selected] or {})
            entry["model_dir"] = model_dir
            models[selected] = entry
            smolvla["models"] = models
    if device:
        smolvla["device"] = device
    result.update({
        "enabled": True,
        "host": host,
        "port": port,
        "backend": "smolvla",
        "smolvla": smolvla,
        # A test server stays up between sessions; the card's idle reap belongs
        # on the robot's gateway, not on the laptop.
        "idle_reap_ms": 0,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = (pathlib.Path(__file__).resolve().parents[1]
                      / "config.yaml")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=15732)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    gateway_config = server_config(config, host=args.host, port=args.port,
                                   model_name=args.model_name,
                                   model_dir=args.model_dir,
                                   device=args.device)
    gateway = RuntimeGateway(gateway_config)
    gateway.start()
    checkpoint = gateway_config["smolvla"].get("model_name") or "config"
    print(f"[remote-vla] motus.vla/1 listening on "
          f"http://{args.host}:{args.port} (checkpoint={checkpoint})",
          flush=True)

    stopping = threading.Event()

    def _stop(_signum: int, _frame) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        stopping.wait()
    finally:
        gateway.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
