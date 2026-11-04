"""The local SmolVLA backend through the runtime gateway.

A fake checkpoint and a fake `lerobot`/`torch` stack stand in for the GPU
image, so the real `SmolVLAProvider` still owns the checkpoint config,
observation mapping and action chunk. What must be identical here is the
*contract*: the native provider and the gateway must answer the same
capabilities and the same chunk for the same checkpoint.

These are not a substitute for hardware acceptance. The real checkpoint test
at the bottom is skipped unless `ACTUCORE_SMOLVLA_CHECKPOINT` is set on a
machine with lerobot and a GPU.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_runtime_smolvla.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import sys
import time
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.vla.providers.smolvla import SmolVLAProvider  # noqa: E402
from plugins.vla.providers.vla_cloud import SCHEMA, VLACloudProvider  # noqa: E402
from runtime import RuntimeGateway  # noqa: E402

CAPABILITY_FIELDS = ("model", "control_mode", "action_dim", "chunk_size",
                     "control_hz", "needs_state", "n_cameras", "image_size",
                     "supports_rtc")

CHECKPOINT_CONFIG = {
    "type": "smolvla",
    "fps": 30,
    "n_action_steps": 5,
    "input_features": {
        "observation.images.cam": {"type": "VISUAL", "shape": [3, 8, 8]},
        "observation.state": {"type": "STATE", "shape": [6]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}

TORCH = '''
class Tensor:
    def __init__(self, value):
        self.value = value

    @property
    def ndim(self):
        depth, value = 0, self.value
        while isinstance(value, list):
            depth += 1
            value = value[0] if value else None
        return depth

    def to(self, _device):
        return self

    def detach(self):
        return self

    def tolist(self):
        return self.value

    def __getitem__(self, index):
        return Tensor(self.value[index])

    def unsqueeze(self, _dim):
        return Tensor([self.value])


def as_tensor(value):
    return Tensor(value)


class no_grad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def empty_cache():
        pass
'''

FACTORY = '''
import os
import time


def make_pre_post_processors(config, pretrained_path=None, preprocessor_overrides=None):
    time.sleep(float(os.environ.get("FAKE_SMOLVLA_LOAD_S", "0")))
    return (lambda batch: batch, lambda chunk: chunk)
'''

MODEL = '''
import os

import torch


class SmolVLAPolicy:
    def __init__(self):
        self.config = {}

    @classmethod
    def from_pretrained(cls, source):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_action_chunk(self, batch):
        state = batch.get("observation.state")
        values = state.value if hasattr(state, "value") else [0.0] * 6
        steps = int(os.environ.get("FAKE_SMOLVLA_STEPS", "5"))
        # Fold the rest of the observation contract into the output so a lost
        # image mapping or prompt cannot still pass the parity test.
        bias = 0.01 * len(batch.get("task") or "")
        bias += 0.1 * sum(1 for key in batch if "image" in key)
        return torch.Tensor(
            [[[float(v) + bias + 0.01 * step for v in values]
              for step in range(steps)]])
'''


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def fake_stack(tmp_path_factory, monkeypatch):
    """A tiny torch/lerobot on sys.path, inherited by the spawned worker."""
    root = tmp_path_factory.mktemp("fake_smolvla_stack")
    _write(root / "torch" / "__init__.py", TORCH)
    _write(root / "lerobot" / "__init__.py", "")
    _write(root / "lerobot" / "policies" / "__init__.py", "")
    _write(root / "lerobot" / "policies" / "factory.py", FACTORY)
    _write(root / "lerobot" / "policies" / "smolvla" / "__init__.py", "")
    _write(root / "lerobot" / "policies" / "smolvla" / "modeling_smolvla.py", MODEL)
    _write(root / "lerobot" / "processor" / "__init__.py", "")
    _write(root / "lerobot" / "processor" / "normalize_processor.py",
           "def to_tensor(stats, device=None, dtype=None):\n    return stats\n")
    saved = {name: module for name, module in sys.modules.items()
             if name == "torch" or name == "lerobot" or name.startswith("lerobot.")}
    monkeypatch.syspath_prepend(str(root))
    try:
        yield root
    finally:
        # A fake left in sys.modules is inherited by every later test module;
        # this repo has paid for that once already (see fake_torch below).
        for name in list(sys.modules):
            if name == "torch" or name == "lerobot" or name.startswith("lerobot."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "smolvla_fake"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(CHECKPOINT_CONFIG))
    return directory


def smolvla_config(checkpoint, **overrides):
    config = {
        "model_name": "smolvla_fake",
        "model_dir": str(checkpoint),
        "device": "cpu",
        "feature_map": {"main": "observation.images.cam",
                        "state": "observation.state"},
    }
    config.update(overrides)
    return config


def make_gateway(checkpoint, **overrides):
    config = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
        "backend": "smolvla",
        "smolvla": smolvla_config(checkpoint),
        "monitor_interval_ms": 25,
        "idle_reap_ms": 0,
    }
    config.update(overrides)
    gateway = RuntimeGateway(config)
    gateway.start()
    return gateway


def observation(state=None):
    return types.SimpleNamespace(
        images={"main": b"\xff\xd8fake-jpeg"},
        state=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] if state is None else state,
        prompt="pick it up",
        t_capture_ms=1_700_000_000_000,
    )


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _request(url, payload=None):
    import urllib.request

    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body,
                                     method="POST" if body else "GET",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def worker_status(gateway):
    return _request(gateway.url + "/status")[1]["worker"]


def test_runtime_smolvla_reuses_the_cards_checkpoint_settings():
    from runtime import worker_config

    runtime = worker_config(
        {"enabled": True, "backend": "smolvla"},
        {"model_name": "smolvla_base", "models": {"smolvla_base": {"model_dir": "/m"}},
         "feature_map": {"main": "observation.images.camera1"}, "device": "cuda"})

    assert runtime["smolvla"]["model_name"] == "smolvla_base"
    assert runtime["smolvla"]["models"] == {"smolvla_base": {"model_dir": "/m"}}
    assert runtime["smolvla"]["feature_map"] == {"main": "observation.images.camera1"}


def test_an_explicit_runtime_checkpoint_overrides_the_card():
    """Same checkpoint by default, but a deliberate A/B must still be possible."""
    from runtime import worker_config

    runtime = worker_config(
        {"enabled": True, "backend": "smolvla",
         "smolvla": {"model_name": "smolvla_tianyi"}},
        {"model_name": "smolvla_base"})

    assert runtime["smolvla"]["model_name"] == "smolvla_tianyi"


def test_native_and_runtime_capabilities_agree(fake_stack, checkpoint):
    native = SmolVLAProvider({}, smolvla_config(checkpoint))
    assert wait_for(native.health)
    expected = native.capabilities()

    gateway = make_gateway(checkpoint)
    try:
        through_gateway = VLACloudProvider({}, {"endpoint": gateway.url}).capabilities()

        assert through_gateway["schema"] == SCHEMA
        assert {key: through_gateway[key] for key in CAPABILITY_FIELDS} == \
               {key: expected[key] for key in CAPABILITY_FIELDS}
    finally:
        gateway.stop()
        native.close()


def test_runtime_chunk_matches_the_native_provider(fake_stack, checkpoint):
    native = SmolVLAProvider({}, smolvla_config(checkpoint))
    assert wait_for(native.health)
    expected = native.infer(observation())

    gateway = make_gateway(checkpoint)
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        assert wait_for(provider.health)
        actions = provider.infer(observation())

        assert actions == expected
        assert len(actions) == 5 and all(len(step) == 6 for step in actions)
        assert provider.capabilities()["control_mode"] == "joint_position"
    finally:
        gateway.stop()
        native.close()


def test_health_is_false_while_weights_load_and_the_card_shows_loading(
        fake_stack, checkpoint, monkeypatch):
    from plugins.vla import VLAPlugin

    monkeypatch.setenv("FAKE_SMOLVLA_LOAD_S", "0.6")
    gateway = make_gateway(checkpoint)
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        card = VLAPlugin({"provider": "vla_cloud", "endpoint": gateway.url},
                         executor=None)
        card._provider = provider
        card._running = True

        assert provider.health() is False
        assert card._state() == "loading"
        status, body = _request(gateway.url + "/healthz")
        assert status == 503 and body["status"] == "loading"

        status, body = _request(gateway.url + "/infer",
                                {"schema": SCHEMA, "session_id": "loading", "seq": 1})
        assert status == 503 and body["error"]["code"] == "loading"

        assert wait_for(provider.health, timeout=5)
        assert card._state() == "running"
        assert provider.infer(observation())
    finally:
        gateway.stop()


def test_a_reaped_worker_wakes_without_leaking_a_loading_error(
        fake_stack, checkpoint):
    gateway = make_gateway(checkpoint, idle_reap_ms=100)
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        assert wait_for(provider.health)
        assert wait_for(lambda: worker_status(gateway)["state"] == "stopped")

        status, body = _request(gateway.url + "/infer",
                                {"schema": SCHEMA, "session_id": "wake", "seq": 1})
        assert status == 503 and body["error"]["code"] == "loading"

        assert wait_for(provider.health)
        assert provider.infer(observation())
    finally:
        gateway.stop()


def test_kill_worker_five_times_recovers_and_records_resident_memory(
        fake_stack, checkpoint):
    gateway = make_gateway(checkpoint)
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        assert wait_for(provider.health)
        assert provider.infer(observation())
        rss = []

        for count in range(1, 6):
            pid = gateway.worker.pid
            assert pid
            os.kill(pid, signal.SIGKILL)

            def restarted(count=count):
                worker = worker_status(gateway)
                return (worker["state"] == "ready" and worker["healthy"]
                        and worker["restart_count"] == count)

            assert wait_for(restarted), f"restart {count}/5 did not become ready"
            worker = worker_status(gateway)
            assert worker["pid"] != pid
            assert worker["rss_mb"] and worker["rss_mb"] > 0
            rss.append(worker["rss_mb"])
            assert provider.infer(observation())

        assert len(rss) == 5
    finally:
        gateway.stop()


REAL_CHECKPOINT = os.environ.get("ACTUCORE_SMOLVLA_CHECKPOINT", "")


@pytest.mark.skipif(not REAL_CHECKPOINT,
                    reason="hardware acceptance: set ACTUCORE_SMOLVLA_CHECKPOINT "
                           "on a machine with lerobot to load the real weights")
def test_the_real_checkpoint_through_the_gateway_matches_native():
    config = {"model_dir": REAL_CHECKPOINT,
              "device": os.environ.get("ACTUCORE_SMOLVLA_DEVICE", "cuda")}
    # Capabilities come from the checkpoint config; read them before the GPU
    # load so this comparison does not put two copies of the model on the GPU.
    native = SmolVLAProvider({}, config)
    expected = native.capabilities()
    native.close()

    gateway = RuntimeGateway({
        "enabled": True, "host": "127.0.0.1", "port": 0, "backend": "smolvla",
        "smolvla": config, "idle_reap_ms": 0,
    })
    gateway.start()
    try:
        provider = VLACloudProvider({}, {"endpoint": gateway.url})
        assert wait_for(provider.health, timeout=120)
        through_gateway = provider.capabilities()
        assert {key: through_gateway[key] for key in CAPABILITY_FIELDS} == \
               {key: expected[key] for key in CAPABILITY_FIELDS}
    finally:
        gateway.stop()
