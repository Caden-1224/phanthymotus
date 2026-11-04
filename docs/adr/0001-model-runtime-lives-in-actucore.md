# Model runtime lives inside ActuCore for v1

**Status**: proposed

The v1 model runtime is implemented under `actucore/runtime/` and deployed in the existing ActuCore container; model inference runs in a separate worker process. A standalone `runtime/` service would add another image, port, service fragment, health endpoint and deployment surface, and would overlap ModelHub's on-robot serving role. The process boundary that matters for failure isolation is between the gateway and the model worker, not between the gateway and ActuCore.

**Consequences**: perception cannot reuse the runtime without extracting it later; that extraction is a separate deployment decision.

**Considered options**:
- Top-level `runtime/` service: stronger platform story, but too much deployment work for v1 and a boundary conflict with ModelHub.
- ModelHub VLA engine: wrong deployment target today (separate repo, Thor/JP7.2), and ModelHub is not the execution layer.
