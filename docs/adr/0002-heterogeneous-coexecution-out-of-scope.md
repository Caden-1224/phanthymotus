# Heterogeneous co-execution is out of scope for v1

**Status**: proposed

V1 targets multiple execution targets and process isolation; it does not split one model across CPU + GPU/NPU/DLA. SmolVLA is a transformer/flow-matching policy, Orin's DLA is CNN-oriented, and intra-model partitioning would require model surgery, numerical parity work and a real checkpoint. It would turn the project into a model deployment/compiler effort rather than a runtime engineering deliverable.

**Consequences**: the runtime must not claim "heterogeneous co-execution" or "heterogeneous acceleration". The accepted story is heterogeneous execution targets plus local-first placement, not simultaneous use of multiple compute units.
