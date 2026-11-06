# PhanthyMotus

PhanthyMotus is a robot agent platform built around a sense → think → act loop. This glossary covers the execution-model runtime used by ActuCore.

## Language

**Model runtime**:
The ActuCore subsystem that owns model worker processes and routes model inference. It is not ModelHub: ModelHub serves the agent loop's LLM/VLM, while the model runtime executes execution models such as VLA policies.
_Avoid_: ModelHub, inference server

**Model worker**:
An independent process that hosts one model instance and executes inference requests. It has no ROS/DDS responsibilities.
_Avoid_: model process, provider

**Runtime gateway**:
The local entry point that speaks `motus.vla/1` to the VLA card and routes each request to a model worker or a remote execution target.
_Avoid_: proxy

**Execution target**:
A place where a model worker runs: the local Jetson GPU or a remote GPU endpoint.
_Avoid_: backend (a backend is the implementation used on a target)

**Backend**:
A concrete model execution implementation, such as `torch_cuda`, `remote_http`, or `mock`, used on an execution target.
_Avoid_: provider (the VLA card's provider is the card-facing adapter)

**Session**:
One VLA card lifetime from start to stop. `execute` changes the task but not the session; placement and failover are scoped to it.
_Avoid_: task, request

**Placement**:
The choice of execution target for a session. v1 is local-first: local GPU by default, remote only for overflow or failover.
_Avoid_: load balancing (v1 does not balance per chunk)

**Failover**:
An exceptional mid-session switch from the local target to the remote target after the local worker becomes unhealthy.
_Avoid_: fallback (fallback may mean any target choice)

**Failback**:
A new session returning to the local target after a degraded period in which local was unavailable or overloaded. Existing failed-over sessions are not moved back.
_Avoid_: failover (failover moves an existing session; failback affects future sessions)

**Action chunk**:
The sequence of future actions returned by one model inference; the VLA card paces it out as individual commands.
_Avoid_: trajectory, plan

**Heterogeneous execution targets**:
Multiple execution targets with different hardware, such as a Jetson GPU and an x86 RTX laptop. This is not simultaneous use of multiple compute units.
_Avoid_: heterogeneous computing

**Heterogeneous co-execution**:
One model or task using multiple compute units simultaneously, such as CPU + GPU/NPU/DLA. Out of scope for v1.
_Avoid_: heterogeneous execution (ambiguous)
