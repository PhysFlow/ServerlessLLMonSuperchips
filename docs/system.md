# System

## Components

| Component | Process | Document |
|---|---|---|
| control_plane | one process per machine | [control_plane.md](control_plane.md) |
| sglang | one worker process per MIG instance, customized PhysFlow/sglang | [sglang.md](sglang.md) |
| runtime | inside every sglang worker process | [runtime.md](runtime.md) |
| hybrid_gemm | inside every sglang worker process | not open source yet |

## Relations

```mermaid
flowchart TB
    Client["Client"]
    ControlPlane["control_plane"]
    SGLang["sglang, one worker per MIG"]
    Runtime["runtime"]
    HybridGemm["hybrid_gemm"]
    GPU["H800 MIG instance"]
    Host["pinned CPU memory"]

    Client -->|OpenAI requests| ControlPlane
    ControlPlane -->|starts one worker per MIG, places models, forwards requests over HTTP| SGLang
    SGLang -->|calls before_batch outside CUDA graph capture| Runtime
    Runtime -->|ratio tensor| SGLang
    SGLang -->|linear layers call it with the ratio tensor| HybridGemm
    Runtime -->|reads HBM and host link utilization through NVML| GPU
    HybridGemm -->|computes on| GPU
    HybridGemm -->|reads weights directly| Host
```
