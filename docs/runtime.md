# Runtime

## Data types

RuntimeSample, ControllerStatus

## Classes

### Runtime

```python
get_ratio_handle() -> RatioHandle
get_ratio() -> float
set_ratio(value, stream=None) -> None
observe(sample) -> None
before_batch(stream=None) -> float
set_manual_ratio(value) -> None
clear_manual_ratio() -> None
status() -> ControllerStatus
close() -> None
```

### RatioHandle

```python
tensor -> torch.Tensor
```

### RatioController

```python
observe(sample) -> None
set_manual_ratio(value) -> None
clear_manual_ratio() -> None
before_batch(stream=None) -> float
status() -> ControllerStatus
```

### RatioPolicy (Protocol)

```python
select(sample, current_ratio) -> float | None
```

### BandwidthBalancePolicy

```python
select(sample, current_ratio) -> float | None
```

### RuntimeObserver

```python
start(device) -> None
close() -> None
observe(sample) -> None
snapshot(now_ns=None) -> RuntimeSample
```

## Functions

### ratio_validation.py

```python
validate_ratio(value) -> float
```

## Relations

```mermaid
%%{init: {'class': {'hideEmptyMembersBox': true}}}%%
classDiagram
    direction LR

    class Runtime
    class RatioHandle
    class RatioController
    class BandwidthBalancePolicy {
        <<RatioPolicy>>
    }
    class RuntimeObserver

    class SGLangModelRunner {
        <<external>>
    }
    class SGLangHybridGemmWrapper {
        <<external>>
    }
    class NVML {
        <<external>>
    }

    SGLangModelRunner --> Runtime : creates one per worker, calls before_batch outside CUDA graph capture
    SGLangHybridGemmWrapper --> RatioHandle : holds, reads tensor

    Runtime --> RatioHandle : creates and holds
    Runtime --> RatioController : creates and holds
    Runtime --> RuntimeObserver : creates and holds
    Runtime ..> BandwidthBalancePolicy : creates

    RatioHandle ..> Runtime : weak reference, reads the ratio tensor

    RatioController --> Runtime : holds, calls get_ratio / set_ratio
    RatioController --> RuntimeObserver : holds, calls snapshot
    RatioController --> BandwidthBalancePolicy : holds, calls select

    RuntimeObserver ..> NVML : background thread reads GPM every second, HBM utilization of this MIG and traffic on the shared host link; link capacity comes from NVML
```
