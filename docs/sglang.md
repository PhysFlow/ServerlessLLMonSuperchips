# SGLang

MemServe runs a customized SGLang: PhysFlow/sglang, branch memserve, based on upstream SGLang v0.5.12. This document covers only the changes MemServe depends on.

## Data types

ServerArgs

## Classes

### ServerArgs

```python
linear_gemm_backend -> str
hybrid_gemm_asym_ratio -> float
```

### ModelRunner

```python
memserve_runtime -> Runtime | None
forward(forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None, reinit_attn_backend=False, split_forward_count=1) -> ModelRunnerOutput
```

### UnquantizedLinearMethod

```python
create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, **extra_weight_attrs) -> None
process_weights_after_loading(layer) -> None
apply(layer, x, bias=None) -> torch.Tensor
```

## Functions

### layers/hybrid_gemm_wrapper/entrypoint.py

```python
set_ratio_handle(handle) -> None
apply_hybrid_gemm(x, weight, bias=None) -> torch.Tensor
```

### layers/quantization/unquant.py

```python
_allocate_pinned_block(num_bytes) -> torch.Tensor
_empty_pinned_host_tensor(shape, dtype) -> torch.Tensor
```

### model_loader/loader.py

```python
device_loading_context(module, target_device) -> ContextManager[None]
```

## Relations

```mermaid
%%{init: {'class': {'hideEmptyMembersBox': true}}}%%
classDiagram
    direction LR

    class ServerArgs
    class ModelRunner
    class UnquantizedLinearMethod
    class apply_hybrid_gemm {
        <<hybrid_gemm_wrapper>>
    }
    class pinned_blocks {
        <<unquant.py>>
    }
    class device_loading_context {
        <<loader.py>>
    }

    class SGLangWorker {
        <<control_plane>>
    }
    class Runtime {
        <<runtime>>
    }
    class RatioHandle {
        <<runtime>>
    }
    class hybrid_gemm {
        <<hybrid_gemm>>
    }

    SGLangWorker ..> ServerArgs : launches with the hybrid_gemm backend
    ModelRunner --> Runtime : creates and holds, calls before_batch
    ModelRunner ..> apply_hybrid_gemm : calls set_ratio_handle
    apply_hybrid_gemm ..> RatioHandle : reads tensor
    UnquantizedLinearMethod ..> pinned_blocks : allocates weights
    device_loading_context ..> UnquantizedLinearMethod : keeps weights on CPU
    UnquantizedLinearMethod --> apply_hybrid_gemm : calls in apply
    apply_hybrid_gemm --> hybrid_gemm : calls
```
