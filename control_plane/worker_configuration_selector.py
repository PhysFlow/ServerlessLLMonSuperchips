from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable

from .link_bandwidth import link_bytes_per_s

__all__ = ["WorkerConfigurationSelector"]

logger = logging.getLogger("memserve.control_plane")

CHUNK_SIZES = (256, 512, 1024, 2048)
RATIOS = tuple(step / 20 for step in range(21))
CHUNK_TOLERANCE = 1.1
BF16_FLOPS_PER_SM = 7.5e12

ModelConfigLoader = Callable[[str], object]


class WorkerConfigurationSelector:
    def __init__(self, *, nvml=None, load_model_config: ModelConfigLoader | None = None) -> None:
        self._nvml = nvml
        self._load_model_config = load_model_config or _load_hf_config

    def select(self, model_path: str, instance_id: str) -> tuple[int, float]:
        shapes = _linear_shapes(self._load_model_config(model_path))
        sm_count, hbm_bytes_per_s, link_bytes_per_s = self._device(instance_id)
        flops_per_s = sm_count * BF16_FLOPS_PER_SM

        def cost(m: int, ratio: float) -> float:
            return sum(_gemm_seconds(m, n, k, ratio, link_bytes_per_s, hbm_bytes_per_s, flops_per_s)
                       for n, k in shapes)

        per_token = {chunk: min(cost(chunk, ratio) for ratio in RATIOS) / chunk for chunk in CHUNK_SIZES}
        best = min(per_token.values())
        chunk = min(size for size in CHUNK_SIZES if per_token[size] <= best * CHUNK_TOLERANCE)
        prefill = {ratio: cost(chunk, ratio) for ratio in RATIOS}
        decode = {ratio: cost(1, ratio) for ratio in RATIOS}
        lowest_prefill, lowest_decode = min(prefill.values()), min(decode.values())
        ratio = min(RATIOS, key=lambda value: (prefill[value] / lowest_prefill + decode[value] / lowest_decode,
                                               -value))
        logger.info(
            "worker configuration for %s on %s: chunk %d, ratio %.2f "
            "(SMs %d, HBM %.1f GB/s, link %.1f GB/s, prefill %.3f ms, decode %.3f ms)",
            model_path, instance_id, chunk, ratio, sm_count, hbm_bytes_per_s / 1e9, link_bytes_per_s / 1e9,
            prefill[ratio] * 1e3, decode[ratio] * 1e3,
        )
        return chunk, ratio

    def _device(self, instance_id: str) -> tuple[int, float, float]:
        nvml = self._nvml or importlib.import_module("pynvml")
        nvml.nvmlInit()
        try:
            mig = nvml.nvmlDeviceGetHandleByUUID(instance_id)
            gpu = nvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig)
            sm_count = nvml.nvmlDeviceGetAttributes(mig).multiprocessorCount
            memory_share = nvml.nvmlDeviceGetMemoryInfo(mig).total / nvml.nvmlDeviceGetMemoryInfo(gpu).total
            memory_clock_mhz = nvml.nvmlDeviceGetMaxClockInfo(gpu, nvml.NVML_CLOCK_MEM)
            bus_width_bits = nvml.nvmlDeviceGetMemoryBusWidth(gpu)
            hbm_bytes_per_s = memory_clock_mhz * 1e6 * 2 * bus_width_bits / 8 * memory_share
            return sm_count, hbm_bytes_per_s, link_bytes_per_s(nvml, gpu)
        finally:
            nvml.nvmlShutdown()


def _linear_shapes(config) -> tuple[tuple[int, int], ...]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    return (
        ((heads + 2 * kv_heads) * head_dim, hidden),
        (hidden, heads * head_dim),
        (2 * intermediate, hidden),
        (hidden, intermediate),
    )


def _gemm_seconds(m: int, n: int, k: int, ratio: float,
                  link_bytes_per_s: float, hbm_bytes_per_s: float, flops_per_s: float) -> float:
    padded_m = math.ceil(m / 256) * 256
    asymmetric_n = math.floor(ratio * (n // 256) + 0.5) * 256
    symmetric_n = n - asymmetric_n
    link_bytes = (padded_m // 256 * symmetric_n + asymmetric_n) * k * 2
    hbm_bytes = k // 256 * padded_m * asymmetric_n * 4
    return max(link_bytes / link_bytes_per_s, hbm_bytes / hbm_bytes_per_s, 2 * padded_m * n * k / flops_per_s)


def _load_hf_config(model_path: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)
