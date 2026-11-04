from __future__ import annotations

import importlib

from .types import MigResourceCapacity

__all__ = ["MigResourceDiscoverer"]


class MigResourceDiscoverer:
    def __init__(self, *, nvml=None) -> None:
        self._nvml = nvml

    def discover(self) -> tuple[MigResourceCapacity, ...]:
        if self._nvml is None:
            try:
                nvml = importlib.import_module("pynvml")
            except ImportError as error:
                raise RuntimeError("MIG discovery requires nvidia-ml-py") from error
        else:
            nvml = self._nvml
        try:
            nvml.nvmlInit()
        except Exception as error:
            raise RuntimeError("NVML could not initialize; an NVIDIA GPU and driver are required") from error
        try:
            resources = []
            for gpu_index in range(nvml.nvmlDeviceGetCount()):
                gpu = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
                try:
                    count = nvml.nvmlDeviceGetMaxMigDeviceCount(gpu)
                except nvml.NVMLError_NotSupported:
                    continue
                for mig_index in range(count):
                    try:
                        mig = nvml.nvmlDeviceGetMigDeviceHandleByIndex(gpu, mig_index)
                    except nvml.NVMLError_NotFound:
                        continue
                    resources.append(MigResourceCapacity(_text(nvml.nvmlDeviceGetUUID(mig))))
            if len({item.instance_id for item in resources}) != len(resources):
                raise ValueError("NVML reported duplicate MIG instance IDs")
            return tuple(sorted(resources, key=lambda item: item.instance_id))
        finally:
            nvml.nvmlShutdown()


def _text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value
