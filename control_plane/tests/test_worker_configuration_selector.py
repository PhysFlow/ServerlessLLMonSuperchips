from types import SimpleNamespace

import pytest

from memserve_control_plane.worker_configuration_selector import WorkerConfigurationSelector

QWEN_7B = SimpleNamespace(hidden_size=3584, intermediate_size=18944, num_attention_heads=28,
                          num_key_value_heads=4, head_dim=128)
GIB = 1024 ** 3


class FakeNvml:
    NVML_CLOCK_MEM = 2
    NVML_FI_DEV_C2C_LINK_COUNT = 170
    NVML_FI_DEV_C2C_LINK_GET_MAX_BW = 172

    def __init__(self, *, sm_count=30, mig_memory=20 * GIB, gpu_memory=80 * GIB,
                 memory_clock_mhz=1593, bus_width_bits=5120, pcie_lane_mbps=32000, pcie_lanes=16,
                 c2c_speeds=()):
        self.sm_count = sm_count
        self.mig_memory = mig_memory
        self.gpu_memory = gpu_memory
        self.memory_clock_mhz = memory_clock_mhz
        self.bus_width_bits = bus_width_bits
        self.pcie_lane_mbps = pcie_lane_mbps
        self.pcie_lanes = pcie_lanes
        self.c2c_speeds = c2c_speeds
        self.calls = []

    def nvmlInit(self):
        self.calls.append("init")

    def nvmlShutdown(self):
        self.calls.append("shutdown")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.calls.append(uuid)
        return "mig"

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):
        assert handle == "mig"
        return "gpu"

    def nvmlDeviceGetAttributes(self, handle):
        assert handle == "mig"
        return SimpleNamespace(multiprocessorCount=self.sm_count)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=self.mig_memory if handle == "mig" else self.gpu_memory)

    def nvmlDeviceGetMaxClockInfo(self, handle, clock):
        assert (handle, clock) == ("gpu", self.NVML_CLOCK_MEM)
        return self.memory_clock_mhz

    def nvmlDeviceGetMemoryBusWidth(self, handle):
        assert handle == "gpu"
        return self.bus_width_bits

    def nvmlDeviceGetPcieSpeed(self, handle):
        assert handle == "gpu"
        return self.pcie_lane_mbps

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        assert handle == "gpu"
        return self.pcie_lanes

    def nvmlDeviceGetFieldValues(self, handle, fields):
        assert handle == "gpu"
        def value(field):
            field_id, index = field if isinstance(field, tuple) else (field, 0)
            if field_id == self.NVML_FI_DEV_C2C_LINK_COUNT:
                number = len(self.c2c_speeds)
            else:
                number = self.c2c_speeds[index]
            return SimpleNamespace(nvmlReturn=0, value=SimpleNamespace(uiVal=number))
        return [value(field) for field in fields]


def selector(nvml):
    return WorkerConfigurationSelector(nvml=nvml, load_model_config=lambda path: QWEN_7B)


def test_slow_link_selects_all_asymmetric_and_largest_chunk():
    nvml = FakeNvml(pcie_lane_mbps=12500, pcie_lanes=16)
    assert selector(nvml).select("models/qwen", "MIG-a") == (2048, 1.0)
    assert nvml.calls == ["init", "MIG-a", "shutdown"]


def test_slow_hbm_selects_all_symmetric():
    assert selector(FakeNvml(memory_clock_mhz=1, c2c_speeds=(225_000, 225_000))).select("models/qwen", "MIG-a")[1] == 0.0


def test_selection_is_deterministic():
    nvml = FakeNvml(sm_count=32, memory_clock_mhz=2619, bus_width_bits=6144, gpu_memory=96 * GIB, mig_memory=24 * GIB, c2c_speeds=(225_000, 225_000))
    first = selector(nvml).select("models/qwen", "MIG-a")
    assert selector(nvml).select("models/qwen", "MIG-a") == first


def test_link_bandwidth_falls_back_to_pcie():
    slow, fast = FakeNvml(pcie_lane_mbps=2500, pcie_lanes=1), FakeNvml(pcie_lane_mbps=10 ** 9)
    assert selector(slow).select("models/qwen", "MIG-a")[1] == 1.0
    assert selector(fast).select("models/qwen", "MIG-a")[1] < 1.0


def test_nvml_is_shut_down_when_a_query_fails(monkeypatch):
    nvml = FakeNvml()

    def fail(handle):
        raise RuntimeError("no attributes")

    monkeypatch.setattr(nvml, "nvmlDeviceGetAttributes", fail)
    with pytest.raises(RuntimeError):
        selector(nvml).select("models/qwen", "MIG-a")
    assert nvml.calls[-1] == "shutdown"
