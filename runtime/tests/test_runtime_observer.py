import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pynvml

from memserve_runtime import RuntimeObserver, RuntimeSample
from memserve_runtime.runtime_observer import _GpmSource


METRIC_VALUES = {
    pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL: 13.7,
    pynvml.NVML_GPM_METRIC_PCIE_RX_PER_SEC: 16_000.0,
    pynvml.NVML_GPM_METRIC_C2C_DATA_RX_PER_SEC: 0.0,
}


class FakeNvml:
    def __init__(self, failing=(), supported=True, c2c_speeds=()):
        self.failing = set(failing)
        self.supported = supported
        self.c2c_speeds = c2c_speeds
        self.calls = []

    def __getattr__(self, name):
        return getattr(pynvml, name)

    def nvmlInit(self):
        self.calls.append("init")

    def nvmlShutdown(self):
        self.calls.append("shutdown")

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return uuid

    def nvmlDeviceGetDeviceHandleFromMigDeviceHandle(self, handle):
        return "parent"

    def nvmlDeviceGetGpuInstanceId(self, handle):
        return 6

    def nvmlGpmQueryDeviceSupport(self, device):
        return SimpleNamespace(isSupportedDevice=int(self.supported))

    def nvmlDeviceGetPcieSpeed(self, device):
        return 16_000

    def nvmlDeviceGetCurrPcieLinkWidth(self, device):
        return 16

    def nvmlDeviceGetFieldValues(self, device, fields):
        def value(field):
            field_id, index = field if isinstance(field, tuple) else (field, 0)
            if field_id == self.NVML_FI_DEV_C2C_LINK_COUNT:
                number = len(self.c2c_speeds)
            else:
                number = self.c2c_speeds[index]
            return SimpleNamespace(nvmlReturn=0, value=SimpleNamespace(uiVal=number))
        return [value(field) for field in fields]

    def nvmlGpmSampleAlloc(self):
        return pynvml.c_nvmlGpmSample_t()

    def nvmlGpmSampleFree(self, sample):
        self.calls.append("free")

    def nvmlGpmMigSampleGet(self, device, instance, sample):
        self.calls.append(("mig", device, instance))

    def nvmlGpmSampleGet(self, device, sample):
        self.calls.append(("device", device))

    def nvmlGpmMetricsGet(self, request):
        for index in range(request.numMetrics):
            metric = request.metrics[index]
            failed = metric.metricId in self.failing
            metric.nvmlReturn = 1 if failed else 0
            metric.value = 0.0 if failed else METRIC_VALUES[metric.metricId]
        return request


class BlockingSource:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.sampling = False
        self.closed = False
        self.closed_during_sample = False

    def sample(self):
        self.sampling = True
        self.entered.set()
        self.release.wait(5)
        self.sampling = False
        return None

    def close(self):
        self.closed_during_sample = self.sampling
        self.closed = True


def gpm_source(nvml, uuid="MIG-example", environment=None):
    with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": uuid, **(environment or {})}):
        return _GpmSource(SimpleNamespace(index=0), nvml=nvml)


class RuntimeObserverTests(unittest.TestCase):
    def test_disabling_auto_sampling_keeps_manual_observations_available(self):
        class Source:
            def sample(self):
                raise AssertionError("sampling should be disabled")

        observer = RuntimeObserver(source=Source())
        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "0"}):
            observer.start(SimpleNamespace(index=0))
        observer.observe(RuntimeSample(hbm_utilization=0.4, timestamp_ns=100))

        self.assertEqual(observer.snapshot(100).hbm_utilization, 0.4)
        observer.close()

    def test_close_closes_active_sampling_source(self):
        class Source:
            closed = False

            def sample(self):
                return RuntimeSample(hbm_utilization=0.5)

            def close(self):
                self.closed = True

        source = Source()
        observer = RuntimeObserver(source=source, sample_interval_s=60)
        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "1"}):
            observer.start(SimpleNamespace(index=0))

        observer.close()

        self.assertTrue(source.closed)

    def test_close_waits_for_in_flight_sample_before_closing_source(self):
        source = BlockingSource()
        observer = RuntimeObserver(source=source, sample_interval_s=0.01)
        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "1"}):
            observer.start(SimpleNamespace(index=0))
        self.assertTrue(source.entered.wait(5))

        closer = threading.Thread(target=observer.close)
        closer.start()
        time.sleep(0.05)
        self.assertFalse(source.closed)
        source.release.set()
        closer.join(5)

        self.assertFalse(source.closed_during_sample)
        self.assertTrue(source.closed)

    def test_close_keeps_source_open_when_sampling_thread_is_stuck(self):
        source = BlockingSource()
        observer = RuntimeObserver(source=source, sample_interval_s=0.01)
        with patch.dict("os.environ", {"MEMSERVE_AUTO_RATIO": "1"}):
            observer.start(SimpleNamespace(index=0))
        self.assertTrue(source.entered.wait(5))

        with self.assertLogs("memserve_runtime.runtime_observer", "WARNING"):
            observer.close()

        self.assertFalse(source.closed)
        source.release.set()

    def test_observe_smooths_each_measurement_independently(self):
        observer = RuntimeObserver(ema_weight=0.5)
        observer.observe(RuntimeSample(link_utilization=0.8, timestamp_ns=100))
        observer.observe(
            RuntimeSample(
                link_utilization=0.4,
                hbm_utilization=0.2,
                timestamp_ns=200,
            )
        )
        observer.observe(RuntimeSample(request_latency_ms=12, timestamp_ns=300))

        snapshot = observer.snapshot(300)

        self.assertAlmostEqual(snapshot.link_utilization, 0.6)
        self.assertEqual(snapshot.hbm_utilization, 0.2)
        self.assertEqual(snapshot.request_latency_ms, 12)

    def test_snapshot_omits_expired_measurements(self):
        observer = RuntimeObserver(stale_after_s=1)
        observer.observe(RuntimeSample(link_utilization=0.8, timestamp_ns=100))

        self.assertEqual(observer.snapshot(1_000_000_100).link_utilization, 0.8)
        self.assertIsNone(observer.snapshot(1_000_000_101).link_utilization)

    def test_older_sample_does_not_replace_newer_measurement(self):
        observer = RuntimeObserver()
        observer.observe(RuntimeSample(link_utilization=0.8, timestamp_ns=200))
        observer.observe(RuntimeSample(link_utilization=0.2, timestamp_ns=100))

        self.assertEqual(observer.snapshot(200).link_utilization, 0.8)


class RuntimeSampleTests(unittest.TestCase):
    def test_rejects_negative_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timestamp_ns"):
            RuntimeSample(hbm_utilization=0.5, timestamp_ns=-1)

    def test_rejects_utilization_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "hbm_utilization"):
            RuntimeSample(hbm_utilization=1.1)


class GpmSourceTests(unittest.TestCase):
    def test_mig_source_reads_instance_hbm_and_device_link(self):
        nvml = FakeNvml()
        source = gpm_source(nvml)

        sample = source.sample()

        link_bytes_per_s = 16_000e6 / 8 * 16 * 128 / 130
        self.assertAlmostEqual(source.link_bytes_per_s, link_bytes_per_s)
        self.assertAlmostEqual(sample.hbm_utilization, 0.137)
        self.assertAlmostEqual(sample.link_utilization, 16_000 * 2**20 / link_bytes_per_s)
        self.assertIn(("mig", "parent", 6), nvml.calls)
        self.assertIn(("device", "parent"), nvml.calls)

    def test_c2c_link_bandwidth_comes_from_nvml(self):
        source = gpm_source(FakeNvml(c2c_speeds=(225_000, 225_000)))

        self.assertEqual(source.link_bytes_per_s, 450e9)

    def test_incomplete_c2c_bandwidth_is_rejected(self):
        nvml = FakeNvml(c2c_speeds=(225_000, 225_000))
        original = nvml.nvmlDeviceGetFieldValues

        def incomplete(device, fields):
            values = original(device, fields)
            if len(fields) == 2:
                values[1].nvmlReturn = 1
            return values

        nvml.nvmlDeviceGetFieldValues = incomplete
        with self.assertRaisesRegex(RuntimeError, "C2C"):
            gpm_source(nvml)

    def test_failed_metric_is_reported_as_missing(self):
        source = gpm_source(FakeNvml(failing={pynvml.NVML_GPM_METRIC_DRAM_BW_UTIL}))

        sample = source.sample()

        self.assertIsNone(sample.hbm_utilization)
        self.assertIsNotNone(sample.link_utilization)

    def test_sample_is_none_when_every_metric_fails(self):
        source = gpm_source(FakeNvml(failing=set(METRIC_VALUES)))

        self.assertIsNone(source.sample())

    def test_unsupported_gpu_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "GPM"):
            gpm_source(FakeNvml(supported=False))

    def test_numeric_ordinal_is_rejected_before_nvml(self):
        nvml = FakeNvml()
        with self.assertRaisesRegex(ValueError, "GPU/MIG UUID"):
            gpm_source(nvml, uuid="0")
        self.assertEqual(nvml.calls, [])

    def test_close_frees_samples_and_shuts_down_nvml(self):
        nvml = FakeNvml()
        source = gpm_source(nvml)

        source.close()

        self.assertEqual(nvml.calls.count("free"), 4)
        self.assertEqual(nvml.calls[-1], "shutdown")


if __name__ == "__main__":
    unittest.main()
