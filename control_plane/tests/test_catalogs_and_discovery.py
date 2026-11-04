import pytest

from memserve_control_plane.errors import ModelDisabled, SchedulableModelNotFound
from memserve_control_plane.mig_resource_discoverer import MigResourceDiscoverer
from memserve_control_plane.model_catalog import (
    MAX_CONSECUTIVE_STARTUP_FAILURES, STARTUP_RETRY_DELAY_SECONDS, ModelCatalog,
)
from memserve_control_plane.types import WorkerExecutionConfiguration
from test_basic_control_plane import FakeNvml


def test_model_catalog_looks_up_configurations_and_rejects_duplicates():
    a, b = (WorkerExecutionConfiguration(model, f"models/{model}") for model in ("a", "b"))
    catalog = ModelCatalog((b, a))
    assert catalog.get_configuration("a") == a
    assert catalog.require_schedulable("b") == b
    with pytest.raises(SchedulableModelNotFound):
        catalog.get_configuration("missing")
    with pytest.raises(ValueError, match="twice"):
        ModelCatalog((a, a))


def test_model_is_disabled_after_consecutive_startup_failures():
    catalog = ModelCatalog((WorkerExecutionConfiguration("a", "models/a"),))
    for _ in range(MAX_CONSECUTIVE_STARTUP_FAILURES - 1):
        catalog.mark_failed("a", "HybridGEMM mismatch")
        assert catalog.require_schedulable("a").model_id == "a"
    catalog.mark_failed("a", "HybridGEMM mismatch")
    with pytest.raises(ModelDisabled) as disabled:
        catalog.require_schedulable("a")
    details = dict(disabled.value.details)
    assert details["failures"] == str(MAX_CONSECUTIVE_STARTUP_FAILURES)
    assert details["reason"] == "HybridGEMM mismatch"
    assert catalog.get_configuration("a").model_id == "a"


def test_successful_start_resets_startup_failures():
    catalog = ModelCatalog((WorkerExecutionConfiguration("a", "models/a"),))
    for _ in range(MAX_CONSECUTIVE_STARTUP_FAILURES - 1):
        catalog.mark_failed("a", "killed")
    catalog.mark_started("a")
    for _ in range(MAX_CONSECUTIVE_STARTUP_FAILURES - 1):
        catalog.mark_failed("a", "killed")
    assert catalog.require_schedulable("a").model_id == "a"


def test_failed_start_must_wait_before_retrying():
    now = [100.0]
    catalog = ModelCatalog((WorkerExecutionConfiguration("a", "models/a"),), clock=lambda: now[0])
    assert catalog.startup_retry_delay("a") == 0
    catalog.mark_failed("a", "killed")
    assert catalog.startup_retry_delay("a") == STARTUP_RETRY_DELAY_SECONDS
    now[0] += STARTUP_RETRY_DELAY_SECONDS - 10
    assert catalog.startup_retry_delay("a") == 10
    now[0] += 10
    assert catalog.startup_retry_delay("a") == 0
    catalog.mark_failed("a", "killed")
    catalog.mark_started("a")
    assert catalog.startup_retry_delay("a") == 0


def test_nvml_init_failure_has_actionable_error(monkeypatch):
    nvml = FakeNvml()
    def fail():
        raise RuntimeError("driver missing")
    monkeypatch.setattr(nvml, "nvmlInit", fail)
    with pytest.raises(RuntimeError, match="NVIDIA GPU and driver"):
        MigResourceDiscoverer(nvml=nvml).discover()
    assert not nvml.stopped


@pytest.mark.parametrize("failure", ["duplicate", "read"])
def test_nvml_shutdown_runs_after_discovery_failure(monkeypatch, failure):
    nvml = FakeNvml()
    if failure == "duplicate":
        monkeypatch.setattr(nvml, "nvmlDeviceGetMigDeviceHandleByIndex", lambda *args: "mig")
    else:
        def fail(_):
            raise RuntimeError("device disappeared")
        monkeypatch.setattr(nvml, "nvmlDeviceGetUUID", fail)
    with pytest.raises((ValueError, RuntimeError)):
        MigResourceDiscoverer(nvml=nvml).discover()
    assert nvml.stopped


def test_non_mig_gpu_is_skipped(monkeypatch):
    nvml = FakeNvml()
    def unsupported(_):
        raise nvml.NVMLError_NotSupported
    monkeypatch.setattr(nvml, "nvmlDeviceGetMaxMigDeviceCount", unsupported)
    assert MigResourceDiscoverer(nvml=nvml).discover() == ()
    assert nvml.stopped
