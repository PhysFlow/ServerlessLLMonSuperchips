import pytest

from memserve_control_plane.errors import WorkerUnavailable
from memserve_control_plane.worker_registry import WorkerRegistry
from test_scheduling_transactions import CONFIG, System


def test_worker_must_be_registered_under_its_own_instance_id():
    system = System()
    with pytest.raises(ValueError, match="instance_id"):
        WorkerRegistry({"b": system.workers["a"]})


def test_unknown_worker_is_unavailable():
    system = System()
    with pytest.raises(WorkerUnavailable):
        system.registry.get_worker("unknown")
    with pytest.raises(WorkerUnavailable):
        with system.registry.verify_worker("unknown"):
            pass


def test_verify_worker_reports_stopped_worker_as_absent():
    system = System((CONFIG,))
    with system.registry.verify_worker("a") as running:
        assert running.active_model_id == "target"
    with system.registry.verify_worker("b") as stopped:
        assert stopped is None
