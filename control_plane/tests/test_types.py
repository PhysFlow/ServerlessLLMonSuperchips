import pytest

from memserve_control_plane.types import WorkerLifecycleState, WorkerStatus


@pytest.mark.parametrize("active_model_id,worker_generation", [(None, 1), ("target", 0)])
def test_worker_status_requires_consistent_identity_and_generations(active_model_id, worker_generation):
    with pytest.raises(ValueError):
        WorkerStatus("a", WorkerLifecycleState.READY, active_model_id, worker_generation)
