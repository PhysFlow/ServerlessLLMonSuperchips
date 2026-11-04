import asyncio
from contextlib import contextmanager
import time
from threading import Event
from types import SimpleNamespace

import pytest

from memserve_control_plane import application
from memserve_control_plane.application import ControlPlane
from memserve_control_plane.config import ControlPlaneConfig, GatewayConfig, ModelConfig
from memserve_control_plane.errors import (
    ModelDisabled, NoFeasibleModelPlacement, WorkerRequestFailed, WorkerStartupFailed,
)
from memserve_control_plane import model_catalog
from memserve_control_plane.model_catalog import MAX_CONSECUTIVE_STARTUP_FAILURES
from memserve_control_plane.types import (
    ChatMessage, MigResourceCapacity, ModelInferenceRequest,
    WorkerLifecycleState, WorkerStatus,
)


class Backend:

    def __init__(self):
        self.events = []
        self.gates = {}
        self.fail_models = set()
        self.fail_after_send = False
        self.generation = 0

    def hold_start(self, model_id):
        self.gates[model_id] = Event()
        return self.gates[model_id]

    def count(self, kind, model_id=None):
        return sum(1 for event in self.events
                   if event[1] == kind and (model_id is None or event[2] == model_id))


class SGLangLikeWorker:
    def __init__(self, backend, instance_id):
        self.backend = backend
        self.instance_id = instance_id
        self.state = WorkerLifecycleState.STOPPED
        self.config = None
        self.generation = 0

    def start(self, config):
        self.backend.events.append((self.instance_id, "start", config.model_id))
        gate = self.backend.gates.get(config.model_id)
        if gate is not None and not gate.wait(5):
            raise WorkerStartupFailed("test gate was never released")
        if config.model_id in self.backend.fail_models:
            raise WorkerStartupFailed("HybridGEMM does not support this Linear invocation")
        self.backend.generation += 1
        self.generation = self.backend.generation
        self.config = config
        self.state = WorkerLifecycleState.READY

    def stop(self):
        if self.state is not WorkerLifecycleState.STOPPED:
            self.backend.events.append((self.instance_id, "stop", self.config.model_id))
        self.state = WorkerLifecycleState.STOPPED

    def get_status(self):
        active = self.config.model_id if self.state is WorkerLifecycleState.READY else None
        generation = self.generation if active else 0
        return WorkerStatus(self.instance_id, self.state, active, generation)

    @contextmanager
    def verify_status(self):
        yield self.get_status()

    def submit_request(self, request):
        self.backend.events.append((self.instance_id, "submit", request.model_id))
        if self.backend.fail_after_send:
            raise WorkerRequestFailed("request timed out after it was sent")
        return iter(("hello",))


def control_plane(*model_ids, mig_count=2, queue_timeout=5.0):
    backend = Backend()
    config = ControlPlaneConfig(
        tuple(ModelConfig(model_id, f"models/{model_id}") for model_id in model_ids),
        gateway=GatewayConfig(queue_timeout=queue_timeout),
    )
    resources = tuple(MigResourceCapacity(f"mig-{index}") for index in range(mig_count))

    def build_workers(instance_ids):
        return {instance_id: SGLangLikeWorker(backend, instance_id) for instance_id in instance_ids}

    plane = ControlPlane(
        config, discoverer=SimpleNamespace(discover=lambda: resources), build_workers=build_workers,
        server_runner=lambda app, host, port: _noop(),
    )
    return plane, backend


async def _noop():
    pass


def request(model_id, request_id):
    return ModelInferenceRequest(request_id, model_id, (ChatMessage("user", "hi"),))


async def submit(plane, model_id, request_id):
    stream = await plane.gateway.schedule(request(model_id, request_id))
    async with stream:
        return [item async for item in stream]


async def wait_for(condition, timeout=5.0):
    async def poll():
        while not condition():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(poll(), timeout)


def placed_models(plane):
    return {item.instance_id: item.model_id for item in plane.placements.snapshot_placements().placements}


def test_model_startup_failure_leaves_the_mig_usable_by_other_models(monkeypatch):
    monkeypatch.setattr(model_catalog, "STARTUP_RETRY_DELAY_SECONDS", 0)
    plane, backend = control_plane("bad", "good", mig_count=1)
    backend.fail_models.add("bad")

    async def run():
        await plane.start()
        for attempt in range(MAX_CONSECUTIVE_STARTUP_FAILURES):
            with pytest.raises(WorkerStartupFailed):
                await submit(plane, "bad", f"attempt-{attempt}")
        with pytest.raises(ModelDisabled):
            await submit(plane, "bad", "disabled")
        assert await submit(plane, "good", "after") == ["hello"]
        await plane.shutdown()

    asyncio.run(run())
    assert backend.count("start", "bad") == MAX_CONSECUTIVE_STARTUP_FAILURES
    assert backend.count("start", "good") == 1


def test_transient_startup_failure_does_not_disable_the_model(monkeypatch):
    monkeypatch.setattr(model_catalog, "STARTUP_RETRY_DELAY_SECONDS", 0)
    plane, backend = control_plane("flaky", "other", mig_count=1)

    async def fail_then_recover():
        backend.fail_models.add("flaky")
        for attempt in range(MAX_CONSECUTIVE_STARTUP_FAILURES - 1):
            with pytest.raises(WorkerStartupFailed):
                await submit(plane, "flaky", f"failure-{attempt}")
        backend.fail_models.discard("flaky")
        assert await submit(plane, "flaky", "recovered") == ["hello"]
        assert await submit(plane, "other", "evicts-flaky") == ["hello"]

    async def run():
        await plane.start()
        await fail_then_recover()
        await fail_then_recover()
        await plane.shutdown()

    asyncio.run(run())
    assert backend.count("start", "flaky") == 2 * MAX_CONSECUTIVE_STARTUP_FAILURES


def test_queued_request_waits_out_the_retry_delay_after_a_failed_start(monkeypatch):
    monkeypatch.setattr(model_catalog, "STARTUP_RETRY_DELAY_SECONDS", 0.3)
    plane, backend = control_plane("flaky", mig_count=1)
    backend.fail_models.add("flaky")

    async def run():
        await plane.start()
        with pytest.raises(WorkerStartupFailed):
            await submit(plane, "flaky", "failed")
        backend.fail_models.discard("flaky")
        started = time.monotonic()
        assert await submit(plane, "flaky", "retried") == ["hello"]
        await plane.shutdown()
        return time.monotonic() - started

    assert asyncio.run(run()) >= 0.3
    assert backend.count("start", "flaky") == 2


def test_cold_start_does_not_block_a_loaded_model():
    plane, backend = control_plane("loaded", "cold")

    async def run():
        await plane.start()
        await submit(plane, "loaded", "warm-up")
        gate = backend.hold_start("cold")
        cold = asyncio.create_task(submit(plane, "cold", "cold-1"))
        await wait_for(lambda: backend.count("start", "cold") == 1)
        assert await asyncio.wait_for(submit(plane, "loaded", "during-cold-start"), 2) == ["hello"]
        assert not cold.done()
        gate.set()
        assert await cold == ["hello"]
        await plane.shutdown()

    asyncio.run(run())


def test_concurrent_requests_for_a_loading_model_start_it_once():
    plane, backend = control_plane("a")

    async def run():
        await plane.start()
        gate = backend.hold_start("a")
        first = asyncio.create_task(submit(plane, "a", "one"))
        await wait_for(lambda: backend.count("start", "a") == 1)
        second = asyncio.create_task(submit(plane, "a", "two"))
        await asyncio.sleep(0.05)
        gate.set()
        assert await asyncio.gather(first, second) == [["hello"], ["hello"]]
        await plane.shutdown()

    asyncio.run(run())
    assert backend.count("start", "a") == 1


def test_request_waits_while_every_mig_is_busy_instead_of_failing():
    plane, backend = control_plane("a", "b", "c")

    async def run():
        await plane.start()
        stream_a = await plane.gateway.schedule(request("a", "long-a"))
        stream_b = await plane.gateway.schedule(request("b", "long-b"))
        waiting = asyncio.create_task(submit(plane, "c", "queued"))
        await asyncio.sleep(0.1)
        assert not waiting.done()
        await stream_a.aclose()
        assert await asyncio.wait_for(waiting, 2) == ["hello"]
        assert placed_models(plane) == {"mig-0": "c", "mig-1": "b"}
        await stream_b.aclose()
        await plane.shutdown()

    asyncio.run(run())


def test_queued_request_fails_after_queue_timeout():
    plane, _ = control_plane("a", "b", mig_count=1, queue_timeout=0.2)

    async def run():
        await plane.start()
        stream = await plane.gateway.schedule(request("a", "long"))
        with pytest.raises(NoFeasibleModelPlacement):
            await submit(plane, "b", "queued")
        await stream.aclose()
        await plane.shutdown()

    asyncio.run(run())


def test_eviction_replaces_the_least_recently_used_model():
    plane, backend = control_plane("a", "b", "c")

    async def run():
        await plane.start()
        await submit(plane, "a", "one")
        await submit(plane, "b", "two")
        await submit(plane, "a", "three")
        await submit(plane, "c", "four")
        assert placed_models(plane) == {"mig-0": "a", "mig-1": "c"}
        await plane.shutdown()

    asyncio.run(run())


def test_request_failure_after_send_is_not_resubmitted():
    plane, backend = control_plane("a")
    backend.fail_after_send = True

    async def run():
        await plane.start()
        with pytest.raises(WorkerRequestFailed):
            await submit(plane, "a", "one")
        await plane.shutdown()

    asyncio.run(run())
    assert backend.count("submit") == 1


def test_crashed_worker_is_recovered_and_its_mig_reused(monkeypatch):
    monkeypatch.setattr(application, "_RECOVERY_SECONDS", 0.01)
    plane, backend = control_plane("a", mig_count=1)

    async def run():
        await plane.start()
        await submit(plane, "a", "before-crash")
        plane.workers.get_worker("mig-0").state = WorkerLifecycleState.STOPPED
        assert await asyncio.wait_for(submit(plane, "a", "after-crash"), 2) == ["hello"]
        await plane.shutdown()

    asyncio.run(run())
    assert backend.count("start", "a") == 2
