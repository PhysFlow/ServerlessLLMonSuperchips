import asyncio
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from threading import Condition, Event, RLock

import pytest

from memserve_control_plane.errors import (
    WorkerRequestFailed, WorkerStartupFailed, WorkerStateInconsistent, WorkerUnavailable,
)
from memserve_control_plane.mig_state_table import MigStateTable
from memserve_control_plane.model_catalog import ModelCatalog
from memserve_control_plane.model_placement_selector import ModelPlacementSelector
from memserve_control_plane.model_placer import ModelPlacer
from memserve_control_plane.request_scheduler import RequestScheduler
from memserve_control_plane.types import (
    ChatMessage, MigResourceCapacity, ModelInferenceRequest, ModelPlacementDecision,
    WorkerExecutionConfiguration, WorkerLifecycleState, WorkerStatus,
)
from memserve_control_plane.worker_registry import WorkerRegistry


CONFIG = WorkerExecutionConfiguration("target", "models/target", 32, 0.5)
OLD_CONFIG = replace(CONFIG, model_id="old", model_path="models/old")
REQUEST = ModelInferenceRequest("request-1", "target", (ChatMessage("user", "hello"),))


class FakeStream:
    def __init__(self, worker):
        self.worker = worker
        self.tokens = iter(("first", "second"))
        self.closed = False
        self.fail_read = False
        self.fail_close = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail_read:
            raise WorkerUnavailable("stream failed")
        return next(self.tokens)

    def close(self):
        if self.fail_close:
            raise WorkerUnavailable("abort was not confirmed")
        if not self.closed:
            self.closed = True
            self.worker.inflight -= 1


class FakeWorker:
    def __init__(self, system, instance_id):
        self.system = system
        self.instance_id = instance_id
        self.config = None
        self.generation = 0
        self.state = WorkerLifecycleState.STOPPED
        self.inflight = 0
        self.fail_start = False
        self.fail_submit = False
        self.fail_stop = False
        self.on_start = lambda: None
        self.stream = None

    def start(self, config):
        self.system.events.append((self.instance_id, "start", config.model_id))
        self.config = config
        self.on_start()
        if self.fail_start:
            raise WorkerStartupFailed("fake startup failed")
        self.generation = next(self.system.generations)
        self.state = WorkerLifecycleState.READY

    def stop(self):
        if self.fail_stop:
            raise WorkerUnavailable("stop not confirmed")
        self.system.events.append((self.instance_id, "stop"))
        self.state = WorkerLifecycleState.STOPPED

    def get_status(self):
        active = self.config.model_id if self.state is WorkerLifecycleState.READY else None
        return WorkerStatus(self.instance_id, self.state, active, self.generation)

    @contextmanager
    def verify_status(self):
        yield self.get_status()

    def submit_request(self, request):
        if self.fail_submit:
            raise WorkerUnavailable("submission failed")
        self.inflight += 1
        self.stream = FakeStream(self)
        return self.stream


class System:
    def __init__(self, initial=(), max_retries=0):
        self.generations = count(1)
        self.events = []
        self.models = ModelCatalog((CONFIG, OLD_CONFIG))
        self.workers = {instance: FakeWorker(self, instance) for instance in ("a", "b")}
        self.registry = WorkerRegistry(self.workers)
        self.table = MigStateTable(
            tuple(MigResourceCapacity(instance) for instance in self.workers), self.registry.verify_worker,
        )
        changed = Condition(RLock())
        self.placer = ModelPlacer(
            models=self.models, placements=self.table, workers=self.registry,
            selector=ModelPlacementSelector(), changed=changed,
        )
        self.scheduler = RequestScheduler(
            models=self.models, placements=self.table, workers=self.registry, placer=self.placer,
            changed=changed, max_retries=max_retries,
        )
        for config in initial:
            asyncio.run(self.scheduler.provision(config.model_id))
        self.events.clear()


def run_request(system, request=REQUEST):
    async def consume():
        async with await system.scheduler.schedule_and_submit(request) as stream:
            return [item async for item in stream]
    return asyncio.run(consume())


def observed(system):
    return system.table.observe(system.table.snapshot_placements())


def assert_admission_released(system):
    snapshot = system.table.snapshot_placements()
    placement = snapshot.placements[0]
    decision = ModelPlacementDecision(
        placement.instance_id, placement.model_id, placement.worker_generation,
        requires_restart=True,
    )
    system.table.reserve_placement(decision)
    system.table.rollback_placement(placement.instance_id)


def test_direct_route_preserves_worker_and_generation():
    system = System((CONFIG,))
    generation = system.workers["a"].generation

    assert run_request(system) == ["first", "second"]

    assert system.events == []
    assert system.workers["a"].generation == generation


def test_empty_mig_starts_worker_and_commits_observed_generations():
    system = System()

    run_request(system)

    placement = system.table.snapshot_placements().placements[0]
    assert placement.worker_generation == system.workers["a"].generation
    assert system.table.snapshot_placements().switching == ()


def test_redeployment_stops_old_worker_before_starting_target():
    system = System((OLD_CONFIG,))
    system.table.reserve_placement(ModelPlacementDecision("b", "old", None))
    system.registry.get_worker("b").start(OLD_CONFIG)
    system.table.commit_placement("b")
    system.events.clear()

    run_request(system)

    assert system.events == [("a", "stop"), ("a", "start", "target")]
    assert system.table.snapshot_placements().placements[0].model_id == "target"


def test_start_failure_stops_target_rolls_back_and_frees_the_mig():
    system = System()
    system.workers["a"].fail_start = True

    with pytest.raises(WorkerStartupFailed):
        run_request(system)

    assert system.table.snapshot_placements().switching == ()
    assert system.workers["a"].state is WorkerLifecycleState.STOPPED
    assert "a" in observed(system)[1]


def test_unconfirmed_stop_retains_reservation_budget():
    system = System()

    def fail(worker):
        worker.fail_start = True
        worker.fail_stop = True

    fail(system.workers["a"])
    with pytest.raises(WorkerStartupFailed):
        run_request(system)

    assert system.table.snapshot_placements().switching == ("a",)
    assert "a" not in observed(system)[1]


def test_submit_failure_preserves_committed_worker_and_releases_admission():
    system = System()
    system.workers["a"].fail_submit = True

    with pytest.raises(WorkerUnavailable, match="submission"):
        run_request(system)

    assert len(system.table.snapshot_placements().placements) == 1
    assert system.workers["a"].state is WorkerLifecycleState.READY
    system.workers["a"].fail_submit = False
    assert run_request(system, replace(REQUEST, request_id="retry")) == ["first", "second"]


def test_close_before_first_token_closes_backend_and_releases_request():
    system = System((CONFIG,))

    async def close():
        stream = await system.scheduler.schedule_and_submit(REQUEST)
        await stream.aclose()
        await stream.aclose()

    asyncio.run(close())

    assert system.workers["a"].stream.closed
    assert system.workers["a"].inflight == 0
    assert_admission_released(system)


def test_stream_failure_closes_backend_request():
    system = System((CONFIG,))

    async def consume():
        stream = await system.scheduler.schedule_and_submit(REQUEST)
        system.workers["a"].stream.fail_read = True
        with pytest.raises(WorkerUnavailable, match="stream failed"):
            await anext(stream)

    asyncio.run(consume())

    assert system.workers["a"].inflight == 0
    assert_admission_released(system)


def test_failed_stream_close_keeps_admission_and_isolates_worker():
    system = System((CONFIG,))

    async def close():
        stream = await system.scheduler.schedule_and_submit(REQUEST)
        system.workers["a"].stream.fail_close = True
        with pytest.raises(WorkerUnavailable, match="abort"):
            await stream.aclose()
        assert system.workers["a"].inflight == 1
        assert "a" not in observed(system)[0]
        system.workers["a"].stream.fail_close = False
        await stream.aclose()

    asyncio.run(close())


def test_duplicate_active_request_is_not_submitted_twice():
    system = System((CONFIG,))

    async def schedule_twice():
        async with await system.scheduler.schedule_and_submit(REQUEST):
            with pytest.raises(WorkerStateInconsistent, match="already"):
                await system.scheduler.schedule_and_submit(REQUEST)

    asyncio.run(schedule_twice())


def test_setup_cancellation_waits_for_start_and_closes_resulting_stream():
    system = System()
    entered, release = Event(), Event()

    def start():
        entered.set()
        assert release.wait(timeout=5)

    system.workers["a"].on_start = start

    async def cancel():
        task = asyncio.create_task(system.scheduler.schedule_and_submit(REQUEST))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())

    assert system.workers["a"].inflight == 0
    assert system.workers["a"].stream.closed


def test_unknown_worker_after_failed_stop_keeps_mig_reserved():
    system = System()

    def fail(worker):
        worker.fail_start = True
        worker.fail_stop = True

    fail(system.workers["a"])
    with pytest.raises(WorkerStartupFailed):
        run_request(system)

    assert system.table.snapshot_placements().switching == ("a",)


def test_cancelling_async_stream_aborts_backend_before_releasing_admission():
    system = System((CONFIG,))

    async def cancel_read():
        entered = asyncio.Event()
        worker = system.workers["a"]

        class BlockingStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                worker.inflight = 0

        def submit(request):
            worker.inflight = 1
            return BlockingStream()

        worker.submit_request = submit
        stream = await system.scheduler.schedule_and_submit(REQUEST)
        task = asyncio.create_task(anext(stream))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_read())
    assert_admission_released(system)


def test_cancelled_backend_close_does_not_release_admission():
    system = System((CONFIG,))

    async def close():
        class CancelledClose:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return "token"

            async def aclose(self):
                raise asyncio.CancelledError

        system.workers["a"].submit_request = lambda request: CancelledClose()
        stream = await system.scheduler.schedule_and_submit(REQUEST)
        with pytest.raises(asyncio.CancelledError):
            await stream.aclose()

    asyncio.run(close())
    assert "a" not in observed(system)[0]
    from memserve_control_plane.errors import StaleModelPlacementReservation
    with pytest.raises(StaleModelPlacementReservation, match="busy"):
        assert_admission_released(system)


def fail_submissions(worker, *errors):
    remaining = list(errors)
    submit = worker.submit_request

    def failing(request):
        if remaining:
            raise remaining.pop(0)
        return submit(request)

    worker.submit_request = failing
    return remaining


def test_retryable_failure_is_rerouted_up_to_the_limit():
    system = System((CONFIG,), max_retries=2)
    remaining = fail_submissions(system.workers["a"], WorkerUnavailable("again"), WorkerUnavailable("again"))

    assert run_request(system) == ["first", "second"]

    assert remaining == []
    assert_admission_released(system)


def test_scheduling_stops_after_exhausting_retries():
    system = System((CONFIG,), max_retries=1)
    remaining = fail_submissions(system.workers["a"], WorkerUnavailable("a"), WorkerUnavailable("b"), WorkerUnavailable("c"))

    with pytest.raises(WorkerUnavailable, match="b"):
        run_request(system)

    assert len(remaining) == 1
    assert_admission_released(system)


def test_non_retryable_failure_is_not_retried():
    system = System((CONFIG,), max_retries=3)
    remaining = fail_submissions(system.workers["a"], WorkerRequestFailed("sent"), WorkerUnavailable("unused"))

    with pytest.raises(WorkerRequestFailed):
        run_request(system)

    assert len(remaining) == 1
