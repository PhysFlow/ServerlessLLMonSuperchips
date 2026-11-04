import asyncio
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from memserve_control_plane.application import ControlPlane
from memserve_control_plane.config import ConfigError, ControlPlaneConfig, ModelConfig, load_config
from memserve_control_plane.errors import (
    ModelDisabled, NoFeasibleModelPlacement, WorkerStartupFailed,
)
from memserve_control_plane.mig_resource_discoverer import MigResourceDiscoverer
from memserve_control_plane import model_catalog
from memserve_control_plane.model_catalog import MAX_CONSECUTIVE_STARTUP_FAILURES
from memserve_control_plane.model_placement_selector import ModelPlacementSelector
from memserve_control_plane.types import (
    ChatMessage, MigResourceCapacity, ModelInferenceRequest, ModelPlacement,
    ModelPlacementSnapshot, WorkerLifecycleState, WorkerStatus,
)


def test_minimal_user_config_needs_only_model_and_path(tmp_path):
    path = tmp_path / "control-plane.json"
    path.write_text(json.dumps({"models": [{"model_id": "a", "model_path": "org/a"}]}))

    config = load_config(path)

    assert config.models == (ModelConfig("a", "org/a"),)
    assert config.engine.kind == "sglang"
    assert config.gateway.port == 8000


@pytest.mark.parametrize("payload,message", [
    ({"models": []}, "non-empty"),
    ({"models": [{"model_id": "a", "model_path": ""}]}, "model_path"),
    ({"models": [{"model_id": "a", "model_path": "x", "resident": "yes"}]}, "resident"),
    ({"models": [{"model_id": "a", "model_path": "x"}, {"model_id": "a", "model_path": "y"}]}, "unique"),
])
def test_invalid_user_config_is_rejected(tmp_path, payload, message):
    path = tmp_path / "control-plane.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


class FakeNvml:
    class NVMLError_NotSupported(Exception):
        pass

    class NVMLError_NotFound(Exception):
        pass

    def __init__(self):
        self.stopped = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.stopped = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return "gpu"

    def nvmlDeviceGetMaxMigDeviceCount(self, gpu):
        return 2

    def nvmlDeviceGetMigDeviceHandleByIndex(self, gpu, index):
        if index:
            raise self.NVMLError_NotFound
        return "mig"

    def nvmlDeviceGetUUID(self, mig):
        return b"MIG-a"


def test_mig_resources_come_from_nvml_without_calibration_file():
    nvml = FakeNvml()
    resources = MigResourceDiscoverer(nvml=nvml).discover()
    assert resources == (MigResourceCapacity("MIG-a"),)
    assert nvml.stopped


class FakeWorker:
    def __init__(self, instance_id, events, fail_start_model=None):
        self.instance_id = instance_id
        self.generation = 0
        self.events = events
        self.state = WorkerLifecycleState.STOPPED
        self.config = None
        self.fail_start_model = fail_start_model

    def start(self, config):
        if config.model_id == self.fail_start_model:
            raise WorkerStartupFailed("fake startup failed")
        self.generation += 1
        self.config = config
        self.state = WorkerLifecycleState.READY
        self.events.append((self.instance_id, "start", config.model_id))

    def stop(self):
        self.state = WorkerLifecycleState.STOPPED
        self.events.append((self.instance_id, "stop"))

    def get_status(self):
        active = self.config.model_id if self.state is WorkerLifecycleState.READY else None
        return WorkerStatus(self.instance_id, self.state, active, self.generation if active else 0)

    @contextmanager
    def verify_status(self):
        yield self.get_status()

    def submit_request(self, request):
        self.events.append((self.instance_id, "request", request.model_id))
        return iter(("hello", " world"))


def system(*model_ids, mig_count=2, resident=(), fail_start_model=None):
    config = ControlPlaneConfig(tuple(
        ModelConfig(model_id, f"models/{model_id}", model_id in resident)
        for model_id in model_ids
    ))
    resources = tuple(MigResourceCapacity(f"mig-{index}") for index in range(mig_count))
    events = []
    workers = {}

    def build_workers(instance_ids):
        workers.update((instance_id, FakeWorker(instance_id, events, fail_start_model))
                       for instance_id in instance_ids)
        return workers

    plane = ControlPlane(
        config, discoverer=SimpleNamespace(discover=lambda: resources), build_workers=build_workers,
        server_runner=lambda app, host, port: _noop(),
    )
    return plane, events, workers


async def _noop():
    pass


async def submit(plane, model_id, request_id):
    request = ModelInferenceRequest(request_id, model_id, (ChatMessage("user", "hi"),))
    stream = await plane.gateway.schedule(request)
    async with stream:
        return [item async for item in stream]


def test_matching_worker_is_reused_and_empty_mig_is_used_first():
    plane, events, _ = system("a", "b")

    async def run():
        await plane.start()
        assert await submit(plane, "a", "one") == ["hello", " world"]
        assert await submit(plane, "a", "two") == ["hello", " world"]
        await submit(plane, "b", "three")
        await plane.shutdown()

    asyncio.run(run())
    assert [item for item in events if item[1] == "start"] == [
        ("mig-0", "start", "a"), ("mig-1", "start", "b"),
    ]


def test_concurrent_requests_share_one_model_worker():
    plane, events, _ = system("a", mig_count=1)

    async def run():
        await plane.start()
        result = await asyncio.gather(submit(plane, "a", "one"), submit(plane, "a", "two"))
        await plane.shutdown()
        return result

    assert asyncio.run(run()) == [["hello", " world"], ["hello", " world"]]
    assert sum(event[1] == "start" for event in events) == 1


def test_idle_worker_can_be_replaced_on_the_same_mig():
    plane, events, _ = system("a", "b", mig_count=1)

    async def run():
        await plane.start()
        await submit(plane, "a", "one")
        await submit(plane, "b", "two")
        await plane.shutdown()

    asyncio.run(run())
    assert events[:5] == [
        ("mig-0", "start", "a"), ("mig-0", "request", "a"),
        ("mig-0", "stop"), ("mig-0", "start", "b"), ("mig-0", "request", "b"),
    ]


def test_resident_model_starts_with_control_plane():
    plane, events, _ = system("a", resident=("a",))
    asyncio.run(plane.start())
    assert events[0] == ("mig-0", "start", "a")
    asyncio.run(plane.shutdown())


def test_repeatedly_failing_model_is_not_placed_again(monkeypatch):
    monkeypatch.setattr(model_catalog, "STARTUP_RETRY_DELAY_SECONDS", 0)
    plane, _, _ = system("a", mig_count=1, fail_start_model="a")

    async def run():
        await plane.start()
        for attempt in range(MAX_CONSECUTIVE_STARTUP_FAILURES):
            with pytest.raises(WorkerStartupFailed):
                await submit(plane, "a", f"attempt-{attempt}")
        with pytest.raises(ModelDisabled):
            await submit(plane, "a", "disabled")
        await plane.shutdown()

    asyncio.run(run())


def test_busy_worker_is_not_replaced():
    placement = ModelPlacement("mig-0", "a", 1)
    snapshot = ModelPlacementSnapshot((placement,), active_requests=(("mig-0", 1),))
    status = WorkerStatus("mig-0", WorkerLifecycleState.READY, "a", 1)
    with pytest.raises(NoFeasibleModelPlacement):
        ModelPlacementSelector().select_placement(
            "b", (MigResourceCapacity("mig-0"),), snapshot,
            {"mig-0": status}, frozenset(),
        )


def test_eviction_excludes_switching_busy_and_mismatched_workers():
    resources = tuple(MigResourceCapacity(f"mig-{index}") for index in range(4))
    placements = tuple(ModelPlacement(f"mig-{index}", "old", 1) for index in range(4))
    statuses = {
        "mig-0": WorkerStatus("mig-0", WorkerLifecycleState.READY, "old", 1),
        "mig-1": WorkerStatus("mig-1", WorkerLifecycleState.READY, "old", 1),
        "mig-2": WorkerStatus("mig-2", WorkerLifecycleState.READY, "old", 1),
        "mig-3": WorkerStatus("mig-3", WorkerLifecycleState.READY, "old", 2),
    }

    decision = ModelPlacementSelector().select_placement(
        "new", resources, ModelPlacementSnapshot(placements, ("mig-1",), (("mig-2", 1),)), statuses, frozenset(),
    )

    assert (decision.instance_id, decision.requires_restart) == ("mig-0", True)


def test_open_response_stream_delays_redeployment_until_it_closes():
    plane, _, _ = system("a", "b", mig_count=1)

    async def run():
        await plane.start()
        request = ModelInferenceRequest("one", "a", (ChatMessage("user", "hi"),))
        stream = await plane.gateway.schedule(request)
        waiting = asyncio.create_task(submit(plane, "b", "two"))
        await asyncio.sleep(0.1)
        assert not waiting.done()
        await stream.aclose()
        assert await asyncio.wait_for(waiting, 2) == ["hello", " world"]
        await plane.shutdown()

    asyncio.run(run())


def test_asgi_chat_request_returns_model_output():
    plane, _, _ = system("a")

    async def run():
        await plane.start()
        messages = [{"type": "http.request", "body": json.dumps({
            "model": "a", "messages": [{"role": "user", "content": "hi"}],
        }).encode(), "more_body": False}]
        sent = []

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await plane.gateway({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
                            receive, send)
        await plane.shutdown()
        return sent

    sent = asyncio.run(run())
    assert sent[0]["status"] == 200
    assert json.loads(sent[1]["body"])["choices"][0]["message"]["content"] == "hello world"


def test_asgi_stream_returns_text_chunks_and_done():
    plane, _, _ = system("a")

    async def run():
        await plane.start()
        body = json.dumps({"model": "a", "messages": [{"role": "user", "content": "hi"}],
                           "stream": True}).encode()
        sent = []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        await plane.gateway({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
                            receive, send)
        await plane.shutdown()
        return sent

    sent = asyncio.run(run())
    assert sent[0]["status"] == 200
    payload = b"".join(item.get("body", b"") for item in sent).decode()
    assert "hello" in payload and " world" in payload
    assert "data: [DONE]" in payload


def test_no_mig_fails_start():
    plane, _, _ = system("a", mig_count=0)
    with pytest.raises(NoFeasibleModelPlacement, match="no MIG"):
        asyncio.run(plane.start())
