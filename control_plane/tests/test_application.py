import asyncio

import pytest

from memserve_control_plane.errors import WorkerStartupFailed
from test_basic_control_plane import submit, system


@pytest.mark.parametrize("property_name", ["gateway", "workers", "placements"])
def test_unstarted_services_are_unavailable(property_name):
    plane, _, _ = system("a")
    with pytest.raises(RuntimeError, match="not started"):
        getattr(plane, property_name)


def test_start_once_and_shutdown_is_repeatable():
    plane, _, _ = system("a")
    async def run():
        await plane.shutdown()
        await plane.start()
        with pytest.raises(RuntimeError, match="once"):
            await plane.start()
        await plane.shutdown()
        await plane.shutdown()
    asyncio.run(run())


def test_run_shuts_down_even_when_server_fails():
    plane, events, _ = system("a", resident=("a",))
    async def fail(app, host, port):
        assert app is plane.gateway
        assert (host, port) == ("127.0.0.1", 8000)
        raise RuntimeError("server failed")
    plane._server_runner = fail
    with pytest.raises(RuntimeError, match="server failed"):
        asyncio.run(plane.run())
    assert ("mig-0", "stop") in events


def test_failed_resident_start_cleans_up_other_workers():
    plane, events, _ = system("a", "b", resident=("a", "b"), fail_start_model="b")
    with pytest.raises(WorkerStartupFailed):
        asyncio.run(plane.start())
    assert ("mig-0", "stop") in events


def test_shutdown_continues_after_one_worker_stop_fails():
    plane, events, _ = system("a", "b")
    async def run():
        await plane.start()
        await submit(plane, "a", "one")
        await submit(plane, "b", "two")
        def fail():
            raise RuntimeError("cannot stop")
        plane.workers.get_worker("mig-0").stop = fail
        await plane.shutdown()
    asyncio.run(run())
    assert ("mig-1", "stop") in events
