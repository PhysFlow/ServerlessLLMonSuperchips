from contextlib import nullcontext
from dataclasses import replace
import io
import json
import signal
import subprocess
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from memserve_control_plane.errors import WorkerStartupFailed, WorkerStateInconsistent, WorkerUnavailable
from memserve_control_plane import model_serving_worker as worker_module
from memserve_control_plane.model_serving_worker import SGLangWorker
from memserve_control_plane.types import WorkerLifecycleState
from test_scheduling_transactions import CONFIG


class Process:
    def __init__(self):
        self.dead = False
        self.signals = []
        self.time_out = False

    def poll(self):
        return 0 if self.dead else None

    def terminate(self):
        self.signals.append(signal.SIGTERM)

    def kill(self):
        self.signals.append(signal.SIGKILL)

    def wait(self, timeout):
        if self.time_out:
            self.time_out = False
            raise subprocess.TimeoutExpired("sglang", timeout)
        self.dead = True


def worker():
    return SGLangWorker("test-mig", cuda_visible_devices="MIG-test", startup_timeout=0.01, poll_interval=0)


def test_sglang_launch_configuration_environment_and_generations(monkeypatch):
    backend = SGLangWorker(
        "test-mig", cuda_visible_devices="MIG-test",
        python_executable="/tmp/memserve-venv/bin/python",
        startup_timeout=0.01, poll_interval=0,
    )
    calls = []
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", "true")
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(backend, "_is_healthy", lambda: True)
    monkeypatch.setattr(backend, "_warm_up", lambda deadline: None)
    backend.start(CONFIG)
    first = backend.get_status()
    command, kwargs = calls[0]
    assert command[command.index("--model-path") + 1] == "models/target"
    assert command[command.index("--linear-gemm-backend") + 1] == "hybrid_gemm"
    assert command[command.index("--chunked-prefill-size") + 1] == "32"
    assert command[command.index("--hybrid-gemm-asym-ratio") + 1] == "0.5"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "MIG-test"
    assert kwargs["env"]["PATH"] == "/tmp/memserve-venv/bin:/usr/bin:/bin"
    assert kwargs["env"]["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] == "false"
    assert kwargs["start_new_session"] is True
    with pytest.raises(WorkerStateInconsistent):
        backend.start(CONFIG)
    backend.stop()
    backend.start(CONFIG)
    assert backend.get_status().worker_generation > first.worker_generation
    backend.stop()


def test_sglang_launch_selects_missing_chunk_and_ratio(monkeypatch):
    class Selector:
        def __init__(self):
            self.calls = []

        def select(self, model_path, instance_id):
            self.calls.append((model_path, instance_id))
            return 2048, 0.75

    selector = Selector()
    backend = SGLangWorker("test-mig", cuda_visible_devices="MIG-test", configuration_selector=selector)
    automatic = backend._launch_command(replace(CONFIG, chunk_size=None, initial_asymmetric_work_ratio=None))
    assert automatic[automatic.index("--chunked-prefill-size") + 1] == "2048"
    assert automatic[automatic.index("--hybrid-gemm-asym-ratio") + 1] == "0.75"
    assert selector.calls == [("models/target", "MIG-test")]
    partial = backend._launch_command(replace(CONFIG, initial_asymmetric_work_ratio=None))
    assert partial[partial.index("--chunked-prefill-size") + 1] == "32"
    assert partial[partial.index("--hybrid-gemm-asym-ratio") + 1] == "0.75"
    backend._launch_command(CONFIG)
    assert len(selector.calls) == 2


def test_sglang_launch_resolves_python_name_for_path(monkeypatch):
    backend = SGLangWorker(
        "test-mig", cuda_visible_devices="MIG-test",
        python_executable="python3", startup_timeout=0.01, poll_interval=0,
    )
    calls = []
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(worker_module.shutil, "which", lambda command: "/opt/venv/bin/python3")
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: calls.append(kwargs) or Process())
    monkeypatch.setattr(backend, "_is_healthy", lambda: True)
    monkeypatch.setattr(backend, "_warm_up", lambda deadline: None)

    backend.start(CONFIG)
    assert calls[0]["env"]["PATH"] == "/opt/venv/bin:/usr/bin:/bin"
    backend.stop()


def test_sglang_health_check_uses_non_generating_endpoint(monkeypatch):
    backend = worker()
    calls = []

    def healthy(url, timeout):
        calls.append((url, timeout))
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(worker_module.urllib_request, "urlopen", healthy)
    assert backend._is_healthy()
    assert calls == [("http://127.0.0.1:30000/health", 5)]

    def starting(url, timeout):
        raise HTTPError(url, 503, "starting", {}, None)

    monkeypatch.setattr(worker_module.urllib_request, "urlopen", starting)
    assert not backend._is_healthy()

    def unavailable(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(worker_module.urllib_request, "urlopen", unavailable)
    assert not backend._is_healthy()


def test_sglang_startup_timeout_stops_spawned_process(monkeypatch):
    backend = worker()
    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(backend, "_is_healthy", lambda: False)
    with pytest.raises(WorkerStartupFailed, match="timeout"):
        backend.start(CONFIG)
    assert process.dead
    assert process.signals == [signal.SIGTERM]


def test_sglang_launch_failure_is_a_startup_failure(monkeypatch):
    backend = worker()
    def fail(*args, **kwargs):
        raise FileNotFoundError("python")
    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(WorkerStartupFailed, match="launched"):
        backend.start(CONFIG)
    assert backend.get_status().state is WorkerLifecycleState.STOPPED


def test_sglang_verify_status_reports_current_status():
    backend = worker()
    with backend.verify_status() as status:
        assert status.state is WorkerLifecycleState.STOPPED


def test_sglang_stop_escalates_to_kill_and_is_idempotent():
    backend = worker()
    process = Process()
    process.time_out = True
    backend._process = process
    backend.stop()
    backend.stop()
    assert process.signals == [signal.SIGTERM, signal.SIGKILL]


def test_sglang_unhealthy_or_exited_worker_is_not_ready(monkeypatch):
    backend = worker()
    backend._process = Process()
    monkeypatch.setattr(backend, "_is_healthy", lambda: False)
    with pytest.raises(WorkerUnavailable):
        backend.get_status()
    backend._process.dead = True
    assert backend.get_status().state is WorkerLifecycleState.STOPPED


def test_sglang_start_warms_up_with_default_sampling_before_ready(monkeypatch):
    backend = worker()
    calls = []

    def respond(request, timeout):
        calls.append(json.loads(request.data))
        return nullcontext(io.BytesIO(b'{"choices": []}'))

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(backend, "_is_healthy", lambda: True)
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", respond)
    backend.start(CONFIG)
    assert calls == [{"model": "models/target", "messages": [{"role": "user", "content": "Hello"}],
                      "max_tokens": 1, "stream": False}]
    backend.stop()


def test_sglang_failed_warm_up_fails_startup_and_stops_process(monkeypatch):
    backend = worker()
    process = Process()

    def fail(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(backend, "_is_healthy", lambda: True)
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", fail)
    with pytest.raises(WorkerStartupFailed, match="warm-up"):
        backend.start(CONFIG)
    assert process.dead
