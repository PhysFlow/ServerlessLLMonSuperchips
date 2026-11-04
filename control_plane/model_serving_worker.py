from __future__ import annotations

import os
import json
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import RLock
import time
from typing import Protocol
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .errors import WorkerRequestFailed, WorkerStartupFailed, WorkerStateInconsistent, WorkerUnavailable
from .types import (
    ModelInferenceRequest, WorkerExecutionConfiguration, WorkerLifecycleState, WorkerStatus,
)
from .worker_configuration_selector import WorkerConfigurationSelector

__all__ = ["ModelServingWorker", "SGLangWorker"]


class ModelServingWorker(Protocol):
    instance_id: str

    def start(self, configuration: WorkerExecutionConfiguration) -> None: ...

    def stop(self) -> None: ...

    def get_status(self) -> WorkerStatus: ...

    def verify_status(self) -> AbstractContextManager[WorkerStatus]: ...

    def submit_request(self, request: ModelInferenceRequest) -> Iterator[str]: ...


class SGLangWorker:

    def __init__(self, instance_id: str, *,
                 cuda_visible_devices: str, dtype: str = "bfloat16",
                 host: str = "127.0.0.1", port: int = 30000,
                 python_executable: str = sys.executable, startup_timeout: float = 300.0,
                 request_timeout: float = 600.0, poll_interval: float = 0.2,
                 configuration_selector: WorkerConfigurationSelector | None = None) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (instance_id, cuda_visible_devices, dtype, host)):
            raise ValueError("SGLang startup strings must be non-empty")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        self.instance_id = instance_id
        self._cuda_visible_devices = cuda_visible_devices
        self._dtype = dtype
        self._host = host
        self._port = port
        self._python = python_executable
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._poll_interval = poll_interval
        self._configuration_selector = configuration_selector or WorkerConfigurationSelector()
        self._process: subprocess.Popen | None = None
        self._configuration: WorkerExecutionConfiguration | None = None
        self._worker_generation = 0
        self._base_url = f"http://{host}:{port}"
        self._lock = RLock()

    def start(self, configuration: WorkerExecutionConfiguration) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise WorkerStateInconsistent("SGLang process is already running")
            try:
                self._process = subprocess.Popen(
                    self._launch_command(configuration), env=self._environment(), start_new_session=True,
                )
            except OSError as error:
                raise WorkerStartupFailed("SGLang process could not be launched") from error
            self._configuration = configuration
            self._worker_generation += 1
            try:
                deadline = time.monotonic() + self._startup_timeout
                self._await_ready(deadline)
                self._warm_up(deadline)
            except BaseException:
                self.stop()
                raise

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self._signal_process(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_process(process, signal.SIGKILL)
                process.wait(timeout=10)

    def get_status(self) -> WorkerStatus:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            if not running:
                return WorkerStatus(self.instance_id, WorkerLifecycleState.STOPPED, None,
                                    self._worker_generation)
            if not self._is_healthy():
                raise WorkerUnavailable("SGLang health check failed")
            return WorkerStatus(self.instance_id, WorkerLifecycleState.READY,
                                self._configuration.model_id, self._worker_generation)

    @contextmanager
    def verify_status(self):
        with self._lock:
            yield self.get_status()

    def submit_request(self, request: ModelInferenceRequest) -> Iterator[str]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            if not running or request.model_id != self._configuration.model_id:
                raise WorkerStateInconsistent("worker is not serving the requested model")
            payload = {
                "model": self._configuration.model_path,
                "messages": [{"role": item.role, "content": item.content} for item in request.messages],
                "stream": True,
            }
            if request.max_tokens is not None:
                payload["max_tokens"] = request.max_tokens
            internal_request = urllib_request.Request(
                f"{self._base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                method="POST",
            )
            try:
                return _SGLangTextStream(urllib_request.urlopen(internal_request, timeout=self._request_timeout))
            except OSError as error:
                if _never_sent(error):
                    raise WorkerUnavailable("SGLang request endpoint is unavailable") from error
                raise WorkerRequestFailed("SGLang request failed after it was sent") from error

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        python_bin = os.path.dirname(os.path.abspath(shutil.which(self._python) or self._python))
        inherited_path = environment.get("PATH")
        environment["PATH"] = f"{python_bin}{os.pathsep}{inherited_path}" if inherited_path else python_bin
        environment["CUDA_VISIBLE_DEVICES"] = self._cuda_visible_devices
        environment["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
        return environment

    def _launch_command(self, configuration) -> list[str]:
        chunk_size, ratio = configuration.chunk_size, configuration.initial_asymmetric_work_ratio
        if chunk_size is None or ratio is None:
            selected_chunk_size, selected_ratio = self._configuration_selector.select(
                configuration.model_path, self._cuda_visible_devices,
            )
            chunk_size = selected_chunk_size if chunk_size is None else chunk_size
            ratio = selected_ratio if ratio is None else ratio
        command = [self._python, "-m", "sglang.launch_server", "--model-path", configuration.model_path,
                   "--dtype", self._dtype, "--host", self._host, "--port", str(self._port),
                   "--chunked-prefill-size", str(chunk_size),
                   "--disable-piecewise-cuda-graph",
                   "--linear-gemm-backend", "hybrid_gemm",
                   "--hybrid-gemm-asym-ratio", str(ratio)]
        return command

    def _await_ready(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise WorkerStartupFailed("SGLang process exited before it was healthy")
            if self._is_healthy():
                return
            time.sleep(self._poll_interval)
        raise WorkerStartupFailed("SGLang process did not become healthy before timeout")

    def _warm_up(self, deadline: float) -> None:
        payload = {"model": self._configuration.model_path, "messages": [{"role": "user", "content": "Hello"}],
                   "max_tokens": 1, "stream": False}
        warm_up = urllib_request.Request(
            f"{self._base_url}/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib_request.urlopen(warm_up, timeout=max(deadline - time.monotonic(), 1.0)) as response:
                response.read()
        except OSError as error:
            raise WorkerStartupFailed("SGLang warm-up request failed") from error

    @staticmethod
    def _signal_process(process, signal_number: int) -> None:
        try:
            if hasattr(process, "pid"):
                os.killpg(process.pid, signal_number)
            elif signal_number == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass

    def _is_healthy(self) -> bool:
        try:
            with urllib_request.urlopen(f"{self._base_url}/health", timeout=5) as response:
                return response.status == 200
        except OSError:
            return False


def _never_sent(error: OSError) -> bool:
    if isinstance(error, ConnectionRefusedError):
        return True
    return (isinstance(error, URLError) and not isinstance(error, HTTPError)
            and isinstance(error.reason, ConnectionRefusedError))


class _SGLangTextStream:

    def __init__(self, response) -> None:
        self._response = response
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration
        try:
            for line in self._response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    self.close()
                    raise StopIteration
                chunk = json.loads(data)
                text = chunk["choices"][0].get("delta", {}).get("content")
                if isinstance(text, str) and text:
                    return text
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            self.close()
            raise WorkerUnavailable("SGLang response stream failed") from error
        self.close()
        raise StopIteration

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._response.close()
