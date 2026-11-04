import io
import json
from types import SimpleNamespace

import pytest

from urllib.error import HTTPError, URLError

from memserve_control_plane.errors import WorkerRequestFailed, WorkerStateInconsistent, WorkerUnavailable

from memserve_control_plane import model_serving_worker as worker_module
from memserve_control_plane.model_serving_worker import SGLangWorker
from memserve_control_plane.types import ChatMessage, ModelInferenceRequest, WorkerExecutionConfiguration


def serving_worker(**kwargs):
    worker = SGLangWorker("mig-0", cuda_visible_devices="mig-0", **kwargs)
    worker._configuration = WorkerExecutionConfiguration("model", "org/model")
    worker._process = SimpleNamespace(poll=lambda: None)
    return worker


def test_sglang_request_is_forwarded_and_sse_becomes_text(monkeypatch):
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b'data: [DONE]\n\n'
        )

    monkeypatch.setattr(worker_module.urllib_request, "urlopen", open_request)
    worker = serving_worker()
    request = ModelInferenceRequest("request-1", "model", (ChatMessage("user", "hi"),), max_tokens=8)

    stream = worker.submit_request(request)

    assert list(stream) == ["hello", " world"]
    assert captured["url"] == "http://127.0.0.1:30000/v1/chat/completions"
    assert captured["payload"] == {
        "model": "org/model", "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "max_tokens": 8,
    }
    assert stream._closed


def test_closing_sglang_stream_closes_backend_response(monkeypatch):
    response = io.BytesIO(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", lambda request, timeout: response)
    worker = serving_worker()
    request = ModelInferenceRequest("request-1", "model", (ChatMessage("user", "hi"),))

    stream = worker.submit_request(request)
    assert next(stream) == "hello"
    stream.close()

    assert response.closed


@pytest.mark.parametrize("data", [b'data: not-json\n\n', b'data: {"choices":42}\n\n'])
def test_malformed_backend_stream_is_closed_and_reports_domain_error(monkeypatch, data):
    response = io.BytesIO(data)
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", lambda request, timeout: response)
    worker = serving_worker()
    request = ModelInferenceRequest("r", "model", (ChatMessage("user", "hi"),))
    stream = worker.submit_request(request)
    with pytest.raises(WorkerUnavailable):
        next(stream)
    assert response.closed


def test_refused_connection_is_retryable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise URLError(ConnectionRefusedError("refused"))
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", unavailable)
    worker = serving_worker()
    request = ModelInferenceRequest("r", "model", (ChatMessage("user", "hi"),))
    with pytest.raises(WorkerUnavailable, match="endpoint"):
        worker.submit_request(request)


@pytest.mark.parametrize("failure", [
    TimeoutError("timed out"),
    HTTPError("http://127.0.0.1:30000", 500, "error", {}, None),
    ConnectionResetError("reset"),
])
def test_failure_after_sending_is_not_retryable(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", fail)
    worker = serving_worker()
    request = ModelInferenceRequest("r", "model", (ChatMessage("user", "hi"),))
    with pytest.raises(WorkerRequestFailed):
        worker.submit_request(request)


def test_request_uses_configured_timeout(monkeypatch):
    timeouts = []
    monkeypatch.setattr(worker_module.urllib_request, "urlopen",
                        lambda request, timeout: timeouts.append(timeout) or io.BytesIO(b""))
    worker = serving_worker(request_timeout=42)
    worker.submit_request(ModelInferenceRequest("r", "model", (ChatMessage("user", "hi"),)))
    assert timeouts == [42]


@pytest.mark.parametrize("model_id,running", [("model", False), ("other", True)])
def test_request_is_rejected_unless_the_worker_serves_its_model(monkeypatch, model_id, running):
    sent = []
    monkeypatch.setattr(worker_module.urllib_request, "urlopen", lambda request, timeout: sent.append(request))
    worker = serving_worker()
    if not running:
        worker._process = None
    with pytest.raises(WorkerStateInconsistent):
        worker.submit_request(ModelInferenceRequest("r", model_id, (ChatMessage("user", "hi"),)))
    assert sent == []
