import asyncio
import json

import pytest

from memserve_control_plane.errors import (
    InvalidRequest, NoFeasibleModelPlacement,
    SchedulableModelNotFound, WorkerUnavailable,
)
from memserve_control_plane.model_catalog import ModelCatalog
from memserve_control_plane.openai_compatible_request_gateway import OpenAICompatibleRequestGateway
from memserve_control_plane.types import ChatMessage, WorkerExecutionConfiguration


MODEL = WorkerExecutionConfiguration("target", "models/target")


class FakeStream:
    def __init__(self, items=(), error=None):
        self._items = list(items)
        self._error = error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._items:
            return self._items.pop(0)
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class StubScheduler:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def schedule_and_submit(self, request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_gateway(scheduler=None):
    catalog = ModelCatalog((MODEL,))
    scheduler = scheduler or StubScheduler([FakeStream(["ok"])])
    return OpenAICompatibleRequestGateway(scheduler, catalog), scheduler


def chat_payload(**overrides):
    payload = {"model": "target", "messages": [{"role": "user", "content": "hello"}]}
    payload.update(overrides)
    return payload


def parse_events(events):
    frames = [event for event in events if event.startswith(b"data: ")]
    return frames


def decode(frame):
    assert frame.startswith(b"data: ") and frame.endswith(b"\n\n")
    body = frame[len(b"data: "):-2]
    if body == b"[DONE]":
        return "[DONE]"
    return json.loads(body)


def test_parse_chat_request_defaults():
    gateway, _ = make_gateway()

    request = gateway.parse_request(chat_payload())

    assert request.model_id == "target"
    assert request.messages == (ChatMessage("user", "hello"),)
    assert request.stream is False


def test_parse_completions_prompt_becomes_user_message():
    gateway, _ = make_gateway()

    request = gateway.parse_request({"model": "target", "prompt": ["a", "b"]})

    assert request.messages == (ChatMessage("user", "ab"),)


def test_parse_request_id_override():
    gateway, _ = make_gateway()

    request = gateway.parse_request(chat_payload(), request_id="fixed")

    assert request.request_id == "fixed"


def test_parse_rejects_blank_request_id():
    gateway, _ = make_gateway()

    with pytest.raises(InvalidRequest):
        gateway.parse_request(chat_payload(), request_id="   ")


@pytest.mark.parametrize("payload,error", [
    ({"messages": []}, InvalidRequest),
    ({"model": "target"}, InvalidRequest),
    ({"model": "missing", "messages": [{"role": "user", "content": "x"}]}, SchedulableModelNotFound),
    ({"model": "target", "messages": [{"role": "user", "content": "x"}], "stream": "yes"}, InvalidRequest),
    ({"model": "target", "messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, InvalidRequest),
])
def test_parse_rejects_invalid_payloads(payload, error):
    gateway, _ = make_gateway()

    with pytest.raises(error):
        gateway.parse_request(payload)


def test_error_response_exposes_stable_metadata():
    status, body = OpenAICompatibleRequestGateway.error_response(
        NoFeasibleModelPlacement("no placement", details={"model_id": "target"}),
    )

    assert status == 503
    assert body["error"]["code"] == "no_feasible_model_placement"
    assert body["error"]["retryable"] is False
    assert body["error"]["details"] == {"model_id": "target"}


def test_encode_stream_emits_chunks_then_done_and_closes():
    gateway, _ = make_gateway(StubScheduler([FakeStream(["Hello", " world"])]))
    request = gateway.parse_request(chat_payload(), request_id="req-1")
    stream = asyncio.run(gateway.schedule(request))

    events = asyncio.run(_collect(gateway.encode_stream(request, stream)))

    frames = [decode(frame) for frame in parse_events(events)]
    assert [frame["choices"][0]["delta"].get("content") for frame in frames[:-1]] == ["Hello", " world", None]
    assert frames[-1] == "[DONE]"
    assert frames[0]["id"] == "req-1"
    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert stream.closed


def test_encode_response_returns_single_completion_body():
    gateway, _ = make_gateway(StubScheduler([FakeStream(["Hello", " world"])]))
    request = gateway.parse_request(chat_payload(), request_id="req-1")
    stream = asyncio.run(gateway.schedule(request))

    body = asyncio.run(gateway.encode_response(request, stream))

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert stream.closed


def test_stream_failure_is_encoded_as_sse_error_and_done():
    gateway, _ = make_gateway(StubScheduler([FakeStream(["a"], error=WorkerUnavailable("boom"))]))
    request = gateway.parse_request(chat_payload())
    stream = asyncio.run(gateway.schedule(request))

    events = asyncio.run(_collect(gateway.encode_stream(request, stream)))

    frames = [decode(frame) for frame in parse_events(events)]
    assert frames[0]["choices"][0]["delta"]["content"] == "a"
    assert frames[1]["error"]["code"] == "worker_unavailable"
    assert frames[-1] == "[DONE]"


async def _collect(async_iterator):
    return [item async for item in async_iterator]
