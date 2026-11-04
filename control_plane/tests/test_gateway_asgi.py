import asyncio
import json

import pytest

from memserve_control_plane.openai_compatible_request_gateway import OpenAICompatibleRequestGateway

from test_scheduling_transactions import CONFIG, System, assert_admission_released


def make_app(system):
    return OpenAICompatibleRequestGateway(system.scheduler, system.models)


def chat_request(**overrides):
    payload = {"model": "target", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    return payload


def drive(app, method, path, payload=None, headers=()):
    async def run():
        scope = {"type": "http", "method": method, "path": path, "headers": list(headers)}
        body = b"" if payload is None else json.dumps(payload).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]
        sent = []

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        return sent

    return asyncio.run(run())


def status_and_body(sent):
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return status, payload


def json_body(sent):
    status, payload = status_and_body(sent)
    return status, json.loads(payload)


def sse_frames(sent):
    _, payload = status_and_body(sent)
    frames = []
    for event in payload.split(b"\n\n"):
        if not event:
            continue
        data = event[len(b"data: "):]
        frames.append("[DONE]" if data == b"[DONE]" else json.loads(data))
    return frames


def test_health_returns_ok():
    app = make_app(System((CONFIG,)))

    status, payload = json_body(drive(app, "GET", "/health"))

    assert (status, payload) == (200, {"status": "ok"})


def test_unknown_path_and_method_are_rejected():
    app = make_app(System((CONFIG,)))

    assert json_body(drive(app, "GET", "/nope"))[0] == 404
    assert json_body(drive(app, "GET", "/v1/chat/completions"))[0] == 405


def test_chat_completion_routes_to_running_worker_without_restart():
    system = System((CONFIG,))
    app = make_app(system)

    status, payload = json_body(drive(app, "POST", "/v1/chat/completions", chat_request()))

    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "firstsecond"
    assert system.events == []


def test_streaming_completion_emits_sse_chunks_and_done():
    system = System((CONFIG,))
    app = make_app(system)

    sent = drive(app, "POST", "/v1/chat/completions", chat_request(stream=True))
    frames = sse_frames(sent)

    assert sent[0]["headers"] == [(b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")]
    assert "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames[:-1]) == "firstsecond"
    assert frames[-1] == "[DONE]"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"


def test_request_id_header_is_reflected_in_stream_chunks():
    app = make_app(System((CONFIG,)))

    frames = sse_frames(drive(
        app, "POST", "/v1/chat/completions", chat_request(stream=True),
        headers=[(b"x-request-id", b"trace-9")],
    ))

    assert frames[0]["id"] == "trace-9"


def test_completions_endpoint_uses_text_object():
    app = make_app(System((CONFIG,)))

    status, payload = json_body(drive(app, "POST", "/v1/completions", {"model": "target", "prompt": "hi"}))

    assert status == 200
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "firstsecond"


@pytest.mark.parametrize("payload,status,code", [
    ({"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
     404, "schedulable_model_not_found"),
    ({"model": "target"}, 400, "invalid_request"),
    ({"messages": [{"role": "user", "content": "hi"}]}, 400, "invalid_request"),
])
def test_request_errors_map_to_http_status(payload, status, code):
    app = make_app(System((CONFIG,)))

    response_status, body = json_body(drive(app, "POST", "/v1/chat/completions", payload))

    assert response_status == status
    assert body["error"]["code"] == code


def test_malformed_json_is_rejected_without_traceback():
    app = make_app(System((CONFIG,)))

    async def run():
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
        messages = [{"type": "http.request", "body": b"{not json", "more_body": False}]
        sent = []

        async def receive():
            return messages.pop(0)

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        return sent

    status, body = json_body(asyncio.run(run()))

    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert "Traceback" not in json.dumps(body)


def test_worker_failure_exhausts_retries_and_returns_503():
    system = System((CONFIG,), max_retries=1)
    system.workers["a"].fail_submit = True
    app = make_app(system)

    status, body = json_body(drive(app, "POST", "/v1/chat/completions", chat_request()))

    assert status == 503
    assert body["error"]["code"] == "worker_unavailable"
    assert body["error"]["retryable"] is True


def test_client_disconnect_during_stream_releases_admission():
    system = System((CONFIG,))
    app = make_app(system)

    async def run():
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
        body = json.dumps(chat_request(stream=True)).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]
        calls = 0

        async def receive():
            return messages.pop(0)

        async def send(message):
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise RuntimeError("client disconnected")

        await app(scope, receive, send)

    with pytest.raises(RuntimeError, match="client disconnected"):
        asyncio.run(run())

    assert system.workers["a"].inflight == 0
    assert_admission_released(system)


def test_full_stream_releases_admission():
    system = System((CONFIG,))
    app = make_app(system)

    drive(app, "POST", "/v1/chat/completions", chat_request(stream=True))

    assert system.workers["a"].inflight == 0
    assert_admission_released(system)
