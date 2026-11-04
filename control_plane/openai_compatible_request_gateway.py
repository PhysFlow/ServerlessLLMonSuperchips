from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

from .errors import ControlPlaneError, InvalidRequest
from .inference_response_stream import InferenceResponseStream
from .model_catalog import ModelCatalog
from .request_scheduler import RequestScheduler
from .types import ChatMessage, ModelInferenceRequest, WorkerExecutionConfiguration

__all__ = ["OpenAICompatibleRequestGateway"]

_CHAT_PATH = "/v1/chat/completions"
_COMPLETIONS_PATH = "/v1/completions"
_KINDS = {_CHAT_PATH: "chat", _COMPLETIONS_PATH: "text"}
_DONE = b"data: [DONE]\n\n"
_SSE_START = {"type": "http.response.start", "status": 200, "headers": [
    (b"content-type", b"text/event-stream"), (b"cache-control", b"no-cache")]}
_JSON_HEADERS = ((b"content-type", b"application/json"),)


class OpenAICompatibleRequestGateway:

    def __init__(self, scheduler: RequestScheduler, models: ModelCatalog) -> None:
        self._scheduler = scheduler
        self._models = models

    def parse_request(self, payload: Mapping[str, Any], *, request_id: str | None = None) -> ModelInferenceRequest:
        if not isinstance(payload, Mapping):
            raise InvalidRequest("request body must be a JSON object")
        model = self._require_model(payload)
        messages = _extract_messages(payload)
        return ModelInferenceRequest(
            _parse_request_id(request_id), model.model_id, messages,
            _parse_max_tokens(payload.get("max_tokens")), _parse_stream(payload.get("stream")),
        )

    async def schedule(self, request: ModelInferenceRequest) -> InferenceResponseStream:
        return await self._scheduler.schedule_and_submit(request)

    async def encode_stream(
        self, request: ModelInferenceRequest, stream: InferenceResponseStream, *, kind: str = "chat",
    ) -> AsyncIterator[bytes]:
        created = int(time.time())
        failure: ControlPlaneError | None = None
        try:
            async for text in stream:
                yield _sse(_stream_chunk(request.request_id, created, request.model_id, kind, text))
        except ControlPlaneError as error:
            failure = error
        finally:
            await stream.aclose()
        if failure is not None:
            yield _sse({"error": _error_payload(failure)})
        else:
            yield _sse(_stream_chunk(request.request_id, created, request.model_id, kind, "", "stop"))
        yield _DONE

    async def encode_response(
        self, request: ModelInferenceRequest, stream: InferenceResponseStream, *, kind: str = "chat",
    ) -> dict[str, Any]:
        parts: list[str] = []
        try:
            async for text in stream:
                parts.append(text)
        finally:
            await stream.aclose()
        return _completion_body(request.request_id, int(time.time()), request.model_id, kind, "".join(parts))

    @staticmethod
    def error_response(error: ControlPlaneError) -> tuple[int, dict[str, Any]]:
        return error.http_status, {"error": _error_payload(error)}

    @staticmethod
    def internal_error_response() -> tuple[int, dict[str, Any]]:
        return 500, {"error": {"code": "internal_error", "message": "internal server error",
                               "type": "memserve_control_plane_error"}}

    def _require_model(self, payload: Mapping[str, Any]) -> WorkerExecutionConfiguration:
        model_id = payload.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise InvalidRequest("request must include a non-empty model", details={"field": "model"})
        return self._models.get_configuration(model_id)

    async def __call__(self, scope: Mapping[str, Any], receive, send) -> None:
        if scope.get("type") != "http":
            return
        method, path = scope.get("method"), scope.get("path")
        if path == "/health" and method == "GET":
            await _send_json(send, 200, {"status": "ok"})
            return
        kind = _KINDS.get(path)
        if kind is None:
            await _send_json(send, 404, _error_body("not_found", "unknown path"))
            return
        if method != "POST":
            await _send_json(send, 405, _error_body("method_not_allowed", "only POST is supported"))
            return
        try:
            payload = await _read_json(receive)
            request = self.parse_request(payload, request_id=_header(scope, b"x-request-id"))
            stream = await self.schedule(request)
        except ControlPlaneError as error:
            status, body = self.error_response(error)
            await _send_json(send, status, body)
            return
        except Exception:
            status, body = self.internal_error_response()
            await _send_json(send, status, body)
            return
        if request.stream:
            await _send_event_stream(send, self, request, stream, kind)
            return
        try:
            body = await self.encode_response(request, stream, kind=kind)
        except ControlPlaneError as error:
            status, body = self.error_response(error)
            await _send_json(send, status, body)
            return
        except Exception:
            status, body = self.internal_error_response()
            await _send_json(send, status, body)
            return
        await _send_json(send, 200, body)


async def _send_event_stream(send, gateway, request, stream, kind) -> None:
    await send(dict(_SSE_START))
    events = gateway.encode_stream(request, stream, kind=kind)
    try:
        async for event in events:
            await send({"type": "http.response.body", "body": event, "more_body": True})
    finally:
        await events.aclose()
    await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _read_json(receive) -> Mapping[str, Any]:
    body = b""
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            raise asyncio.CancelledError
        if message.get("type") != "http.request":
            continue
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    if not body:
        raise InvalidRequest("request body must not be empty")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidRequest("request body must be valid JSON") from error
    if not isinstance(payload, Mapping):
        raise InvalidRequest("request body must be a JSON object")
    return payload


async def _send_json(send, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = [*_JSON_HEADERS, (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _header(scope: Mapping[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _extract_messages(payload: Mapping[str, Any]) -> tuple[ChatMessage, ...]:
    if "messages" in payload:
        return _chat_messages(payload["messages"])
    if "prompt" in payload:
        return (ChatMessage("user", _prompt_text(payload["prompt"])),)
    raise InvalidRequest("request must include messages or prompt", details={"field": "messages"})


def _chat_messages(raw: Any) -> tuple[ChatMessage, ...]:
    if not isinstance(raw, list) or not raw:
        raise InvalidRequest("messages must be a non-empty list", details={"field": "messages"})
    messages = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise InvalidRequest("each message must be an object", details={"field": f"messages[{index}]"})
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise InvalidRequest("each message must have a role", details={"field": f"messages[{index}].role"})
        messages.append(ChatMessage(role, _content_text(entry.get("content"), index)))
    return tuple(messages)


def _content_text(content: Any, index: int) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part["text"] for part in content
                 if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
        if parts:
            return "".join(parts)
    raise InvalidRequest("message content must be text", details={"field": f"messages[{index}].content"})


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and all(isinstance(item, str) for item in prompt):
        return "".join(prompt)
    raise InvalidRequest("prompt must be a string or list of strings", details={"field": "prompt"})


def _parse_request_id(raw: str | None) -> str:
    if not raw:
        return str(uuid4())
    if not raw.strip():
        raise InvalidRequest("x-request-id must not be blank", details={"field": "x-request-id"})
    return raw


def _parse_max_tokens(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise InvalidRequest("max_tokens must be a positive integer", details={"field": "max_tokens"})
    return raw


def _parse_stream(raw: Any) -> bool:
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise InvalidRequest("stream must be a boolean", details={"field": "stream"})
    return raw


def _stream_chunk(event_id: str, created: int, model_id: str, kind: str, text: str,
                  finish_reason: str | None = None) -> dict[str, Any]:
    if kind == "chat":
        delta = {"content": text} if text else {}
        return {"id": event_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return {"id": event_id, "object": "text_completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}


def _completion_body(event_id: str, created: int, model_id: str, kind: str, text: str) -> dict[str, Any]:
    if kind == "chat":
        return {"id": event_id, "object": "chat.completion", "created": created, "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}
    return {"id": event_id, "object": "text_completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]}


def _error_payload(error: ControlPlaneError) -> dict[str, Any]:
    payload = {"code": error.code, "message": error.message,
               "type": "memserve_control_plane_error", "retryable": error.retryable}
    if error.details:
        payload["details"] = dict(error.details)
    return payload


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "type": "memserve_control_plane_error"}}


def _sse(payload: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"
