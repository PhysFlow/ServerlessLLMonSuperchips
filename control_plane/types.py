from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class MigResourceCapacity:
    instance_id: str


@dataclass(frozen=True, slots=True)
class WorkerExecutionConfiguration:
    model_id: str
    model_path: str
    chunk_size: int | None = None
    initial_asymmetric_work_ratio: float | None = None


class WorkerLifecycleState(str, Enum):
    READY = "ready"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    instance_id: str
    state: WorkerLifecycleState
    active_model_id: str | None
    worker_generation: int

    def __post_init__(self) -> None:
        if self.state is WorkerLifecycleState.READY and self.active_model_id is None:
            raise ValueError("ready worker must report its model")
        if self.active_model_id is not None and not self.worker_generation:
            raise ValueError("active worker must report a positive generation")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelInferenceRequest:
    request_id: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int | None = None
    stream: bool = False


@dataclass(frozen=True, slots=True)
class ModelPlacementDecision:
    instance_id: str
    model_id: str
    expected_worker_generation: int | None
    requires_restart: bool = False


@dataclass(frozen=True, slots=True)
class ModelPlacement:
    instance_id: str
    model_id: str
    worker_generation: int

    def matches(self, status: WorkerStatus | None) -> bool:
        return (status is not None and status.state is WorkerLifecycleState.READY
                and status.active_model_id == self.model_id
                and status.worker_generation == self.worker_generation)


@dataclass(frozen=True, slots=True)
class ModelPlacementSnapshot:
    placements: tuple[ModelPlacement, ...]
    switching: tuple[str, ...] = ()
    active_requests: tuple[tuple[str, int], ...] = ()
    last_used: tuple[tuple[str, int], ...] = ()

    def requests_on(self, instance_id: str) -> int:
        return dict(self.active_requests).get(instance_id, 0)

    def last_used_at(self, instance_id: str) -> int:
        return dict(self.last_used).get(instance_id, 0)
