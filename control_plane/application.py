from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from threading import Condition, RLock
from typing import Any, Protocol

from .config import ControlPlaneConfig
from .engine import build_workers, check_hybrid_gemm_compatibility
from .errors import NoFeasibleModelPlacement
from .mig_resource_discoverer import MigResourceDiscoverer
from .mig_state_table import MigStateTable
from .model_catalog import ModelCatalog
from .model_placer import ModelPlacer
from .model_placement_selector import ModelPlacementSelector
from .model_serving_worker import ModelServingWorker
from .openai_compatible_request_gateway import OpenAICompatibleRequestGateway
from .request_scheduler import RequestScheduler
from .server import uvicorn_runner
from .types import MigResourceCapacity, WorkerExecutionConfiguration
from .worker_registry import WorkerRegistry

__all__ = ["ControlPlane", "ServerRunner"]

ServerRunner = Callable[[Any, str, int], Awaitable[None]]
WorkerBuilder = Callable[[tuple[str, ...]], Mapping[str, ModelServingWorker]]

_RECOVERY_SECONDS = 5.0


class ResourceDiscoverer(Protocol):
    def discover(self) -> tuple[MigResourceCapacity, ...]: ...


class ControlPlane:
    def __init__(
        self, config: ControlPlaneConfig, *, discoverer: ResourceDiscoverer | None = None,
        build_workers: WorkerBuilder | None = None, server_runner: ServerRunner | None = None,
    ) -> None:
        self._config = config
        self._discoverer = discoverer or MigResourceDiscoverer()
        self._build_workers = build_workers
        self._server_runner = server_runner or uvicorn_runner
        self._models = ModelCatalog(
            WorkerExecutionConfiguration(model.model_id, model.model_path) for model in config.models
        )
        self._registry: WorkerRegistry | None = None
        self._placements: MigStateTable | None = None
        self._placer: ModelPlacer | None = None
        self._scheduler: RequestScheduler | None = None
        self._gateway: OpenAICompatibleRequestGateway | None = None
        self._recovery: asyncio.Task | None = None
        self._started = False

    @property
    def gateway(self) -> OpenAICompatibleRequestGateway:
        return self._require(self._gateway, "gateway")

    @property
    def placements(self) -> MigStateTable:
        return self._require(self._placements, "placements")

    @property
    def workers(self) -> WorkerRegistry:
        return self._require(self._registry, "workers")

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("ControlPlane.start() may only be called once")
        self._prepare()
        self._recovery = asyncio.create_task(self._recover_forever())
        try:
            for model in self._config.models:
                if model.resident:
                    await self._scheduler.provision(model.model_id)
        except BaseException:
            await self.shutdown()
            raise
        self._started = True

    async def run(self) -> None:
        await self.start()
        try:
            await self._server_runner(self.gateway, self._config.gateway.host, self._config.gateway.port)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._recovery is not None:
            self._recovery.cancel()
            with suppress(asyncio.CancelledError):
                await self._recovery
            self._recovery = None
        if self._registry is not None:
            await asyncio.to_thread(self._registry.stop_all)

    def _prepare(self) -> None:
        resources = self._discoverer.discover()
        if not resources:
            raise NoFeasibleModelPlacement("no MIG instances were discovered")
        instance_ids = tuple(item.instance_id for item in resources)
        if self._build_workers is None:
            check_hybrid_gemm_compatibility(self._config.models)
            workers = build_workers(self._config.engine, instance_ids)
        else:
            workers = self._build_workers(instance_ids)
        registry = WorkerRegistry(workers)
        self._registry = registry
        self._placements = MigStateTable(resources, registry.verify_worker)
        changed = Condition(RLock())
        self._placer = ModelPlacer(
            models=self._models, placements=self._placements, workers=registry,
            selector=ModelPlacementSelector(), changed=changed,
        )
        self._scheduler = RequestScheduler(
            models=self._models, placements=self._placements, workers=registry, placer=self._placer, changed=changed,
            queue_timeout=self._config.gateway.queue_timeout,
            max_retries=self._config.gateway.max_retries, retry_delay=self._config.gateway.retry_delay,
        )
        self._gateway = OpenAICompatibleRequestGateway(self._scheduler, self._models)

    async def _recover_forever(self) -> None:
        while True:
            await asyncio.sleep(_RECOVERY_SECONDS)
            await asyncio.to_thread(self._placer.recover_isolated)

    @staticmethod
    def _require(value, name: str):
        if value is None:
            raise RuntimeError(f"ControlPlane is not started: {name} is unavailable")
        return value
