from __future__ import annotations

from typing import Any

__all__ = ["uvicorn_runner"]

_SINGLE_PROCESS_FOR_IN_MEMORY_STATE = 1


async def uvicorn_runner(app: Any, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(
            "uvicorn is required to serve the Control Plane; install the 'server' extra"
        ) from error
    config = uvicorn.Config(app, host=host, port=port, log_level="info", workers=_SINGLE_PROCESS_FOR_IN_MEMORY_STATE)
    await uvicorn.Server(config).serve()
