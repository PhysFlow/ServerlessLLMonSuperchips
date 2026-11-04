from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys


__all__ = ["ConfigError", "ControlPlaneConfig", "EngineConfig", "GatewayConfig", "ModelConfig", "load_config"]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    max_retries: int = 3
    retry_delay: float = 0.0
    queue_timeout: float = 300.0


@dataclass(frozen=True, slots=True)
class EngineConfig:
    kind: str = "sglang"
    python_executable: str = sys.executable
    host: str = "127.0.0.1"
    base_port: int = 30000
    startup_timeout: float = 300.0
    request_timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    model_path: str
    resident: bool = False


@dataclass(frozen=True, slots=True)
class ControlPlaneConfig:
    models: tuple[ModelConfig, ...]
    engine: EngineConfig = EngineConfig()
    gateway: GatewayConfig = GatewayConfig()


def load_config(path: str | Path) -> ControlPlaneConfig:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot load configuration from {path!r}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a JSON object")
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise ConfigError("models must be a non-empty list")
    parsed = tuple(_model(item, index) for index, item in enumerate(models))
    ids = [item.model_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise ConfigError("model_id values must be unique")
    engine = _object(raw.get("engine", {}), "engine")
    if engine.get("kind", "sglang") != "sglang":
        raise ConfigError("engine.kind must be 'sglang'")
    gateway = _object(raw.get("gateway", {}), "gateway")
    return ControlPlaneConfig(
        models=parsed,
        engine=EngineConfig(
            python_executable=_string(engine.get("python_executable", sys.executable), "engine.python_executable"),
            host=_string(engine.get("host", "127.0.0.1"), "engine.host"),
            base_port=_integer(engine.get("base_port", 30000), "engine.base_port", 1, 65535),
            startup_timeout=_positive(engine.get("startup_timeout", 300.0), "engine.startup_timeout"),
            request_timeout=_positive(engine.get("request_timeout", 600.0), "engine.request_timeout"),
        ),
        gateway=GatewayConfig(
            host=_string(gateway.get("host", "127.0.0.1"), "gateway.host"),
            port=_integer(gateway.get("port", 8000), "gateway.port", 1, 65535),
            max_retries=_integer(gateway.get("max_retries", 3), "gateway.max_retries", 0),
            retry_delay=_nonnegative(gateway.get("retry_delay", 0.0), "gateway.retry_delay"),
            queue_timeout=_positive(gateway.get("queue_timeout", 300.0), "gateway.queue_timeout"),
        ),
    )


def _model(raw, index: int) -> ModelConfig:
    item = _object(raw, f"models[{index}]")
    resident = item.get("resident", False)
    if not isinstance(resident, bool):
        raise ConfigError(f"models[{index}].resident must be a boolean")
    return ModelConfig(
        _string(item.get("model_id"), f"models[{index}].model_id"),
        _string(item.get("model_path"), f"models[{index}].model_path"), resident,
    )


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _integer(value, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{name} must be an integer between {minimum} and {maximum or 'infinity'}")
    return value


def _nonnegative(value, name: str) -> float:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} must be a non-negative finite number")
    return float(value)


def _positive(value, name: str) -> float:
    result = _nonnegative(value, name)
    if result == 0:
        raise ConfigError(f"{name} must be positive")
    return result
