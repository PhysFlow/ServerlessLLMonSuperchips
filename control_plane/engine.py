from __future__ import annotations

import logging
import socket
from collections.abc import Callable

from .config import ConfigError, EngineConfig, ModelConfig
from .model_serving_worker import SGLangWorker

__all__ = ["build_workers", "check_hybrid_gemm_compatibility"]

logger = logging.getLogger("memserve.control_plane")

PortCheck = Callable[[str, int], bool]


def build_workers(
    engine: EngineConfig, instance_ids: tuple[str, ...], *, port_check: PortCheck | None = None,
) -> dict[str, SGLangWorker]:
    if engine.kind != "sglang":
        raise ValueError(f"unsupported engine kind {engine.kind!r}")
    ports = _assign_ports(engine, instance_ids, port_check or _port_available)
    return {
        instance_id: SGLangWorker(
            instance_id, cuda_visible_devices=instance_id, host=engine.host, port=ports[instance_id],
            python_executable=engine.python_executable, startup_timeout=engine.startup_timeout,
            request_timeout=engine.request_timeout,
        )
        for instance_id in instance_ids
    }


def _assign_ports(engine: EngineConfig, instance_ids: tuple[str, ...], port_check: PortCheck) -> dict[str, int]:
    ports: dict[str, int] = {}
    for index, instance_id in enumerate(instance_ids):
        port = engine.base_port + index
        if port > 65535:
            raise ConfigError(f"engine.base_port {engine.base_port} does not leave room for {len(instance_ids)} workers")
        if not port_check(engine.host, port):
            raise ConfigError(f"worker port {engine.host}:{port} is already in use")
        ports[instance_id] = port
    return ports


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


HYBRID_GEMM_ALIGNMENT = 256
ModelConfigLoader = Callable[[str], object]


def check_hybrid_gemm_compatibility(
    models: tuple[ModelConfig, ...], load_model_config: ModelConfigLoader | None = None,
) -> None:
    load = load_model_config or _load_hf_config
    problems = []
    for model in models:
        model_id = model.model_id
        try:
            config = load(model.model_path)
        except Exception as error:
            logger.warning("cannot read the config of %s (%s); skipping the HybridGEMM check", model_id, error)
            continue
        dimensions = _linear_dimensions(config)
        if dimensions is None:
            logger.warning("unknown architecture for %s; skipping the HybridGEMM check", model_id)
            continue
        bad = [f"{name}={value}" for name, value in dimensions if value % HYBRID_GEMM_ALIGNMENT]
        if bad:
            problems.append(f"{model_id} ({', '.join(bad)})")
    if problems:
        raise ConfigError(
            f"HybridGEMM needs every linear-layer dimension to be a multiple of {HYBRID_GEMM_ALIGNMENT}; "
            f"incompatible models: {'; '.join(problems)}"
        )


def _linear_dimensions(config) -> tuple[tuple[str, int], ...] | None:
    hidden = getattr(config, "hidden_size", None)
    intermediate = getattr(config, "intermediate_size", None)
    heads = getattr(config, "num_attention_heads", None)
    if not all(isinstance(value, int) and value > 0 for value in (hidden, intermediate, heads)):
        return None
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    return (
        ("hidden_size", hidden),
        ("intermediate_size", intermediate),
        ("attention_output", heads * head_dim),
        ("qkv_output", (heads + 2 * kv_heads) * head_dim),
    )


def _load_hf_config(model_path: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)
