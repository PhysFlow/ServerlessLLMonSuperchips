from types import SimpleNamespace

import pytest

from memserve_control_plane.config import ConfigError, EngineConfig, ModelConfig
from memserve_control_plane.engine import build_workers, check_hybrid_gemm_compatibility
from memserve_control_plane.types import WorkerLifecycleState


ENGINE = EngineConfig(
    kind="sglang", python_executable="python3",
    base_port=31000,
)


def test_each_mig_gets_one_stopped_worker_with_a_sequential_port():
    workers = build_workers(ENGINE, ("mig-a", "mig-b"), port_check=lambda host, port: True)

    assert workers["mig-a"]._base_url == "http://127.0.0.1:31000"
    assert workers["mig-b"]._base_url == "http://127.0.0.1:31001"
    assert workers["mig-a"].get_status().state is WorkerLifecycleState.STOPPED


def test_occupied_worker_port_is_rejected():
    with pytest.raises(ConfigError, match="already in use"):
        build_workers(ENGINE, ("mig-a",), port_check=lambda host, port: False)


SHAPES = {
    "tinyllama": dict(hidden_size=2048, intermediate_size=5632, num_attention_heads=32, num_key_value_heads=4),
    "qwen0.5b": dict(hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2),
    "qwen1.5b": dict(hidden_size=1536, intermediate_size=8960, num_attention_heads=12, num_key_value_heads=2),
    "qwen3b": dict(hidden_size=2048, intermediate_size=11008, num_attention_heads=16, num_key_value_heads=2),
}


def models(*model_ids):
    return tuple(ModelConfig(model_id, model_id) for model_id in model_ids)


def test_hybrid_gemm_compatible_models_pass():
    check_hybrid_gemm_compatibility(
        models("tinyllama", "qwen1.5b", "qwen3b"), lambda path: SimpleNamespace(**SHAPES[path]),
    )


def test_hybrid_gemm_incompatible_model_is_rejected_with_its_dimensions():
    with pytest.raises(ConfigError) as error:
        check_hybrid_gemm_compatibility(
            models("tinyllama", "qwen0.5b"), lambda path: SimpleNamespace(**SHAPES[path]),
        )
    message = str(error.value)
    assert "qwen0.5b (hidden_size=896, attention_output=896, qkv_output=1152)" in message
    assert "tinyllama" not in message


def test_unreadable_or_unknown_model_config_is_skipped():
    def load(path):
        if path == "missing":
            raise OSError("not in cache")
        return SimpleNamespace(model_type="mamba")

    check_hybrid_gemm_compatibility(models("missing", "unknown"), load)
