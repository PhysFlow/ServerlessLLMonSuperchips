import json

import pytest

from memserve_control_plane.config import ConfigError, load_config


@pytest.mark.parametrize("section,field,value", [
    ("gateway", "port", value) for value in (0, 65536, True, "8000")
] + [
    ("engine", "base_port", value) for value in (0, 65536, True)
] + [
    ("gateway", "max_retries", value) for value in (-1, True, 1.5)
] + [
    ("gateway", "retry_delay", value) for value in (-1, True, float("nan"), float("inf"))
] + [
    ("engine", "startup_timeout", value) for value in (0, -1, True, float("nan"), float("inf"))
] + [
    ("gateway", "host", ""), ("engine", "host", None),
    ("engine", "python_executable", " "), ("engine", "kind", "vllm"),
])
def test_invalid_engine_and_gateway_fields(tmp_path, section, field, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": [{"model_id": "a", "model_path": "org/a"}],
                                section: {field: value}}))
    with pytest.raises(ConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize("payload", [None, [], {}, {"models": "a"}, {"models": [None]},
    {"models": [{"model_id": " ", "model_path": "x"}]},
    {"models": [{"model_id": "a"}]},
    {"models": [{"model_id": "a", "model_path": "x"}], "engine": []},
    {"models": [{"model_id": "a", "model_path": "x"}], "gateway": None},
])
def test_invalid_config_structure(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_and_malformed_file_have_config_error(tmp_path):
    path = tmp_path / "config.json"
    with pytest.raises(ConfigError, match="cannot load"):
        load_config(path)
    path.write_text("{broken")
    with pytest.raises(ConfigError, match="cannot load"):
        load_config(path)


def test_explicit_config_values_are_preserved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "models": [{"model_id": "a", "model_path": "/models/a", "resident": True}],
        "engine": {"python_executable": "/bin/python", "host": "localhost",
                   "base_port": 31000, "startup_timeout": 30},
        "gateway": {"host": "0.0.0.0", "port": 9000, "max_retries": 0, "retry_delay": 0.5},
    }))
    config = load_config(path)
    assert config.models[0].resident
    assert config.engine.python_executable == "/bin/python"
    assert config.engine.base_port == 31000
    assert config.engine.startup_timeout == 30
    assert (config.gateway.host, config.gateway.port, config.gateway.max_retries,
            config.gateway.retry_delay) == ("0.0.0.0", 9000, 0, 0.5)
