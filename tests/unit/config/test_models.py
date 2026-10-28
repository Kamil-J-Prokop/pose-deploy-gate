from pathlib import Path

import pytest
from pydantic import ValidationError

from pose_deploy_gate.config import AppConfig
from pose_deploy_gate.config.models import ModelConfig, RunnerConfig


def test_minimal_valid_config():
    config = AppConfig(
        version=1,
        data={"input_dir": "./data"},
        models=[{"id": "dummy", "adapter": {"type": "dummy"}}],
    )
    assert config.version == 1
    assert config.data.input_dir == Path("data")
    assert config.run.name == "default-run"
    assert config.models[0].adapter.type == "dummy"
    assert config.output.dir == Path("./artifacts")
    assert not config.gates.enabled
    assert config.runner == RunnerConfig()


def test_runner_config_uses_defaults() -> None:
    config = RunnerConfig()

    assert config.warmup_iterations == 3
    assert config.continue_on_error is False


def test_runner_config_accepts_zero_warmup_iterations() -> None:
    config = RunnerConfig(warmup_iterations=0)

    assert config.warmup_iterations == 0


def test_runner_config_rejects_negative_warmup_iterations() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RunnerConfig(warmup_iterations=-1)


def test_app_config_uses_default_runner_config() -> None:
    config = AppConfig(
        version=1,
        data={"input_dir": "./data"},
        models=[{"id": "dummy", "adapter": {"type": "dummy"}}],
    )

    assert config.runner == RunnerConfig()


def test_app_config_accepts_runner_config() -> None:
    config = AppConfig(
        version=1,
        data={"input_dir": "./data"},
        models=[{"id": "dummy", "adapter": {"type": "dummy"}}],
        runner={"warmup_iterations": 0, "continue_on_error": True},
    )

    assert config.runner == RunnerConfig(warmup_iterations=0, continue_on_error=True)


def _model(model_id: str, **params) -> dict:
    return {"id": model_id, "adapter": {"type": "dummy", "params": params}}


@pytest.mark.parametrize(
    "ids",
    [("yolo",), ("production", "candidate-a", "candidate-b"), ("model-a", "Model-A", " model-a ")],
)
def test_models_preserve_order_and_exact_ids(ids):
    config = AppConfig(
        version=1, data={"input_dir": "."}, models=[_model(model_id) for model_id in ids]
    )
    assert isinstance(config.models, tuple)
    assert tuple(model.id for model in config.models) == ids


def test_rejects_empty_models():
    with pytest.raises(ValidationError, match="at least 1 item"):
        AppConfig(version=1, data={"input_dir": "."}, models=[])


@pytest.mark.parametrize("model_id", ["", " ", "\t\n"])
def test_rejects_blank_model_id(model_id):
    with pytest.raises(ValidationError, match="model ID must not be blank"):
        AppConfig(version=1, data={"input_dir": "."}, models=[_model(model_id)])


@pytest.mark.parametrize("ids", [("same", "same"), ("first", "same", "same")])
def test_rejects_duplicate_model_ids(ids):
    with pytest.raises(ValidationError, match="model IDs must be unique"):
        AppConfig(version=1, data={"input_dir": "."}, models=[_model(model_id) for model_id in ids])


@pytest.mark.parametrize("include_models", [False, True])
def test_rejects_legacy_adapter_field(include_models):
    values = {"version": 1, "data": {"input_dir": "."}, "adapter": {"type": "dummy"}}
    if include_models:
        values["models"] = [_model("model")]
    with pytest.raises(ValidationError) as exc_info:
        AppConfig(**values)
    assert any(
        error["loc"] == ("adapter",) and error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_requires_models():
    with pytest.raises(ValidationError, match="models"):
        AppConfig(version=1, data={"input_dir": "."})


def test_loads_each_models_adapter_config():
    config = AppConfig(
        version=1,
        data={"input_dir": "."},
        models=[
            _model("production", pose_confidence=0.9),
            _model("candidate", pose_confidence=0.6),
        ],
    )
    assert config.models[0].adapter.type == "dummy"
    assert config.models[0].adapter.params == {"pose_confidence": 0.9}
    assert config.models[1].adapter.type == "dummy"
    assert config.models[1].adapter.params == {"pose_confidence": 0.6}


def test_model_config_has_no_role():
    model = ModelConfig(id="model", adapter={"type": "dummy"})
    assert model.adapter.params == {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelConfig(id="model", adapter={"type": "dummy"}, role="reference")
