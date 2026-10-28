from pathlib import Path

from pose_deploy_gate.config import load_config


def test_load_config_from_yaml(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    config_path = tmp_path / "minimal_valid.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{input_dir}"

models:
  - id: dummy
    adapter:
      type: "dummy"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.version == 1
    assert config.data.input_dir == input_dir
    assert config.models[0].adapter.type == "dummy"
    assert config.run.name == "default-run"


def test_load_config_with_multiple_models() -> None:
    config = load_config(Path("tests/fixtures/config/multiple_models.yaml"))
    assert tuple(model.id for model in config.models) == (
        "production",
        "candidate-a",
        "candidate-b",
    )
    assert tuple(model.adapter.type for model in config.models) == ("dummy",) * 3
    assert tuple(model.adapter.params for model in config.models) == (
        {"pose_confidence": 0.9},
        {"pose_confidence": 0.7},
        {},
    )
