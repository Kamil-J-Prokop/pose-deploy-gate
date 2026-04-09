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

adapter:
  type: "dummy"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.version == 1
    assert config.data.input_dir == input_dir
    assert config.adapter.type == "dummy"
    assert config.run.name == "default-run"
