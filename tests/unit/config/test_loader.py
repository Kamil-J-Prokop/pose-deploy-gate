from pathlib import Path

from pose_deploy_gate.config import load_config


def test_load_config_from_yaml():
    config = load_config("tests/fixtures/config/minimal_valid.yaml")

    assert config.version == 1
    assert config.data.input_dir == Path("data")
    assert config.adapter.type == "dummy"
    assert config.run.name == "default-run"
