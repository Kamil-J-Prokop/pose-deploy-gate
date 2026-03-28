from pathlib import Path

from pose_deploy_gate.config import AppConfig


def test_minimal_valid_config():
    config = AppConfig(version=1, data={"input_dir": "./data"}, adapter={"type": "dummy"})
    assert config.version == 1
    assert config.data.input_dir == Path("data")
    assert config.run.name == "default-run"
    assert config.adapter.type == "dummy"
    assert config.output.dir == Path("./artifacts")
    assert not config.gates.enabled
