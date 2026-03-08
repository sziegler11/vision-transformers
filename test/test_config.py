import json
import os
import tempfile

from src.training.config import ExperimentConfig


def test_default_creation():
    config = ExperimentConfig()
    assert config.image_size == 32
    assert config.learning_rate == 1e-3
    assert config.optimizer == "adamw"
    assert config.dataset == "cifar10"
    assert config.seed == 42


def test_custom_creation():
    config = ExperimentConfig(
        image_size=64, num_epochs=5, experiment_name="test_run"
    )
    assert config.image_size == 64
    assert config.num_epochs == 5
    assert config.experiment_name == "test_run"


def test_to_dict_from_dict_roundtrip():
    config = ExperimentConfig(learning_rate=0.01, experiment_name="roundtrip")
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["learning_rate"] == 0.01

    restored = ExperimentConfig.from_dict(d)
    assert restored == config


def test_from_dict_ignores_extra_keys():
    d = ExperimentConfig().to_dict()
    d["unknown_key"] = "should_be_ignored"
    config = ExperimentConfig.from_dict(d)
    assert not hasattr(config, "unknown_key")


def test_save_load_roundtrip():
    config = ExperimentConfig(batch_size=128, experiment_name="save_test")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        config.save(path)

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["batch_size"] == 128

        loaded = ExperimentConfig.load(path)
        assert loaded == config
