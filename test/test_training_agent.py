import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

from src.agents.training_agent import TrainingAgent, ExperimentResult
from src.training.config import ExperimentConfig


def _make_tiny_loaders(config):
    """Create tiny dataloaders for fast testing."""
    n_train, n_val = 16, 8
    train_images = torch.randn(n_train, config.channels, config.image_size, config.image_size)
    train_labels = torch.randint(0, config.num_classes, (n_train,))
    val_images = torch.randn(n_val, config.channels, config.image_size, config.image_size)
    val_labels = torch.randint(0, config.num_classes, (n_val,))

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=config.batch_size
    )
    val_loader = DataLoader(
        TensorDataset(val_images, val_labels), batch_size=config.batch_size
    )
    return train_loader, val_loader


@patch("src.agents.base.anthropic.Anthropic")
def test_run_experiment_end_to_end(mock_anthropic_cls):
    """Test run_experiment with minimal config (1 epoch, tiny data)."""
    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="test_agent_run"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("experiments", exist_ok=True)
        try:
            with patch(
                "src.agents.training_agent.get_dataloaders",
                return_value=_make_tiny_loaders(config),
            ):
                agent = TrainingAgent(device="cpu")
                result = agent.run_experiment(config)

                assert isinstance(result, ExperimentResult)
                assert result.experiment_name == "test_agent_run"
                assert "best_val_acc" in result.summary
                assert len(result.metrics_history) == 1
                assert os.path.isdir(result.output_dir)
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_list_experiments(mock_anthropic_cls):
    """Test that list_experiments scans the experiments directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            agent = TrainingAgent(device="cpu")

            # No experiments dir yet
            assert agent.list_experiments() == []

            # Create some experiment dirs
            os.makedirs("experiments/exp_a", exist_ok=True)
            os.makedirs("experiments/exp_b", exist_ok=True)

            result = agent.list_experiments()
            assert result == ["exp_a", "exp_b"]
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_get_experiment_result(mock_anthropic_cls):
    """Test loading a saved experiment result."""
    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="test_load_result"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("experiments", exist_ok=True)
        try:
            # First run an experiment to create saved files
            with patch(
                "src.agents.training_agent.get_dataloaders",
                return_value=_make_tiny_loaders(config),
            ):
                agent = TrainingAgent(device="cpu")
                original = agent.run_experiment(config)

                # Now load it back
                loaded = agent.get_experiment_result("test_load_result")

                assert loaded.experiment_name == original.experiment_name
                assert loaded.config == original.config
                assert loaded.summary == original.summary
                assert len(loaded.metrics_history) == len(original.metrics_history)
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_experiment_result_to_dict(mock_anthropic_cls):
    """Test that ExperimentResult serializes to dict."""
    result = ExperimentResult(
        experiment_name="test",
        config={"lr": 0.001},
        summary={"best_val_acc": 0.5},
        metrics_history=[{"epoch": 1}],
        output_dir="experiments/test",
    )
    d = result.to_dict()
    assert d["experiment_name"] == "test"
    assert d["config"] == {"lr": 0.001}
