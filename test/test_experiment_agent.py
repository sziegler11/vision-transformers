import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agents.experiment_agent import ExperimentAgent
from src.agents.analysis_agent import AnalysisReport
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


def _mock_text_response(text):
    """Create a mock Messages API response with a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_analysis_response():
    """Create a mock analysis JSON response."""
    return {
        "findings": ["Model is learning"],
        "diagnosed_issues": [],
        "recommendations": ["Continue training"],
        "overall_assessment": "Good progress.",
    }


def _make_suggestion_response(iteration):
    """Create a mock suggestion JSON response for a given iteration."""
    return {
        "image_size": 32,
        "patch_size": 16,
        "embed_dim": 64,
        "num_heads": 4,
        "num_blocks": 2,
        "mlp_dim": 64,
        "dropout": 0.1,
        "num_classes": 10,
        "channels": 3,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "num_epochs": 1,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "dataset": "cifar10",
        "augmentation": True,
        "experiment_name": f"suggested_iter{iteration}",
        "seed": 42,
    }


@patch("src.agents.base.anthropic.Anthropic")
def test_run_single(mock_anthropic_cls):
    """Test run_single runs one experiment and analyzes it."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    # Mock the analysis LLM call
    mock_client.messages.create.return_value = _mock_text_response(
        json.dumps(_make_analysis_response())
    )

    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="test_single"
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
                agent = ExperimentAgent(device="cpu")
                entry = agent.run_single(config)

                assert "result" in entry
                assert "analysis" in entry
                assert entry["result"].experiment_name == "test_single"
                assert isinstance(entry["analysis"], AnalysisReport)
                assert len(agent.experiment_history) == 1
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_run_search_two_iterations(mock_anthropic_cls):
    """Test run_search for 2 iterations with mocked suggestions."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    # The LLM will be called multiple times:
    # 1. analyze_experiment for iter 1
    # 2. suggest_next_experiment after iter 1
    # 3. analyze_experiment for iter 2
    call_count = [0]
    responses = [
        _make_analysis_response(),       # analysis for iter 1
        _make_suggestion_response(2),    # suggestion after iter 1
        _make_analysis_response(),       # analysis for iter 2
    ]

    def side_effect(**kwargs):
        idx = min(call_count[0], len(responses) - 1)
        resp = responses[idx]
        call_count[0] += 1
        return _mock_text_response(json.dumps(resp))

    mock_client.messages.create.side_effect = side_effect

    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="search_test"
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
                agent = ExperimentAgent(device="cpu")
                history = agent.run_search(config, num_iterations=2)

                assert len(history) == 2
                assert history[0]["result"].experiment_name == "search_test"
                assert history[1]["result"].experiment_name == "search_test_iter2"
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_report_output(mock_anthropic_cls):
    """Test that report generates text output via LLM."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    # First call: analysis response, Second call: report text
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] <= 1:
            return _mock_text_response(json.dumps(_make_analysis_response()))
        else:
            return _mock_text_response(
                "# Experiment Report\n\nThe best configuration achieved 10% accuracy."
            )

    mock_client.messages.create.side_effect = side_effect

    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="test_report"
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
                agent = ExperimentAgent(device="cpu")
                agent.run_single(config)
                report = agent.report()

                assert isinstance(report, str)
                assert len(report) > 0
                assert "Experiment Report" in report
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_report_empty_history(mock_anthropic_cls):
    """Test that report handles empty history gracefully."""
    agent = ExperimentAgent(device="cpu")
    report = agent.report()
    assert report == "No experiments have been run yet."
