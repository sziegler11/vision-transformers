import json
import os
import tempfile
from unittest.mock import patch, MagicMock

from src.agents.analysis_agent import (
    AnalysisAgent,
    AnalysisReport,
    ComparisonReport,
    ANALYSIS_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    SUGGESTION_SYSTEM_PROMPT,
)
from src.training.config import ExperimentConfig
from src.training.metrics import MetricsTracker


def _mock_text_response(text):
    """Create a mock Messages API response with a text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _create_experiment_files(tmpdir, name, config=None, epochs=2):
    """Create experiment files on disk for testing."""
    if config is None:
        config = ExperimentConfig(
            num_epochs=epochs, batch_size=8, experiment_name=name
        )

    exp_dir = os.path.join(tmpdir, "experiments", name)
    os.makedirs(exp_dir, exist_ok=True)

    config.save(os.path.join(exp_dir, "config.json"))

    tracker = MetricsTracker()
    for e in range(1, epochs + 1):
        tracker.record(e, 2.0 - e * 0.1, 2.1 - e * 0.05, 0.1 * e, 0.08 * e)
    tracker.save(os.path.join(exp_dir, "metrics.json"))

    summary = tracker.summary()
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f)

    return config, tracker


@patch("src.agents.base.anthropic.Anthropic")
def test_analyze_experiment(mock_anthropic_cls):
    """Test that analyze_experiment parses LLM response into AnalysisReport."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    analysis_response = {
        "findings": ["Model is converging steadily", "No severe overfitting"],
        "diagnosed_issues": ["Learning rate may be slightly high"],
        "recommendations": ["Try reducing learning rate by 50%"],
        "overall_assessment": "Decent first run with room for improvement.",
    }
    mock_client.messages.create.return_value = _mock_text_response(
        json.dumps(analysis_response)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            _create_experiment_files(tmpdir, "test_exp")

            agent = AnalysisAgent()
            report = agent.analyze_experiment("test_exp")

            assert isinstance(report, AnalysisReport)
            assert report.experiment_name == "test_exp"
            assert len(report.findings) == 2
            assert len(report.diagnosed_issues) == 1
            assert len(report.recommendations) == 1
            assert "improvement" in report.overall_assessment

            # Verify the prompt contains experiment data
            call_kwargs = mock_client.messages.create.call_args.kwargs
            user_msg = call_kwargs["messages"][0]["content"]
            assert "test_exp" in user_msg
            assert "learning_rate" in user_msg
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_compare_experiments(mock_anthropic_cls):
    """Test that compare_experiments produces ComparisonReport from mock response."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    comparison_response = {
        "ranking": [
            {"experiment_name": "exp_a", "best_val_acc": 0.65, "rank": 1},
            {"experiment_name": "exp_b", "best_val_acc": 0.55, "rank": 2},
        ],
        "helpful_hyperparams": ["Lower learning rate", "AdamW optimizer"],
        "harmful_hyperparams": ["High dropout with small model"],
        "overall_summary": "exp_a outperformed exp_b due to better LR tuning.",
    }
    mock_client.messages.create.return_value = _mock_text_response(
        json.dumps(comparison_response)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            _create_experiment_files(tmpdir, "exp_a")
            _create_experiment_files(tmpdir, "exp_b")

            agent = AnalysisAgent()
            report = agent.compare_experiments(["exp_a", "exp_b"])

            assert isinstance(report, ComparisonReport)
            assert report.experiment_names == ["exp_a", "exp_b"]
            assert len(report.ranking) == 2
            assert report.ranking[0]["rank"] == 1
            assert len(report.helpful_hyperparams) == 2
            assert len(report.harmful_hyperparams) == 1

            # Verify prompt contains both experiments
            call_kwargs = mock_client.messages.create.call_args.kwargs
            user_msg = call_kwargs["messages"][0]["content"]
            assert "exp_a" in user_msg
            assert "exp_b" in user_msg
        finally:
            os.chdir(orig_cwd)


@patch("src.agents.base.anthropic.Anthropic")
def test_suggest_next_experiment(mock_anthropic_cls):
    """Test that suggest_next_experiment returns a valid ExperimentConfig."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client

    suggestion_response = {
        "image_size": 32,
        "patch_size": 8,
        "embed_dim": 128,
        "num_heads": 8,
        "num_blocks": 4,
        "mlp_dim": 256,
        "dropout": 0.1,
        "num_classes": 10,
        "channels": 3,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "num_epochs": 15,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "dataset": "cifar10",
        "augmentation": True,
        "experiment_name": "suggested_exp",
        "seed": 42,
    }
    mock_client.messages.create.return_value = _mock_text_response(
        json.dumps(suggestion_response)
    )

    history = [
        {
            "config": ExperimentConfig(experiment_name="prev_exp").to_dict(),
            "summary": {"best_val_acc": 0.5, "best_epoch": 3},
        }
    ]

    agent = AnalysisAgent()
    config = agent.suggest_next_experiment(history)

    assert isinstance(config, ExperimentConfig)
    assert config.patch_size == 8
    assert config.embed_dim == 128
    assert config.learning_rate == 0.0005
    assert config.experiment_name == "suggested_exp"

    # Verify prompt contains history data
    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "prev_exp" in user_msg
    assert "best_val_acc" in user_msg


@patch("src.agents.base.anthropic.Anthropic")
def test_analysis_report_to_dict(mock_anthropic_cls):
    """Test AnalysisReport serialization."""
    report = AnalysisReport(
        experiment_name="test",
        findings=["f1"],
        diagnosed_issues=["d1"],
        recommendations=["r1"],
        overall_assessment="good",
    )
    d = report.to_dict()
    assert d["experiment_name"] == "test"
    assert d["findings"] == ["f1"]


@patch("src.agents.base.anthropic.Anthropic")
def test_comparison_report_to_dict(mock_anthropic_cls):
    """Test ComparisonReport serialization."""
    report = ComparisonReport(
        experiment_names=["a", "b"],
        ranking=[],
        helpful_hyperparams=["x"],
        harmful_hyperparams=["y"],
        overall_summary="summary",
    )
    d = report.to_dict()
    assert d["experiment_names"] == ["a", "b"]
    assert d["overall_summary"] == "summary"
