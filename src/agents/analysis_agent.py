import json
import logging
import os
from dataclasses import dataclass, field, asdict

from src.agents.base import BaseAgent
from src.training.config import ExperimentConfig
from src.training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """Structured report from analyzing a single experiment."""

    experiment_name: str
    findings: list
    diagnosed_issues: list
    recommendations: list
    overall_assessment: str

    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonReport:
    """Structured report from comparing multiple experiments."""

    experiment_names: list
    ranking: list
    helpful_hyperparams: list
    harmful_hyperparams: list
    overall_summary: str

    def to_dict(self):
        return asdict(self)


ANALYSIS_SYSTEM_PROMPT = """You are an expert machine learning researcher analyzing Vision Transformer training experiments.
Given experiment configuration and per-epoch metrics, analyze:
1. Convergence behavior (is the model learning? at what rate?)
2. Overfitting signals (train vs val gap, val loss increasing)
3. Learning rate adequacy (too high = instable, too low = slow convergence)
4. Model capacity (underfitting suggests too small, overfitting suggests too large or needs regularization)

Provide concrete, actionable findings."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "diagnosed_issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "overall_assessment": {"type": "string"},
    },
    "required": ["findings", "diagnosed_issues", "recommendations", "overall_assessment"],
    "additionalProperties": False,
}

COMPARISON_SYSTEM_PROMPT = """You are an expert machine learning researcher comparing multiple Vision Transformer training experiments.
Given configs and metrics for several experiments, produce a ranked comparison identifying which hyperparameters helped and which hurt performance."""

COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string"},
                    "best_val_acc": {"type": "number"},
                    "rank": {"type": "integer"},
                },
                "required": ["experiment_name", "best_val_acc", "rank"],
                "additionalProperties": False,
            },
        },
        "helpful_hyperparams": {
            "type": "array",
            "items": {"type": "string"},
        },
        "harmful_hyperparams": {
            "type": "array",
            "items": {"type": "string"},
        },
        "overall_summary": {"type": "string"},
    },
    "required": ["ranking", "helpful_hyperparams", "harmful_hyperparams", "overall_summary"],
    "additionalProperties": False,
}

SUGGESTION_SYSTEM_PROMPT = """You are an expert machine learning researcher designing the next Vision Transformer experiment.
Based on the full history of experiments (configs and results), reason about what to try next to maximize validation accuracy.
Return a complete experiment configuration as JSON. Be creative — consider non-obvious changes like different patch sizes, model scaling, learning rate adjustments, or regularization changes."""

SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_size": {"type": "integer"},
        "patch_size": {"type": "integer"},
        "embed_dim": {"type": "integer"},
        "num_heads": {"type": "integer"},
        "num_blocks": {"type": "integer"},
        "mlp_dim": {"type": "integer"},
        "dropout": {"type": "number"},
        "num_classes": {"type": "integer"},
        "channels": {"type": "integer"},
        "learning_rate": {"type": "number"},
        "batch_size": {"type": "integer"},
        "num_epochs": {"type": "integer"},
        "weight_decay": {"type": "number"},
        "optimizer": {"type": "string"},
        "scheduler": {"type": "string"},
        "dataset": {"type": "string"},
        "augmentation": {"type": "boolean"},
        "experiment_name": {"type": "string"},
        "seed": {"type": "integer"},
    },
    "required": [
        "image_size", "patch_size", "embed_dim", "num_heads", "num_blocks",
        "mlp_dim", "dropout", "num_classes", "learning_rate", "batch_size",
        "num_epochs", "weight_decay", "optimizer", "scheduler", "augmentation",
        "experiment_name", "seed",
    ],
    "additionalProperties": False,
}


class AnalysisAgent(BaseAgent):
    """LLM-powered agent that analyzes experiment results and suggests improvements."""

    def analyze_experiment(self, name: str) -> AnalysisReport:
        """Analyze a single experiment's results using Claude.

        Args:
            name: Experiment name (subdirectory under experiments/).

        Returns:
            An AnalysisReport with findings, issues, and recommendations.
        """
        output_dir = os.path.join("experiments", name)
        config = ExperimentConfig.load(os.path.join(output_dir, "config.json"))
        tracker = MetricsTracker.load(os.path.join(output_dir, "metrics.json"))

        user_message = (
            f"Experiment: {name}\n\n"
            f"Configuration:\n{json.dumps(config.to_dict(), indent=2)}\n\n"
            f"Per-epoch metrics:\n{json.dumps(tracker.history, indent=2)}\n\n"
            f"Summary:\n{json.dumps(tracker.summary(), indent=2)}"
        )

        result = self._call_llm_json(
            ANALYSIS_SYSTEM_PROMPT, user_message, ANALYSIS_SCHEMA
        )

        return AnalysisReport(
            experiment_name=name,
            findings=result["findings"],
            diagnosed_issues=result["diagnosed_issues"],
            recommendations=result["recommendations"],
            overall_assessment=result["overall_assessment"],
        )

    def compare_experiments(self, names: list) -> ComparisonReport:
        """Compare multiple experiments using Claude.

        Args:
            names: List of experiment names to compare.

        Returns:
            A ComparisonReport with ranking and hyperparameter insights.
        """
        experiments_data = []
        for name in names:
            output_dir = os.path.join("experiments", name)
            config = ExperimentConfig.load(os.path.join(output_dir, "config.json"))
            tracker = MetricsTracker.load(os.path.join(output_dir, "metrics.json"))
            experiments_data.append({
                "name": name,
                "config": config.to_dict(),
                "summary": tracker.summary(),
            })

        user_message = (
            f"Experiments to compare:\n\n"
            f"{json.dumps(experiments_data, indent=2)}"
        )

        result = self._call_llm_json(
            COMPARISON_SYSTEM_PROMPT, user_message, COMPARISON_SCHEMA
        )

        return ComparisonReport(
            experiment_names=names,
            ranking=result["ranking"],
            helpful_hyperparams=result["helpful_hyperparams"],
            harmful_hyperparams=result["harmful_hyperparams"],
            overall_summary=result["overall_summary"],
        )

    def suggest_next_experiment(self, history: list) -> ExperimentConfig:
        """Use Claude to suggest the next experiment based on full history.

        Args:
            history: List of dicts, each with 'config' and 'summary' keys
                     from previous experiments.

        Returns:
            A new ExperimentConfig suggested by Claude.
        """
        user_message = (
            f"Experiment history (configs and results):\n\n"
            f"{json.dumps(history, indent=2)}\n\n"
            f"Based on this history, suggest the next experiment configuration "
            f"to maximize validation accuracy."
        )

        result = self._call_llm_json(
            SUGGESTION_SYSTEM_PROMPT, user_message, SUGGESTION_SCHEMA
        )

        return ExperimentConfig.from_dict(result)
