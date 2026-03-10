import json
import logging
import os
from dataclasses import dataclass, asdict

from src.agents.base import BaseAgent
from src.data.datasets import get_dataloaders
from src.models.vit import VisionTransfomer
from src.training.config import ExperimentConfig
from src.training.metrics import MetricsTracker
from src.training.trainer import Trainer
from src.utils import get_device

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Structured result from a completed experiment."""

    experiment_name: str
    config: dict
    summary: dict
    metrics_history: list
    output_dir: str

    def to_dict(self):
        return asdict(self)


class TrainingAgent(BaseAgent):
    """Agent that executes training experiments.

    This agent is purely execution-focused — no LLM calls are needed.
    It builds models, runs training, and manages experiment results.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = get_device(device)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Build model from config, train it, and return structured results.

        Args:
            config: The experiment configuration specifying model architecture,
                    training hyperparameters, and data settings.

        Returns:
            An ExperimentResult with metrics, summary, and output paths.
        """
        logger.info(f"Starting experiment: {config.experiment_name}")

        model = VisionTransfomer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_blocks=config.num_blocks,
            num_classes=config.num_classes,
            mlp_dim=config.mlp_dim,
            channels=config.channels,
            dropout=config.dropout,
        )

        train_loader, val_loader = get_dataloaders(config)

        trainer = Trainer(model, config, train_loader, val_loader, device=self.device)
        tracker = trainer.train()

        output_dir = os.path.join("experiments", config.experiment_name)

        return ExperimentResult(
            experiment_name=config.experiment_name,
            config=config.to_dict(),
            summary=tracker.summary(),
            metrics_history=tracker.history,
            output_dir=output_dir,
        )

    def list_experiments(self) -> list:
        """Scan the experiments/ directory and return experiment names."""
        exp_dir = "experiments"
        if not os.path.isdir(exp_dir):
            return []
        return sorted(
            name
            for name in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, name))
        )

    def get_experiment_result(self, name: str) -> ExperimentResult:
        """Load a saved experiment result by name.

        Args:
            name: The experiment name (subdirectory under experiments/).

        Returns:
            An ExperimentResult loaded from saved files.

        Raises:
            FileNotFoundError: If the experiment directory or required files don't exist.
        """
        output_dir = os.path.join("experiments", name)

        config_path = os.path.join(output_dir, "config.json")
        metrics_path = os.path.join(output_dir, "metrics.json")
        summary_path = os.path.join(output_dir, "summary.json")

        config = ExperimentConfig.load(config_path)
        tracker = MetricsTracker.load(metrics_path)

        with open(summary_path, "r") as f:
            summary = json.load(f)

        return ExperimentResult(
            experiment_name=name,
            config=config.to_dict(),
            summary=summary,
            metrics_history=tracker.history,
            output_dir=output_dir,
        )
