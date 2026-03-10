import logging

from src.agents.base import BaseAgent
from src.agents.training_agent import TrainingAgent
from src.agents.analysis_agent import AnalysisAgent
from src.training.config import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentAgent(BaseAgent):
    """Orchestrator agent that ties training and analysis together.

    Runs iterative experiment loops: train -> analyze -> suggest -> repeat.
    Uses LLM calls for generating human-readable reports.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.training_agent = TrainingAgent(device=device, **kwargs)
        self.analysis_agent = AnalysisAgent(**kwargs)
        self.experiment_history = []

    def run_single(self, config: ExperimentConfig) -> dict:
        """Run one experiment and analyze it.

        Args:
            config: The experiment configuration.

        Returns:
            A dict with 'result' (ExperimentResult) and 'analysis' (AnalysisReport).
        """
        logger.info(f"Running single experiment: {config.experiment_name}")

        result = self.training_agent.run_experiment(config)
        analysis = self.analysis_agent.analyze_experiment(config.experiment_name)

        entry = {
            "result": result,
            "analysis": analysis,
            "config": result.config,
            "summary": result.summary,
        }
        self.experiment_history.append(entry)

        return entry

    def run_search(self, base_config: ExperimentConfig, num_iterations: int = 5,
                   goal: str = "maximize validation accuracy"):
        """Orchestrate an iterative experiment search.

        Runs a loop of: train -> analyze -> suggest next config -> repeat.

        Args:
            base_config: The initial experiment configuration.
            num_iterations: Number of experiment iterations to run.
            goal: The optimization goal (passed to the suggestion prompt).

        Returns:
            List of dicts, each with 'result' and 'analysis' keys.
        """
        logger.info(
            f"Starting experiment search: {num_iterations} iterations, goal='{goal}'"
        )

        current_config = base_config

        for i in range(num_iterations):
            logger.info(f"Iteration {i + 1}/{num_iterations}")

            entry = self.run_single(current_config)

            if i < num_iterations - 1:
                history_for_suggestion = [
                    {"config": e["config"], "summary": e["summary"]}
                    for e in self.experiment_history
                ]
                next_config = self.analysis_agent.suggest_next_experiment(
                    history_for_suggestion
                )
                next_config.experiment_name = f"{base_config.experiment_name}_iter{i + 2}"
                current_config = next_config

        return self.experiment_history

    def report(self) -> str:
        """Generate a human-readable summary of all experiments using Claude.

        Returns:
            A string with the LLM-generated report.
        """
        if not self.experiment_history:
            return "No experiments have been run yet."

        experiments_summary = []
        for entry in self.experiment_history:
            experiments_summary.append({
                "config": entry["config"],
                "summary": entry["summary"],
                "analysis_findings": entry["analysis"].findings,
                "analysis_assessment": entry["analysis"].overall_assessment,
            })

        import json
        system_prompt = (
            "You are an expert ML researcher. Summarize the following series of "
            "Vision Transformer experiments. Highlight key findings, the best "
            "configuration found, and actionable next steps."
        )
        user_message = json.dumps(experiments_summary, indent=2)

        return self._call_llm(system_prompt, user_message)
