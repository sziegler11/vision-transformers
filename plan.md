# Plan: Agent-Based Vision Transformer Training System

## Goal
Transform this inference-only ViT repo into an AI-agent-driven system where LLM-powered agents can:
1. Kick off training experiments with configurable hyperparameters
2. Analyze results (loss curves, accuracy, metrics)
3. Reason about what went wrong/right and formulate improved follow-up experiments

## Architecture Overview

```
vision-transformers/
├── src/
│   ├── models/
│   │   └── vit.py                    # (existing, minor fixes)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop engine
│   │   ├── config.py                 # Experiment configuration dataclass
│   │   └── metrics.py                # Metric tracking and persistence
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py              # Data loading (CIFAR-10 for demo)
│   └── agents/
│       ├── __init__.py
│       ├── base.py                   # Base agent with Claude API integration
│       ├── experiment_agent.py       # Orchestrator agent (LLM-powered)
│       ├── training_agent.py         # Kicks off & monitors training
│       └── analysis_agent.py         # LLM-powered analysis & suggestion
├── experiments/                      # Runtime dir for experiment outputs (gitignored)
├── test/
│   ├── test_vit.py                   # (existing)
│   ├── test_trainer.py
│   ├── test_config.py
│   ├── test_metrics.py
│   ├── test_datasets.py
│   ├── test_training_agent.py
│   ├── test_analysis_agent.py
│   └── test_experiment_agent.py
└── requirements.txt                  # Updated deps
```

---

## Implementation Steps

### Step 1: Fix existing model for training readiness
**File:** `src/models/vit.py`
- Thread `dropout` parameter through `VisionTransfomer` constructor
- Fix device placement for positional embeddings (`torch.arange(..., device=x.device)`)
- Keep class name typo as-is per CLAUDE.md

### Step 2: Experiment configuration (`src/training/config.py`)
- `ExperimentConfig` dataclass with all hyperparameters:
  - Model params: `image_size`, `patch_size`, `embed_dim`, `num_heads`, `num_blocks`, `mlp_dim`, `dropout`, `num_classes`
  - Training params: `learning_rate`, `batch_size`, `num_epochs`, `weight_decay`, `optimizer` (adam/adamw/sgd), `scheduler` (cosine/step/none)
  - Data params: `dataset` (cifar10 for now), `augmentation` (bool)
  - Meta: `experiment_name`, `seed`
- `to_dict()` / `from_dict()` for JSON serialization
- `save(path)` / `load(path)` to persist configs alongside results

### Step 3: Data loading (`src/data/datasets.py`)
- `get_dataloaders(config) -> (train_loader, val_loader)` function
- Use `torchvision.datasets.CIFAR10` with standard transforms (resize, normalize, optional augmentation)
- Support configurable batch size and train/val split
- Deterministic splitting via seed

### Step 4: Metric tracking (`src/training/metrics.py`)
- `MetricsTracker` class:
  - `record(epoch, train_loss, val_loss, train_acc, val_acc)` — stores per-epoch metrics
  - `save(path)` / `load(path)` — JSON persistence
  - `summary() -> dict` — best epoch, final metrics, convergence info
- Simple JSON-based storage (no external dependencies like wandb)

### Step 5: Training loop (`src/training/trainer.py`)
- `Trainer` class:
  - `__init__(model, config, train_loader, val_loader, device)`
  - `train() -> MetricsTracker` — full training loop
  - Handles: optimizer setup, LR scheduling, train/eval modes, metric recording
  - Saves checkpoints (best model by val accuracy) and final metrics
  - Each experiment writes to `experiments/<experiment_name>/` with:
    - `config.json` — experiment config
    - `metrics.json` — per-epoch metrics
    - `best_model.pt` — best checkpoint
    - `summary.json` — final summary

### Step 6: Base agent with Claude API (`src/agents/base.py`)
- `BaseAgent` class:
  - Wraps the Anthropic Python SDK (`anthropic` package)
  - `__init__(model="claude-sonnet-4-20250514")` — configurable model
  - `_call_llm(system_prompt, user_message) -> str` — sends a message to Claude API, returns text response
  - `_call_llm_json(system_prompt, user_message, schema) -> dict` — calls Claude and parses structured JSON from the response
  - Handles API key via `ANTHROPIC_API_KEY` env var
  - Includes retry logic for transient API errors

### Step 7: Training agent (`src/agents/training_agent.py`)
- `TrainingAgent(BaseAgent)` class:
  - `run_experiment(config: ExperimentConfig) -> ExperimentResult`
    - Builds model from config, gets dataloaders, runs trainer
    - Returns structured result with metrics and paths
  - `list_experiments() -> list[str]` — scans experiments/ dir
  - `get_experiment_result(name) -> ExperimentResult` — loads saved results
  - No LLM calls needed here — this agent is purely execution-focused

### Step 8: Analysis agent (`src/agents/analysis_agent.py`) — LLM-powered
- `AnalysisAgent(BaseAgent)` class:
  - `analyze_experiment(name) -> AnalysisReport`
    - Loads metrics, formats them as a prompt for Claude
    - Claude analyzes: convergence behavior, overfitting signals, learning rate adequacy, model capacity
    - Returns structured `AnalysisReport` with findings and diagnosed issues
  - `compare_experiments(names: list[str]) -> ComparisonReport`
    - Sends all experiment configs + metrics to Claude
    - Claude produces a ranked comparison identifying which hyperparams helped/hurt
  - `suggest_next_experiment(history: list[ExperimentResult]) -> ExperimentConfig`
    - Sends full experiment history (configs + results + analyses) to Claude
    - Claude reasons about what to try next and returns a new `ExperimentConfig` as structured JSON
    - The LLM can make creative, non-obvious suggestions (e.g., "try a much smaller patch size with fewer heads" or "the model is too small for this task, double embed_dim and num_blocks")

### Step 9: Orchestrator agent (`src/agents/experiment_agent.py`) — LLM-powered
- `ExperimentAgent(BaseAgent)` class — ties everything together:
  - `run_search(base_config, num_iterations=5, goal="maximize validation accuracy")`
    - Iteration loop:
      1. Run experiment via `TrainingAgent`
      2. Analyze results via `AnalysisAgent`
      3. Ask Claude (via `AnalysisAgent.suggest_next_experiment`) to propose next config
      4. Claude sees full history and reasons about the next experiment
      5. Repeat
    - Maintains experiment history across iterations
  - `run_single(config)` — run one experiment and analyze
  - `report() -> str` — asks Claude to produce a human-readable summary of all experiments, key findings, and the best configuration found

### Step 10: Update dependencies
**File:** `requirements.txt`
- Add `torchvision==0.17.2` (matches torch 2.2.2)
- Add `anthropic>=0.42.0` (Claude API SDK)

### Step 11: Tests

**`test/test_config.py`:**
- Test ExperimentConfig creation with defaults
- Test serialization round-trip (to_dict/from_dict)
- Test save/load to disk

**`test/test_datasets.py`:**
- Test get_dataloaders returns correct types and batch shapes
- Test deterministic splitting with same seed

**`test/test_metrics.py`:**
- Test MetricsTracker recording and retrieval
- Test save/load round-trip
- Test summary computation (best epoch, final metrics)

**`test/test_trainer.py`:**
- Test Trainer runs 1-2 epochs on tiny data without error
- Test checkpoint saving produces expected files
- Test metrics are recorded correctly

**`test/test_training_agent.py`:**
- Test run_experiment end-to-end with minimal config (1 epoch, tiny data)
- Test list_experiments and get_experiment_result

**`test/test_analysis_agent.py`:**
- Mock the Claude API calls (using `unittest.mock.patch` on the anthropic client)
- Test analyze_experiment parses mock LLM response into AnalysisReport
- Test compare_experiments produces ComparisonReport from mock response
- Test suggest_next_experiment returns a valid ExperimentConfig from mock JSON response
- Test that prompts sent to Claude contain the right experiment data

**`test/test_experiment_agent.py`:**
- Mock Claude API calls throughout
- Test run_single end-to-end with minimal config (1 epoch, tiny data, mocked LLM)
- Test run_search for 2 iterations with mocked suggestions
- Test report output format

All tests mock the Anthropic API so they run without an API key and without network access. Tests should complete in under 60 seconds total (tiny models, 1-2 epochs, CPU).

---

## Key Design Decisions

1. **LLM-powered agents via Claude API** — The analysis and orchestration agents call Claude to reason about experiment results and suggest improvements. This gives agents genuine intelligence: they can notice patterns, make creative suggestions, and explain their reasoning — far beyond what rule-based heuristics can do.

2. **Structured output from LLM** — Agents ask Claude to return JSON matching defined schemas. This ensures LLM suggestions can be programmatically consumed (e.g., turned into an `ExperimentConfig` and run automatically).

3. **Separation of execution and reasoning** — `TrainingAgent` is pure execution (no LLM needed). `AnalysisAgent` is pure reasoning (LLM-powered). `ExperimentAgent` orchestrates the loop between them. This makes each component independently testable.

4. **Mocked tests** — All LLM calls are mocked in tests, so the test suite runs fast, deterministically, and without API keys. Integration testing with real API calls is left to manual runs.

5. **JSON-based persistence** — No database or MLflow dependency. Everything is plain files in `experiments/`, easy to inspect and version.

6. **CIFAR-10 as default dataset** — Small enough for CPU training demos, complex enough to show ViT behavior.

7. **Minimal new dependencies** — Only adding `torchvision` and `anthropic`. No pandas, no matplotlib, no wandb.

8. **Deterministic experiments** — Seed-based reproducibility for the training runs (LLM suggestions are inherently non-deterministic).
