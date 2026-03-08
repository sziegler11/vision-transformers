# AGENTS.md — Vision Transformer Training System

## Project Purpose

An AI-agent-driven Vision Transformer (ViT) training system built on PyTorch. This repository provides a complete pipeline for running, tracking, and analyzing ViT training experiments, designed to be orchestrated by LLM-powered agents that can reason about results and iteratively improve hyperparameters.

Based on: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)

---

## Repository Structure

```
vision-transformers/
├── src/
│   ├── models/
│   │   └── vit.py                # Vision Transformer implementation
│   ├── training/
│   │   ├── config.py             # ExperimentConfig dataclass
│   │   ├── trainer.py            # Training loop engine
│   │   └── metrics.py            # Metric tracking and persistence
│   ├── data/
│   │   └── datasets.py           # Data loading (CIFAR-10)
│   └── agents/                   # (planned) LLM-powered experiment agents
├── experiments/                   # Runtime output directory for experiment results
├── test/                          # Test suite
└── requirements.txt
```

---

## Core Components

### Model (`src/models/vit.py`)
Full ViT implementation with patch embedding, multi-head self-attention, transformer encoder, and classification head. Supports configurable dropout and GPU device placement.

**Note:** The main class is `VisionTransfomer` (typo preserved for backward compatibility).

### Experiment Configuration (`src/training/config.py`)
`ExperimentConfig` dataclass covering model architecture, training hyperparameters, data settings, and experiment metadata. Supports JSON serialization via `to_dict()`/`from_dict()` and file persistence via `save()`/`load()`.

### Data Loading (`src/data/datasets.py`)
`get_dataloaders(config)` builds train/validation DataLoaders from CIFAR-10 with configurable batch size, image augmentation, and deterministic seeding.

### Metrics Tracking (`src/training/metrics.py`)
`MetricsTracker` records per-epoch train/val loss and accuracy, persists to JSON, and computes summaries (best epoch, overfitting gap, convergence info).

### Trainer (`src/training/trainer.py`)
`Trainer` class runs the full training loop with configurable optimizer (Adam, AdamW, SGD), LR scheduler (cosine, step, none), checkpoint saving, and metric recording. Each experiment outputs to `experiments/<name>/` with config, metrics, model checkpoint, and summary files.

---

## Getting Started

```bash
pip install -r requirements.txt
pip install pytest

# Run tests
pytest test/

# Example: train a small ViT on CIFAR-10
python -c "
from src.models.vit import VisionTransfomer
from src.training.config import ExperimentConfig
from src.training.trainer import Trainer
from src.data.datasets import get_dataloaders

config = ExperimentConfig(num_epochs=5, experiment_name='quickstart')
model = VisionTransfomer(
    image_size=config.image_size, patch_size=config.patch_size,
    embed_dim=config.embed_dim, num_heads=config.num_heads,
    num_blocks=config.num_blocks, num_classes=config.num_classes,
    mlp_dim=config.mlp_dim, dropout=config.dropout,
)
train_loader, val_loader = get_dataloaders(config)
trainer = Trainer(model, config, train_loader, val_loader)
tracker = trainer.train()
print(tracker.summary())
"
```

---

## Dependencies

| Package      | Version | Role                        |
|--------------|---------|-----------------------------|
| torch        | 2.2.2   | Neural network framework    |
| torchvision  | 0.17.2  | Datasets and transforms     |
| numpy        | 1.26.4  | Numerical ops (via PyTorch) |

---

## Running Tests

```bash
pytest test/
```

Tests run on CPU with tiny synthetic data and complete in seconds. Dataset tests mock CIFAR-10 downloads so no network access is needed.

---

## Experiment Output Format

Each experiment creates `experiments/<experiment_name>/` containing:
- `config.json` — full experiment configuration
- `metrics.json` — per-epoch train/val loss and accuracy
- `best_model.pt` — model checkpoint at best validation accuracy
- `summary.json` — best epoch, final metrics, overfitting gap

---

## Git Workflow

- Default branch: `master`
- Feature branches: `claude/<description>-<id>`
- Remote: `origin`
