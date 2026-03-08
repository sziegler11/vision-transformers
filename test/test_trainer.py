import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.vit import VisionTransfomer
from src.training.config import ExperimentConfig
from src.training.trainer import Trainer


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


def _make_model(config):
    return VisionTransfomer(
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


def test_trainer_runs_without_error():
    config = ExperimentConfig(
        num_epochs=2, batch_size=8, experiment_name="test_train_run"
    )
    model = _make_model(config)
    train_loader, val_loader = _make_tiny_loaders(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override experiments dir by changing cwd temporarily
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("experiments", exist_ok=True)
        try:
            trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
            tracker = trainer.train()
            assert len(tracker.history) == 2
        finally:
            os.chdir(orig_cwd)


def test_trainer_saves_checkpoint_files():
    config = ExperimentConfig(
        num_epochs=1, batch_size=8, experiment_name="test_checkpoint"
    )
    model = _make_model(config)
    train_loader, val_loader = _make_tiny_loaders(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("experiments", exist_ok=True)
        try:
            trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
            trainer.train()

            exp_dir = os.path.join("experiments", "test_checkpoint")
            assert os.path.exists(os.path.join(exp_dir, "config.json"))
            assert os.path.exists(os.path.join(exp_dir, "metrics.json"))
            assert os.path.exists(os.path.join(exp_dir, "best_model.pt"))
            assert os.path.exists(os.path.join(exp_dir, "summary.json"))
        finally:
            os.chdir(orig_cwd)


def test_trainer_metrics_recorded_correctly():
    config = ExperimentConfig(
        num_epochs=2, batch_size=8, experiment_name="test_metrics_rec"
    )
    model = _make_model(config)
    train_loader, val_loader = _make_tiny_loaders(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs("experiments", exist_ok=True)
        try:
            trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
            tracker = trainer.train()

            for entry in tracker.history:
                assert "epoch" in entry
                assert "train_loss" in entry
                assert "val_loss" in entry
                assert "train_acc" in entry
                assert "val_acc" in entry
                assert 0.0 <= entry["train_acc"] <= 1.0
                assert 0.0 <= entry["val_acc"] <= 1.0
        finally:
            os.chdir(orig_cwd)


def test_trainer_optimizer_variants():
    """Test that all three optimizer options work."""
    for opt_name in ["adam", "adamw", "sgd"]:
        config = ExperimentConfig(
            num_epochs=1, batch_size=8, optimizer=opt_name,
            experiment_name=f"test_opt_{opt_name}"
        )
        model = _make_model(config)
        train_loader, val_loader = _make_tiny_loaders(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs("experiments", exist_ok=True)
            try:
                trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
                tracker = trainer.train()
                assert len(tracker.history) == 1
            finally:
                os.chdir(orig_cwd)


def test_trainer_scheduler_variants():
    """Test that all scheduler options work."""
    for sched_name in ["cosine", "step", "none"]:
        config = ExperimentConfig(
            num_epochs=2, batch_size=8, scheduler=sched_name,
            experiment_name=f"test_sched_{sched_name}"
        )
        model = _make_model(config)
        train_loader, val_loader = _make_tiny_loaders(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            os.chdir(tmpdir)
            os.makedirs("experiments", exist_ok=True)
            try:
                trainer = Trainer(model, config, train_loader, val_loader, device="cpu")
                tracker = trainer.train()
                assert len(tracker.history) == 2
            finally:
                os.chdir(orig_cwd)
