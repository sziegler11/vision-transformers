import torch
from torch.utils.data import TensorDataset
from unittest.mock import patch

from src.training.config import ExperimentConfig
from src.data.datasets import get_dataloaders


def _make_fake_cifar(root, train, download, transform):
    """Return a fake TensorDataset of 32 small images (already tensors)."""
    images = torch.randn(32, 3, 32, 32)
    labels = torch.arange(32) % 10
    return TensorDataset(images, labels)


@patch("src.data.datasets.datasets.CIFAR10", side_effect=_make_fake_cifar)
def test_get_dataloaders_shapes(mock_cifar):
    config = ExperimentConfig(batch_size=8, image_size=32, augmentation=False)
    train_loader, val_loader = get_dataloaders(config)

    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape[0] == 8
    assert batch_images.shape[1] == 3  # channels
    assert batch_labels.shape[0] == 8


@patch("src.data.datasets.datasets.CIFAR10", side_effect=_make_fake_cifar)
def test_get_dataloaders_deterministic(mock_cifar):
    config = ExperimentConfig(batch_size=8, seed=123, augmentation=False)
    loader1, _ = get_dataloaders(config)
    loader2, _ = get_dataloaders(config)

    batch1 = next(iter(loader1))[1]  # labels
    batch2 = next(iter(loader2))[1]
    assert torch.equal(batch1, batch2)


def test_unsupported_dataset():
    config = ExperimentConfig(dataset="imagenet")
    try:
        get_dataloaders(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "imagenet" in str(e)
