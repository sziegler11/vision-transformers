import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils import get_device


def get_dataloaders(config):
    """
    Build train and validation DataLoaders from the experiment config.

    Args:
        config: ExperimentConfig with dataset, image_size, batch_size,
                augmentation, device, and seed fields.

    Returns:
        (train_loader, val_loader) tuple of DataLoaders.
    """
    if config.dataset != "cifar10":
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    if config.augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    generator = torch.Generator().manual_seed(config.seed)

    device = get_device(getattr(config, "device", "auto"))
    use_pin_memory = device != "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
