import json
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a ViT training experiment."""

    # Model params
    image_size: int = 32
    patch_size: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_blocks: int = 2
    mlp_dim: int = 64
    dropout: float = 0.2
    num_classes: int = 10
    channels: int = 3

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 10
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, none

    # Data params
    dataset: str = "cifar10"
    augmentation: bool = True

    # Meta
    experiment_name: str = "default"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        # Filter to only known fields to be robust to extra keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
