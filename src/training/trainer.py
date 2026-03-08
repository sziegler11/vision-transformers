import json
import os

import torch
from torch import nn

from src.training.metrics import MetricsTracker


class Trainer:
    """Training loop engine for ViT experiments."""

    def __init__(self, model, config, train_loader, val_loader, device=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_optimizer(self):
        name = self.config.optimizer.lower()
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {name}")

    def _build_scheduler(self, optimizer):
        name = self.config.scheduler.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_epochs
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, self.config.num_epochs // 3), gamma=0.1
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {name}")

    def _run_epoch(self, loader, optimizer=None):
        """Run one epoch. If optimizer is provided, train; otherwise evaluate."""
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.set_grad_enabled(is_train):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = criterion(logits, labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def train(self):
        """Run the full training loop and return a MetricsTracker."""
        torch.manual_seed(self.config.seed)

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        tracker = MetricsTracker()

        output_dir = os.path.join("experiments", self.config.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        self.config.save(os.path.join(output_dir, "config.json"))

        best_val_acc = -1.0

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, optimizer)
            val_loss, val_acc = self._run_epoch(self.val_loader)

            if scheduler is not None:
                scheduler.step()

            tracker.record(epoch, train_loss, val_loss, train_acc, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, "best_model.pt"),
                )

        tracker.save(os.path.join(output_dir, "metrics.json"))

        summary = tracker.summary()
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return tracker
