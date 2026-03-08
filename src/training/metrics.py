import json


class MetricsTracker:
    """Tracks per-epoch training metrics and persists them as JSON."""

    def __init__(self):
        self.history = []

    def record(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Record metrics for a single epoch."""
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load(cls, path):
        tracker = cls()
        with open(path, "r") as f:
            tracker.history = json.load(f)
        return tracker

    def summary(self):
        """Return a summary dict with best epoch, final metrics, and convergence info."""
        if not self.history:
            return {}

        best = max(self.history, key=lambda e: e["val_acc"])
        final = self.history[-1]

        return {
            "best_epoch": best["epoch"],
            "best_val_acc": best["val_acc"],
            "best_val_loss": best["val_loss"],
            "final_train_loss": final["train_loss"],
            "final_val_loss": final["val_loss"],
            "final_train_acc": final["train_acc"],
            "final_val_acc": final["val_acc"],
            "total_epochs": len(self.history),
            "overfit_gap": final["train_acc"] - final["val_acc"],
        }
