import os
import tempfile

from src.training.metrics import MetricsTracker


def test_record_and_history():
    tracker = MetricsTracker()
    tracker.record(1, 2.0, 2.5, 0.3, 0.25)
    tracker.record(2, 1.5, 2.0, 0.5, 0.4)
    assert len(tracker.history) == 2
    assert tracker.history[0]["epoch"] == 1
    assert tracker.history[1]["train_loss"] == 1.5


def test_save_load_roundtrip():
    tracker = MetricsTracker()
    tracker.record(1, 2.0, 2.5, 0.3, 0.25)
    tracker.record(2, 1.5, 2.0, 0.5, 0.4)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metrics.json")
        tracker.save(path)

        loaded = MetricsTracker.load(path)
        assert len(loaded.history) == 2
        assert loaded.history[0] == tracker.history[0]
        assert loaded.history[1] == tracker.history[1]


def test_summary_best_epoch():
    tracker = MetricsTracker()
    tracker.record(1, 2.0, 2.5, 0.3, 0.25)
    tracker.record(2, 1.5, 2.0, 0.5, 0.45)
    tracker.record(3, 1.0, 2.2, 0.7, 0.40)

    summary = tracker.summary()
    assert summary["best_epoch"] == 2
    assert summary["best_val_acc"] == 0.45
    assert summary["total_epochs"] == 3
    assert summary["final_train_acc"] == 0.7
    assert summary["final_val_acc"] == 0.40
    assert summary["overfit_gap"] == 0.7 - 0.40


def test_summary_empty():
    tracker = MetricsTracker()
    assert tracker.summary() == {}
