"""Training callbacks for PathoVision."""

import logging
import shutil
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Save model when a monitored metric improves.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric name to monitor.
        mode: 'max' or 'min' (whether higher or lower is better).
        save_top_k: Number of best checkpoints to keep.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_top_k: int = 1,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_checkpoints: list[tuple[float, Path]] = []

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def __call__(
        self,
        epoch: int,
        metrics: dict[str, float],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
    ) -> bool:
        """Check and save checkpoint if metric improved.

        Returns:
            True if checkpoint was saved.
        """
        value = metrics.get(self.monitor)
        if value is None:
            logger.warning("Metric '%s' not found in metrics", self.monitor)
            return False

        if not self.is_better(value):
            return False

        self.best_value = value

        # Save checkpoint
        ckpt_path = self.save_dir / f"best_epoch{epoch:03d}_{self.monitor}_{value:.4f}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "best_value": self.best_value,
        }
        torch.save(checkpoint, ckpt_path)

        # Also save as 'best.pt' for convenience
        best_path = self.save_dir / "best.pt"
        shutil.copy2(ckpt_path, best_path)

        # Manage top-k checkpoints
        self.best_checkpoints.append((value, ckpt_path))
        self.best_checkpoints.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )
        while len(self.best_checkpoints) > self.save_top_k:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists() and old_path != best_path:
                old_path.unlink()

        logger.info(
            "Checkpoint saved: %s = %.4f (epoch %d) -> %s",
            self.monitor, value, epoch, ckpt_path.name,
        )
        return True


class EarlyStopping:
    """Stop training when a metric stops improving.

    Args:
        monitor: Metric name to monitor.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' or 'min'.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 15,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def __call__(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    "Early stopping: %s did not improve for %d epochs (best=%.4f)",
                    self.monitor, self.patience, self.best_value,
                )
                return True

        return False


class MetricsLogger:
    """Log training metrics to CSV file.

    Args:
        log_path: Path to the CSV log file.
    """

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False

    def __call__(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log metrics for one epoch."""
        if not self.header_written:
            header = "epoch," + ",".join(sorted(metrics.keys()))
            with open(self.log_path, "w") as f:
                f.write(header + "\n")
            self.header_written = True

        values = ",".join(
            f"{metrics[k]:.6f}" for k in sorted(metrics.keys())
        )
        with open(self.log_path, "a") as f:
            f.write(f"{epoch},{values}\n")
