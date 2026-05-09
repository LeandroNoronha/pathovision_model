"""Main training loop for PathoVision models."""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import EarlyStopping, MetricsLogger, ModelCheckpoint
from src.training.mixup import mixup_criterion, mixup_data
from src.utils.device import get_amp_dtype, get_device

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator with mixed precision, gradient accumulation, and callbacks.

    Args:
        model: The model to train.
        config: Full configuration dictionary.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (optional).
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training settings
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.accumulation_steps = config["data"].get("accumulation_steps", 1)
        self.use_mixed_precision = train_cfg.get("mixed_precision", True)
        self.gradient_clip = train_cfg.get("gradient_clip")
        self.use_mixup = train_cfg.get("mixup", {}).get("enabled", False)
        self.mixup_alpha = train_cfg.get("mixup", {}).get("alpha", 0.2)

        # Mixed precision
        self.amp_dtype = get_amp_dtype(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_mixed_precision)

        # Callbacks
        self._setup_callbacks()

        # State
        self.current_epoch = 0
        self.best_metrics: dict[str, float] = {}

    def _setup_callbacks(self) -> None:
        """Initialize callbacks from config."""
        cb_cfg = self.config.get("callbacks", {})
        output_dir = Path(self.config["experiment"].get("output_dir", "outputs"))

        # Checkpoint
        ckpt_cfg = cb_cfg.get("checkpoint", {})
        self.checkpoint = None
        if ckpt_cfg.get("enabled", True):
            self.checkpoint = ModelCheckpoint(
                save_dir=output_dir / "checkpoints",
                monitor=ckpt_cfg.get("monitor", "val_accuracy"),
                mode=ckpt_cfg.get("mode", "max"),
                save_top_k=ckpt_cfg.get("save_top_k", 1),
            )

        # Early stopping
        es_cfg = cb_cfg.get("early_stopping", {})
        self.early_stopping = None
        if es_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode=es_cfg.get("mode", "min"),
            )

        # ReduceLROnPlateau
        lr_cfg = cb_cfg.get("reduce_lr", {})
        self.reduce_lr = None
        if lr_cfg.get("enabled", True) and self.scheduler is None:
            self.reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if lr_cfg.get("monitor", "val_loss") == "val_loss" else "max",
                factor=lr_cfg.get("factor", 0.2),
                patience=lr_cfg.get("patience", 5),
                min_lr=lr_cfg.get("min_lr", 1e-7),
            )

        # Metrics logger
        self.metrics_logger = MetricsLogger(output_dir / "logs" / "training_log.csv")

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Best metrics achieved during training.
        """
        logger.info("Starting training for %d epochs", self.epochs)
        logger.info("Device: %s, Mixed Precision: %s", self.device, self.use_mixed_precision)
        logger.info("Batch size: %d x %d accumulation = %d effective",
                     self.config["data"]["batch_size"], self.accumulation_steps,
                     self.config["data"]["batch_size"] * self.accumulation_steps)

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train epoch
            train_metrics = self._train_epoch()

            # Validate epoch
            val_metrics = self._validate_epoch()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start
            metrics["epoch_time"] = epoch_time

            # Log
            self._log_epoch(epoch, metrics)
            self.metrics_logger(epoch, metrics)

            # Callbacks
            if self.checkpoint:
                self.checkpoint(epoch, metrics, self.model, self.optimizer, self.config)

            if self.reduce_lr:
                monitor_value = metrics.get("val_loss", metrics.get("train_loss"))
                if monitor_value is not None:
                    self.reduce_lr.step(monitor_value)

            if self.scheduler:
                self.scheduler.step()

            if self.early_stopping and self.early_stopping(metrics):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # Load best checkpoint
        if self.checkpoint:
            best_path = Path(self.config["experiment"]["output_dir"]) / "checkpoints" / "best.pt"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.best_metrics = ckpt["metrics"]
                logger.info("Loaded best model from epoch %d", ckpt["epoch"])

        return self.best_metrics

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Optional Mixup
            if self.use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_mixed_precision):
                logits = self.model(images)
                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % self.accumulation_steps == 0:
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            if not self.use_mixup:
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item() * self.accumulation_steps:.4f}")

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {"train_loss": avg_loss, "train_accuracy": accuracy}

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def _log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log epoch metrics."""
        lr = metrics.get("lr", 0)
        epoch_time = metrics.get("epoch_time", 0)
        logger.info(
            "Epoch %d/%d - "
            "train_loss: %.4f, train_acc: %.4f - "
            "val_loss: %.4f, val_acc: %.4f - "
            "lr: %.2e - time: %.1fs",
            epoch, self.epochs,
            metrics.get("train_loss", 0), metrics.get("train_accuracy", 0),
            metrics.get("val_loss", 0), metrics.get("val_accuracy", 0),
            lr, epoch_time,
        )
