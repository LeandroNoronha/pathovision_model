"""Train a PathoVision model.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/improved.yaml
    python scripts/train.py --config configs/base.yaml training.epochs=100 training.loss.name=focal
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import create_dataloaders
from src.data.sampler import compute_class_weights
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.utils.config import apply_cli_overrides, load_config
from src.utils.device import get_device, print_gpu_info
from src.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PathoVision model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides: key=value")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config = load_config(args.config)
    if args.overrides:
        config = apply_cli_overrides(config, args.overrides)

    # Reproducibility
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    # GPU info
    print_gpu_info()

    # Data
    print("\n=== Loading Data ===")
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    dataloaders = create_dataloaders(config, train_transform, val_transform)

    # Model
    print("\n=== Building Model ===")
    model = build_model(config)

    # Loss
    class_weights = None
    weight_cfg = config["training"]["loss"].get("class_weights")
    if weight_cfg and weight_cfg != "null":
        class_weights = compute_class_weights(
            dataloaders["train"].dataset,
            method=weight_cfg if isinstance(weight_cfg, str) else "inverse_frequency",
        )
        device = get_device()
        class_weights = class_weights.to(device)

    criterion = build_loss(config, class_weights)

    # Optimizer
    train_cfg = config["training"]
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    ) if opt_cfg["name"] == "adamw" else torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    # Scheduler (optional cosine)
    scheduler = None
    sched_cfg = train_cfg.get("scheduler", {})
    if sched_cfg.get("name") == "cosine_warmup":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"] - sched_cfg.get("warmup_epochs", 0),
            eta_min=sched_cfg.get("min_lr", 1e-7),
        )

    # Train
    print("\n=== Starting Training ===")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    best_metrics = trainer.train()

    print("\n=== Training Complete ===")
    print(f"Best metrics: {best_metrics}")
    print(f"Checkpoint saved to: {config['experiment']['output_dir']}/checkpoints/best.pt")


if __name__ == "__main__":
    main()
