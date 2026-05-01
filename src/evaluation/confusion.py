"""Confusion matrix generation and visualization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes, n_classes).
        class_names: List of class names.
        save_path: Path to save the figure (optional).
        title: Plot title.
        normalize: If True, normalize by row (true class).
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fmt = ".1f"
        title = f"{title} (Normalized %)"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={"label": "Count" if not normalize else "Percentage (%)"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: list[str],
    save_dir: str | Path,
    experiment_name: str = "pathovision",
) -> None:
    """Generate both absolute and normalized confusion matrices.

    Args:
        cm: Confusion matrix array.
        class_names: List of class names.
        save_dir: Directory to save figures.
        experiment_name: Prefix for filenames.
    """
    save_dir = Path(save_dir)

    plot_confusion_matrix(
        cm, class_names,
        save_path=save_dir / f"{experiment_name}_cm_absolute.png",
        title=f"{experiment_name} - Confusion Matrix",
        normalize=False,
    )

    plot_confusion_matrix(
        cm, class_names,
        save_path=save_dir / f"{experiment_name}_cm_normalized.png",
        title=f"{experiment_name} - Confusion Matrix",
        normalize=True,
    )

    plt.close("all")
