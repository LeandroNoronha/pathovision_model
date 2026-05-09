"""ROC and Precision-Recall curve generation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> dict[str, float]:
    """Plot per-class ROC curves with AUC values.

    Args:
        y_true: Ground truth labels (N,).
        y_proba: Predicted probabilities (N, num_classes).
        class_names: Ordered class names.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary mapping class name to AUC value.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    auc_scores = {}

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")

    # Macro average
    macro_auc = np.mean(list(auc_scores.values()))
    auc_scores["macro_avg"] = macro_auc

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (Macro AUC = {macro_auc:.3f})", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curves saved to %s", save_path)

    plt.close(fig)
    return auc_scores


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> dict[str, float]:
    """Plot per-class Precision-Recall curves with Average Precision.

    Args:
        y_true: Ground truth labels (N,).
        y_proba: Predicted probabilities (N, num_classes).
        class_names: Ordered class names.
        save_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Dictionary mapping class name to AP value.
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=figsize)
    ap_scores = {}

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        ap_scores[cls] = ap
        ax.plot(recall, precision, color=color, lw=2, label=f"{cls} (AP = {ap:.3f})")

    macro_ap = np.mean(list(ap_scores.values()))
    ap_scores["macro_avg"] = macro_ap

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves (Macro AP = {macro_ap:.3f})", fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curves saved to %s", save_path)

    plt.close(fig)
    return ap_scores
