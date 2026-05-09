"""Ensemble methods combining CNN and classical ML predictions."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def soft_voting(
    probabilities_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine predictions using weighted soft voting.

    Args:
        probabilities_list: List of probability arrays (N, num_classes) from different models.
        weights: Optional weights per model (must sum to 1).

    Returns:
        Tuple of (predicted_classes, averaged_probabilities).
    """
    n_models = len(probabilities_list)

    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        assert len(weights) == n_models
        total = sum(weights)
        weights = [w / total for w in weights]

    # Weighted average of probabilities
    avg_proba = np.zeros_like(probabilities_list[0])
    for proba, weight in zip(probabilities_list, weights):
        avg_proba += weight * proba

    predictions = avg_proba.argmax(axis=1)

    logger.info(
        "Soft voting: %d models, weights=%s",
        n_models, [f"{w:.2f}" for w in weights],
    )

    return predictions, avg_proba


def stacking(
    base_predictions: list[np.ndarray],
    y_train: np.ndarray,
    meta_learner: Any = None,
) -> Any:
    """Train a meta-learner on base model predictions (stacking).

    Args:
        base_predictions: List of prediction probability arrays from base models.
            Each has shape (N, num_classes).
        y_train: True labels for the training set.
        meta_learner: Scikit-learn classifier to use as meta-learner.
            Default: LogisticRegression.

    Returns:
        Trained meta-learner.
    """
    if meta_learner is None:
        from sklearn.linear_model import LogisticRegression
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    # Stack all base predictions as meta-features
    meta_features = np.hstack(base_predictions)

    logger.info(
        "Stacking: %d base models -> %d meta-features",
        len(base_predictions), meta_features.shape[1],
    )

    meta_learner.fit(meta_features, y_train)
    return meta_learner


def stacking_predict(
    meta_learner: Any,
    base_predictions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using a trained stacking ensemble.

    Args:
        meta_learner: Trained meta-learner.
        base_predictions: List of prediction probability arrays from base models.

    Returns:
        Tuple of (predicted_classes, probabilities).
    """
    meta_features = np.hstack(base_predictions)
    predictions = meta_learner.predict(meta_features)
    probabilities = (
        meta_learner.predict_proba(meta_features)
        if hasattr(meta_learner, "predict_proba")
        else None
    )
    return predictions, probabilities
