"""Classical ML classifiers for the hybrid CNN+ML pipeline.

Uses CNN-extracted features as input to train SVM, Random Forest,
XGBoost, and LightGBM classifiers.
"""

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def get_classifier(name: str, num_classes: int = 7, **kwargs: Any) -> Any:
    """Create a classical ML classifier.

    Args:
        name: Classifier name ('svm', 'random_forest', 'xgboost', 'lightgbm').
        num_classes: Number of classes.
        **kwargs: Additional classifier parameters.

    Returns:
        Scikit-learn compatible classifier.
    """
    if name == "svm":
        return SVC(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 10.0),
            gamma=kwargs.get("gamma", "scale"),
            class_weight="balanced",
            probability=True,
            random_state=42,
        )

    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", None),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        return XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            objective="multi:softprob",
            num_class=num_classes,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        return LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", -1),
            learning_rate=kwargs.get("learning_rate", 0.1),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    else:
        raise ValueError(f"Unknown classifier: {name}")


def train_classifier(
    classifier: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    normalize: bool = True,
) -> tuple[Any, StandardScaler | None]:
    """Train a classical ML classifier on extracted features.

    Args:
        classifier: Scikit-learn compatible classifier.
        X_train: Training features (N, D).
        y_train: Training labels (N,).
        normalize: Whether to normalize features first.

    Returns:
        Tuple of (trained_classifier, scaler).
    """
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    logger.info("Training %s on %d samples, %d features",
                type(classifier).__name__, X_train.shape[0], X_train.shape[1])

    classifier.fit(X_train, y_train)

    logger.info("Training complete")
    return classifier, scaler


def predict_classifier(
    classifier: Any,
    X: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with a trained classifier.

    Args:
        classifier: Trained classifier.
        X: Input features (N, D).
        scaler: Optional scaler (must match training).

    Returns:
        Tuple of (predictions, probabilities).
    """
    if scaler is not None:
        X = scaler.transform(X)

    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X) if hasattr(classifier, "predict_proba") else None

    return predictions, probabilities
