"""Train classical ML classifiers on CNN-extracted features.

Usage:
    python scripts/train_hybrid.py --features-dir outputs/features
    python scripts/train_hybrid.py --features-dir outputs/features --classifiers svm xgboost
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.metrics import compute_full_report
from src.hybrid.classifiers import get_classifier, predict_classifier, train_classifier
from src.hybrid.ensemble import soft_voting
from src.hybrid.feature_extractor import load_features


CLASS_NAMES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid CNN+ML classifiers")
    parser.add_argument("--features-dir", type=str, default="outputs/features")
    parser.add_argument("--classifiers", nargs="+",
                        default=["svm", "random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--output-dir", type=str, default="results/10_hybrid")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("Loading extracted features...")
    X_train, y_train = load_features(features_dir / "train_features.npz")
    X_test, y_test = load_features(features_dir / "test_features.npz")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train and evaluate each classifier
    all_probas = []
    results = {}

    for clf_name in args.classifiers:
        print(f"\n{'='*60}")
        print(f"Training: {clf_name}")
        print(f"{'='*60}")

        try:
            clf = get_classifier(clf_name, num_classes=len(CLASS_NAMES))
            clf, scaler = train_classifier(clf, X_train, y_train)
            y_pred, y_proba = predict_classifier(clf, X_test, scaler)

            report = compute_full_report(y_test, y_pred, CLASS_NAMES)

            print(f"\n{clf_name} Results:")
            print(f"  Global Accuracy: {report['global']['global_accuracy']:.2f}%")
            print(f"  Weighted F1:     {report['global']['f1_weighted']:.2f}%")
            print(f"\n{report['classification_report']}")

            results[clf_name] = report["global"]
            if y_proba is not None:
                all_probas.append(y_proba)

        except ImportError as e:
            print(f"  Skipping {clf_name}: {e}")

    # Ensemble (soft voting)
    if len(all_probas) >= 2:
        print(f"\n{'='*60}")
        print("Ensemble (Soft Voting)")
        print(f"{'='*60}")

        y_ensemble, ensemble_proba = soft_voting(all_probas)
        report = compute_full_report(y_test, y_ensemble, CLASS_NAMES)

        print(f"  Global Accuracy: {report['global']['global_accuracy']:.2f}%")
        print(f"  Weighted F1:     {report['global']['f1_weighted']:.2f}%")
        results["ensemble"] = report["global"]

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Classifier':<20} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['global_accuracy']:>9.2f}% {metrics['f1_weighted']:>14.2f}%")


if __name__ == "__main__":
    main()
