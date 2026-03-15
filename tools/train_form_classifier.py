"""
Form Quality Classifier Training Script

Trains and evaluates multiple scikit-learn classifiers on synthetic pose data
to find the best model for exercise form quality classification.

Models evaluated:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM with RBF kernel)
- Random Forest
- Gradient Boosting

The best model is selected based on cross-validation accuracy and saved
as a joblib artifact for use in the real-time pipeline.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.generate_test_data import (
    SQUAT_FEATURES, PUSHUP_FEATURES, LABELS,
    SQUAT_ERROR_TYPES, PUSHUP_ERROR_TYPES,
    generate_dataset,
)


def get_feature_columns(exercise: str) -> list[str]:
    """Get feature column names for an exercise type."""
    if exercise == "squat":
        return SQUAT_FEATURES
    return PUSHUP_FEATURES


def get_error_types(exercise: str) -> list[str]:
    """Get error type names for an exercise type."""
    if exercise == "squat":
        return SQUAT_ERROR_TYPES
    return PUSHUP_ERROR_TYPES


def train_and_evaluate(exercise: str, data_path: Path | None = None) -> dict:
    """
    Train multiple classifiers, evaluate via cross-validation,
    and save the best model.

    Args:
        exercise: "squat" or "pushup"
        data_path: optional path to pre-generated CSV data

    Returns:
        dict with evaluation results for all models
    """
    # Load or generate data
    if data_path and data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"Generating fresh {exercise} data...")
        df = generate_dataset(exercise, n_samples=3000)

    feature_cols = get_feature_columns(exercise)
    X = df[feature_cols].values
    y = df["form_quality"].values

    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Classes: {np.bincount(y)}")

    # Define classifiers to evaluate
    classifiers = {
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )),
        ]),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n{'='*60}")
    print(f"Evaluating classifiers for {exercise.upper()}")
    print(f"{'='*60}")

    for name, pipeline in classifiers.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "all_scores": scores.tolist(),
        }
        print(f"\n{name}:")
        print(f"  Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"  Per-fold: {[f'{s:.4f}' for s in scores]}")

    # Select best model
    best_name = max(results, key=lambda k: results[k]["mean_accuracy"])
    best_pipeline = classifiers[best_name]

    print(f"\n{'='*60}")
    print(f"Best model: {best_name} ({results[best_name]['mean_accuracy']:.4f})")
    print(f"{'='*60}")

    # Train final model on full dataset
    best_pipeline.fit(X, y)

    # Final evaluation on training set (for reporting)
    y_pred = best_pipeline.predict(X)
    print(f"\nClassification Report (full training set):")
    target_names = [LABELS[i] for i in sorted(LABELS.keys())]
    print(classification_report(y, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    # Feature importances (if available)
    clf = best_pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feature_importance = dict(zip(feature_cols, importances.tolist()))
        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
        print(f"\nFeature Importances:")
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.4f}")
        results["feature_importances"] = feature_importance

    # Save model
    model_dir = project_root / "backend" / "models" / "pretrained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{exercise}_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        "exercise": exercise,
        "best_model": best_name,
        "features": feature_cols,
        "labels": LABELS,
        "error_types": get_error_types(exercise),
        "training_samples": len(X),
        "cross_validation_results": results,
        "confusion_matrix": cm.tolist(),
    }

    metadata_path = model_dir / f"{exercise}_classifier_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return results


def main():
    """Train classifiers for all supported exercises."""
    data_dir = project_root / "backend" / "models" / "training_data"

    all_results = {}
    for exercise in ["squat", "pushup"]:
        data_path = data_dir / f"{exercise}_training_data.csv"
        results = train_and_evaluate(exercise, data_path)
        all_results[exercise] = results

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for exercise, results in all_results.items():
        best = max(results, key=lambda k: results[k]["mean_accuracy"]
                   if isinstance(results[k], dict) and "mean_accuracy" in results[k] else -1)
        acc = results[best]["mean_accuracy"]
        print(f"  {exercise}: {best} ({acc:.4f} accuracy)")


if __name__ == "__main__":
    main()
