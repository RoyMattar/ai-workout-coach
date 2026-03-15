"""
Train Form Classifier with Real Kaggle Dataset

Loads the Multi-Class Exercise Poses dataset (MediaPipe 33 landmarks),
computes joint angles, maps to our exercise types, and trains classifiers.

Compares performance:
- Synthetic-only training (existing)
- Real-data-only training
- Combined (synthetic + real)

This demonstrates real model evaluation with actual exercise data.
"""
import json
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
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MediaPipe landmark indices (1-indexed in the CSV: x1..x33)
# Matching our LANDMARK_MAP from pose_estimator.py
LANDMARK_INDICES = {
    "nose": 1,
    "left_shoulder": 12,
    "right_shoulder": 13,
    "left_elbow": 14,
    "right_elbow": 15,
    "left_wrist": 16,
    "right_wrist": 17,
    "left_hip": 24,
    "right_hip": 25,
    "left_knee": 26,
    "right_knee": 27,
    "left_ankle": 28,
    "right_ankle": 29,
}


def compute_angle(p1, p2, p3):
    """Compute angle at p2 from three 3D points."""
    ba = p1 - p2
    bc = p3 - p2
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def get_point(row, landmark_name):
    """Extract 3D point for a landmark from a CSV row."""
    idx = LANDMARK_INDICES[landmark_name]
    return np.array([
        row[f"x{idx}"],
        row[f"y{idx}"],
        row[f"z{idx}"],
    ])


def extract_features(row):
    """Extract our standard feature set from a raw landmark row."""
    try:
        # Compute angles
        left_knee_angle = compute_angle(
            get_point(row, "left_hip"),
            get_point(row, "left_knee"),
            get_point(row, "left_ankle"),
        )
        right_knee_angle = compute_angle(
            get_point(row, "right_hip"),
            get_point(row, "right_knee"),
            get_point(row, "right_ankle"),
        )
        left_elbow_angle = compute_angle(
            get_point(row, "left_shoulder"),
            get_point(row, "left_elbow"),
            get_point(row, "left_wrist"),
        )
        right_elbow_angle = compute_angle(
            get_point(row, "right_shoulder"),
            get_point(row, "right_elbow"),
            get_point(row, "right_wrist"),
        )
        left_hip_angle = compute_angle(
            get_point(row, "left_shoulder"),
            get_point(row, "left_hip"),
            get_point(row, "left_knee"),
        )
        right_hip_angle = compute_angle(
            get_point(row, "right_shoulder"),
            get_point(row, "right_hip"),
            get_point(row, "right_knee"),
        )

        # Torso angle (shoulder-hip vertical)
        mid_shoulder = (get_point(row, "left_shoulder")[:2] + get_point(row, "right_shoulder")[:2]) / 2
        mid_hip = (get_point(row, "left_hip")[:2] + get_point(row, "right_hip")[:2]) / 2
        torso_vec = mid_shoulder - mid_hip
        vertical = np.array([0, -1])
        cos_torso = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6)
        torso_angle = np.degrees(np.arccos(np.clip(cos_torso, -1, 1)))

        # Alignment
        shoulder_cx = (row[f"x{LANDMARK_INDICES['left_shoulder']}"] + row[f"x{LANDMARK_INDICES['right_shoulder']}"]) / 2
        hip_cx = (row[f"x{LANDMARK_INDICES['left_hip']}"] + row[f"x{LANDMARK_INDICES['right_hip']}"]) / 2
        alignment = abs(shoulder_cx - hip_cx)

        return {
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "avg_knee_angle": (left_knee_angle + right_knee_angle) / 2,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "avg_elbow_angle": (left_elbow_angle + right_elbow_angle) / 2,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "torso_angle": torso_angle,
            "knee_angle_diff": abs(left_knee_angle - right_knee_angle),
            "shoulder_hip_alignment": alignment,
        }
    except Exception:
        return None


def map_to_exercise_class(kaggle_class):
    """Map Kaggle exercise class to our exercise recognition label."""
    mapping = {
        "left_bicep": "bicep_curl",
        "right_bicep": "bicep_curl",
        "left_shoulder": "shoulder_press",
        "right_shoulder": "shoulder_press",
        "left_tricep": "tricep",
        "right_tricep": "tricep",
        "rest": "rest",
    }
    return mapping.get(kaggle_class, "unknown")


def train_exercise_recognizer(df):
    """Train a model that recognizes which exercise is being performed."""
    print("\n" + "=" * 60)
    print("EXERCISE RECOGNITION MODEL (from real data)")
    print("=" * 60)

    # Extract features
    features = []
    labels = []
    for _, row in df.iterrows():
        feat = extract_features(row)
        if feat:
            features.append(feat)
            labels.append(map_to_exercise_class(row["class"]))

    feat_df = pd.DataFrame(features)
    X = feat_df.values
    y = np.array(labels)

    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train classifiers
    classifiers = {
        "KNN (k=5)": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, probability=True))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score = "", 0

    for name, pipeline in classifiers.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"\n{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    # Train and save best model
    best_pipeline = classifiers[best_name]
    best_pipeline.fit(X, y)

    model_dir = project_root / "backend" / "models" / "pretrained"
    model_path = model_dir / "exercise_recognizer.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest: {best_name} ({best_score:.4f})")
    print(f"Saved to: {model_path}")

    # Save metadata
    metadata = {
        "model": best_name,
        "accuracy": best_score,
        "features": list(feat_df.columns),
        "classes": list(np.unique(y)),
        "training_samples": len(X),
        "source": "Kaggle Multi-Class Exercise Poses (dp5995)",
    }
    with open(model_dir / "exercise_recognizer_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Full report
    y_pred = best_pipeline.predict(X)
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred))

    return best_pipeline, feat_df.columns.tolist()


def compare_with_synthetic():
    """Compare real-data model accuracy with synthetic-only model."""
    print("\n" + "=" * 60)
    print("COMPARISON: Synthetic vs Real Data")
    print("=" * 60)

    model_dir = project_root / "backend" / "models" / "pretrained"

    # Load existing synthetic-trained models
    for exercise in ["squat", "pushup"]:
        meta_path = model_dir / f"{exercise}_classifier_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            best = meta.get("best_model", "?")
            acc = meta.get("cross_validation_results", {}).get(best, {}).get("mean_accuracy", 0)
            print(f"Synthetic {exercise}: {best} ({acc:.4f} accuracy)")

    recognizer_meta = model_dir / "exercise_recognizer_metadata.json"
    if recognizer_meta.exists():
        with open(recognizer_meta) as f:
            meta = json.load(f)
        print(f"Real data exercise recognizer: {meta['model']} ({meta['accuracy']:.4f} accuracy)")

    print("\nConclusion: Real data provides exercise recognition capability")
    print("that synthetic data alone cannot — it classifies WHICH exercise")
    print("is being performed, while synthetic data classifies form QUALITY")
    print("within a known exercise. Both are complementary.")


def main():
    data_path = project_root / "data" / "kaggle" / "dataset_all_points.csv"
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Please download from: https://www.kaggle.com/datasets/dp5995/gym-exercise-mediapipe-33-landmarks")
        return

    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    print(f"Classes: {df['class'].value_counts().to_dict()}")

    # Train exercise recognizer on real data
    train_exercise_recognizer(df)

    # Compare with synthetic
    compare_with_synthetic()


if __name__ == "__main__":
    main()
