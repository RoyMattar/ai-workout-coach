"""
Train Real-Data Form Classifiers for All Exercises

For exercises with labeled correct/wrong video data, trains a supervised
classifier. For exercises with only unlabeled videos, trains using the
tuned rule-based system as a labeling oracle (knowledge distillation).

This produces per-exercise KNN models saved as {exercise}_form_realdata.joblib.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from backend.pose_estimator import PoseEstimator
from backend.exercises.squat import SquatAnalyzer
from backend.exercises.pushup import PushupAnalyzer
from backend.exercises.deadlift import DeadliftAnalyzer
from backend.exercises.bicep_curl import BicepCurlAnalyzer
from backend.exercises.shoulder_press import ShoulderPressAnalyzer
from backend.exercises.plank import PlankAnalyzer

UNIFIED_FEATURES = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_elbow_angle", "right_elbow_angle",
    "torso_angle", "shoulder_hip_alignment",
]

EXERCISE_ANALYZERS = {
    "squat": SquatAnalyzer,
    "pushup": PushupAnalyzer,
    "deadlift": DeadliftAnalyzer,
    "bicep_curl": BicepCurlAnalyzer,
    "shoulder_press": ShoulderPressAnalyzer,
    "plank": PlankAnalyzer,
}

# Map video folder names to our exercise types
FOLDER_EXERCISE_MAP = {
    "squat": "squat",
    "push-up": "pushup",
    "deadlift": "deadlift",
    "romanian deadlift": "deadlift",
    "barbell biceps curl": "bicep_curl",
    "hammer curl": "bicep_curl",
    "shoulder press": "shoulder_press",
    "plank": "plank",
}


def extract_frame_features(frame, pose_estimator):
    """Extract unified angle features from a single frame."""
    result = pose_estimator.process_frame(frame)
    if not result.is_valid:
        return None
    return {f: result.angles.get(f, 170.0 if "angle" in f else 0.0) for f in UNIFIED_FEATURES}


def extract_video_features(video_path, pose_estimator, max_frames=100):
    """Extract features from all sampled frames of a video."""
    cap = cv2.VideoCapture(str(video_path))
    features = []
    fc = 0
    while fc < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fc += 1
        if fc % 3 != 0:
            continue
        feat = extract_frame_features(frame, pose_estimator)
        if feat:
            features.append(feat)
    cap.release()
    return features


def label_with_rules(features_list, exercise_type, analyzer_class):
    """Label frames using the rule-based analyzer as oracle."""
    from backend.pose_estimator import PoseResult, Landmark

    analyzer = analyzer_class()
    labels = []

    for feat in features_list:
        # Create a PoseResult from angles (no actual landmarks needed for angle-based checks)
        # We create minimal landmarks for the pose validation
        dummy_landmarks = {
            "left_shoulder": Landmark(0.55, 0.25, 0, 0.9),
            "right_shoulder": Landmark(0.45, 0.25, 0, 0.9),
            "left_hip": Landmark(0.53, 0.50, 0, 0.9),
            "right_hip": Landmark(0.47, 0.50, 0, 0.9),
            "left_knee": Landmark(0.53, 0.70, 0, 0.9),
            "right_knee": Landmark(0.47, 0.70, 0, 0.9),
            "left_ankle": Landmark(0.53, 0.90, 0, 0.9),
            "right_ankle": Landmark(0.47, 0.90, 0, 0.9),
            "left_elbow": Landmark(0.58, 0.38, 0, 0.9),
            "right_elbow": Landmark(0.42, 0.38, 0, 0.9),
            "left_wrist": Landmark(0.60, 0.50, 0, 0.9),
            "right_wrist": Landmark(0.40, 0.50, 0, 0.9),
            "nose": Landmark(0.50, 0.15, 0, 0.9),
        }

        pose = PoseResult(
            landmarks=dummy_landmarks,
            angles=feat,
            is_valid=True,
            confidence=0.9,
        )

        result = analyzer.analyze(pose)
        # Label: 1 = good form (no warning/error), 0 = bad form
        has_issues = any(
            e.severity.value in ("warning", "error", "critical")
            for e in result.errors
            if e.error_type not in ("not_in_position", "no_pose_detected")
        )
        labels.append(0 if has_issues else 1)
        analyzer.reset()

    return labels


def train_exercise_model(exercise, features_list, labels):
    """Train and save a KNN model for the exercise."""
    X = pd.DataFrame(features_list)[UNIFIED_FEATURES].values
    y = np.array(labels)

    if len(np.unique(y)) < 2:
        print(f"  Skipping {exercise}: only one class in data ({np.unique(y)})")
        return None

    # Train KNN (best performer from our comparison)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ])

    cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    print(f"  KNN accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Samples: {len(X)} (good={sum(y==1)}, bad={sum(y==0)})")

    # Train on full data and save
    pipeline.fit(X, y)
    model_dir = project_root / "backend" / "models" / "pretrained"
    model_path = model_dir / f"{exercise}_form_realdata.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  Saved to: {model_path}")

    return scores.mean()


def main():
    eval_dir = project_root / "data" / "evaluation"
    workout_dir = eval_dir / "Workout:Exercises Video"

    print("=" * 70)
    print("TRAINING REAL-DATA FORM CLASSIFIERS FOR ALL EXERCISES")
    print("=" * 70)

    pose_estimator = PoseEstimator()
    results = {}

    # 1. Push-up: Use labeled correct/wrong videos
    pushup_base = eval_dir / "LSTM Exercise Classification - Push Up Videos"
    if pushup_base.exists():
        print(f"\n--- PUSHUP (labeled correct/wrong videos) ---")
        all_features = []
        all_labels = []

        for label_val, folder in [(1, "Correct sequence"), (0, "Wrong sequence")]:
            videos = sorted((pushup_base / folder).glob("*.mp4"))[:25]
            for video in videos:
                feats = extract_video_features(video, pose_estimator)
                for f in feats:
                    all_features.append(f)
                    all_labels.append(label_val)

        acc = train_exercise_model("pushup", all_features, all_labels)
        if acc:
            results["pushup"] = acc

    # 2. Other exercises: Use videos + rule-based labeling
    if workout_dir.exists():
        for folder_name, exercise in FOLDER_EXERCISE_MAP.items():
            folder = workout_dir / folder_name
            if not folder.exists():
                continue
            if exercise == "pushup":
                continue  # Already done above with labeled data

            videos = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.avi")))[:20]
            if not videos:
                continue

            print(f"\n--- {exercise.upper()} ({len(videos)} videos from '{folder_name}') ---")

            all_features = []
            for video in videos:
                feats = extract_video_features(video, pose_estimator)
                all_features.extend(feats)

            if len(all_features) < 20:
                print(f"  Too few frames ({len(all_features)}), skipping")
                continue

            # Label using rule-based analyzer
            analyzer_class = EXERCISE_ANALYZERS.get(exercise)
            if not analyzer_class:
                continue

            labels = label_with_rules(all_features, exercise, analyzer_class)
            acc = train_exercise_model(exercise, all_features, labels)
            if acc:
                results[exercise] = acc

    pose_estimator.close()

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    for exercise, acc in sorted(results.items()):
        print(f"  {exercise}: {acc:.1%} accuracy")

    print(f"\nModels saved in: backend/models/pretrained/*_form_realdata.joblib")


if __name__ == "__main__":
    main()
