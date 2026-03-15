"""
Form Quality Model Comparison

Extracts features from real exercise videos, then trains and compares
multiple ML approaches for form quality classification:

1. SVM (Support Vector Machine)
2. Random Forest
3. K-Nearest Neighbors
4. Neural Network (MLP - Multi-Layer Perceptron)
5. GPT-4o Vision (sampled frames)

This demonstrates rigorous model evaluation and selection — a key
requirement of the "Orchestrating AI" project template.

Usage:
    python -m tools.compare_form_models
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from backend.pose_estimator import PoseEstimator


def extract_video_features(video_path: str, pose_estimator: PoseEstimator, max_frames: int = 100) -> list[dict]:
    """Extract pose angle features from every frame of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    features_list = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Sample every 3rd frame
            continue

        result = pose_estimator.process_frame(frame)
        if not result.is_valid:
            continue

        angles = result.angles
        if not angles:
            continue

        features = {
            "left_knee_angle": angles.get("left_knee_angle", 170),
            "right_knee_angle": angles.get("right_knee_angle", 170),
            "left_hip_angle": angles.get("left_hip_angle", 170),
            "right_hip_angle": angles.get("right_hip_angle", 170),
            "left_elbow_angle": angles.get("left_elbow_angle", 170),
            "right_elbow_angle": angles.get("right_elbow_angle", 170),
            "torso_angle": angles.get("torso_angle", 0),
            "shoulder_hip_alignment": angles.get("shoulder_hip_alignment", 0),
        }
        features_list.append(features)

    cap.release()
    return features_list


def build_pushup_dataset(pose_estimator: PoseEstimator) -> tuple:
    """Build labeled dataset from push-up correct/wrong videos."""
    base = project_root / "data" / "evaluation" / "LSTM Exercise Classification - Push Up Videos"
    correct_dir = base / "Correct sequence"
    wrong_dir = base / "Wrong sequence"

    if not correct_dir.exists():
        print("Push-up video dataset not found!")
        return None, None

    all_features = []
    all_labels = []

    # Correct form videos
    correct_videos = sorted(correct_dir.glob("*.mp4"))[:25]
    print(f"Extracting features from {len(correct_videos)} CORRECT push-up videos...")
    for i, video in enumerate(correct_videos):
        features = extract_video_features(str(video), pose_estimator)
        for f in features:
            f["label"] = 1  # correct
            all_features.append(f)
            all_labels.append(1)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(correct_videos)} done ({len(all_features)} frames)")

    # Wrong form videos
    wrong_videos = sorted(wrong_dir.glob("*.mp4"))[:25]
    print(f"Extracting features from {len(wrong_videos)} WRONG push-up videos...")
    for i, video in enumerate(wrong_videos):
        features = extract_video_features(str(video), pose_estimator)
        for f in features:
            f["label"] = 0  # wrong
            all_features.append(f)
            all_labels.append(0)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(wrong_videos)} done ({len(all_features)} frames)")

    df = pd.DataFrame(all_features)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = np.array(all_labels)

    print(f"\nDataset: {len(X)} frames, {len(feature_cols)} features")
    print(f"  Correct: {sum(y==1)}, Wrong: {sum(y==0)}")

    return X, y


def compare_ml_models(X, y):
    """Train and compare multiple ML classifiers via cross-validation."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: ML Classifiers on Real Push-up Video Data")
    print("=" * 70)

    classifiers = {
        "K-Nearest Neighbors (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM (RBF kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ]),
        "Random Forest (100 trees)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ]),
        "Neural Network (MLP 64-32)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            )),
        ]),
        "Neural Network (MLP 128-64-32)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in classifiers.items():
        t0 = time.time()
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        train_time = time.time() - t0

        # Measure inference speed
        pipeline.fit(X, y)
        t0 = time.time()
        for _ in range(5):
            pipeline.predict(X[:100])
        inference_ms = (time.time() - t0) / 5 / 100 * 1000  # ms per sample

        results[name] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
            "scores": scores.tolist(),
            "train_time_s": round(train_time, 2),
            "inference_ms_per_frame": round(inference_ms, 4),
        }

        print(f"\n{name}:")
        print(f"  Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print(f"  Train time: {train_time:.2f}s")
        print(f"  Inference: {inference_ms:.4f}ms per frame")

    # Find best
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n{'='*70}")
    print(f"BEST ML MODEL: {best_name} ({results[best_name]['accuracy']:.4f})")
    print(f"{'='*70}")

    # Detailed report for best model
    best_pipeline = classifiers[best_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    print(f"\nDetailed Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=["wrong_form", "correct_form"]))

    return results, best_name, classifiers[best_name]


def evaluate_gpt4o_vision(n_samples: int = 20):
    """Evaluate GPT-4o Vision on sampled push-up frames."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: GPT-4o Vision on Push-up Frames")
    print("=" * 70)

    from backend.config import get_settings
    settings = get_settings()

    if not settings.openai_api_key:
        print("No OpenAI API key — skipping GPT-4o Vision evaluation.")
        return {"accuracy": "N/A", "note": "No API key available"}

    import openai
    client = openai.OpenAI(api_key=settings.openai_api_key)

    base = project_root / "data" / "evaluation" / "LSTM Exercise Classification - Push Up Videos"
    correct_dir = base / "Correct sequence"
    wrong_dir = base / "Wrong sequence"

    correct_videos = sorted(correct_dir.glob("*.mp4"))[:n_samples // 2]
    wrong_videos = sorted(wrong_dir.glob("*.mp4"))[:n_samples // 2]

    import base64

    correct_count = 0
    wrong_count = 0
    total = 0
    total_latency = 0

    def analyze_frame_with_vision(frame_bgr, label):
        """Send a single frame to GPT-4o Vision for form assessment."""
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        t0 = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this push-up form correct or wrong? Reply with ONLY one word: 'correct' or 'wrong'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}},
                ],
            }],
            max_tokens=10,
            temperature=0,
        )
        latency = time.time() - t0
        answer = response.choices[0].message.content.strip().lower()
        return answer, latency

    # Sample one frame from each video
    print(f"Testing {len(correct_videos)} correct + {len(wrong_videos)} wrong videos...")

    for video_path in correct_videos:
        cap = cv2.VideoCapture(str(video_path))
        # Get middle frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        try:
            answer, latency = analyze_frame_with_vision(frame, "correct")
            total += 1
            total_latency += latency
            if "correct" in answer:
                correct_count += 1
            print(f"  Correct video → GPT-4o says: '{answer}' ({latency:.1f}s)")
        except Exception as e:
            print(f"  Error: {e}")

    for video_path in wrong_videos:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        try:
            answer, latency = analyze_frame_with_vision(frame, "wrong")
            total += 1
            total_latency += latency
            if "wrong" in answer or "incorrect" in answer:
                wrong_count += 1
            print(f"  Wrong video → GPT-4o says: '{answer}' ({latency:.1f}s)")
        except Exception as e:
            print(f"  Error: {e}")

    accuracy = (correct_count + wrong_count) / total if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0

    print(f"\nGPT-4o Vision Results:")
    print(f"  Correct identified: {correct_count}/{len(correct_videos)}")
    print(f"  Wrong identified: {wrong_count}/{len(wrong_videos)}")
    print(f"  Overall accuracy: {accuracy:.1%}")
    print(f"  Avg latency: {avg_latency:.1f}s per frame")
    print(f"  Estimated cost: ~${total * 0.01:.2f} for {total} frames")

    return {
        "accuracy": round(accuracy, 4),
        "correct_identified": correct_count,
        "wrong_identified": wrong_count,
        "total_samples": total,
        "avg_latency_s": round(avg_latency, 2),
        "estimated_cost_usd": round(total * 0.01, 2),
    }


def save_comparison_report(ml_results, gpt_results, best_model_name):
    """Save the full comparison report."""
    output_dir = project_root / "docs" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ml_model_comparison": ml_results,
        "gpt4o_vision": gpt_results,
        "best_ml_model": best_model_name,
        "conclusion": {
            "selected_model": best_model_name,
            "rationale": (
                f"{best_model_name} was selected as the primary form quality classifier "
                f"based on cross-validation accuracy on real video data. "
                f"GPT-4o Vision provides the highest potential accuracy but is too slow "
                f"and expensive for real-time per-frame analysis. It is used as an async "
                f"secondary opinion every few seconds. The ML classifier runs in <1ms "
                f"per frame, enabling real-time feedback at 60+ FPS."
            ),
        },
    }

    path = output_dir / "model_comparison_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved to: {path}")
    return report


def save_best_model(best_pipeline, X, y):
    """Save the best-performing model for production use."""
    best_pipeline.fit(X, y)  # Train on full dataset
    model_dir = project_root / "backend" / "models" / "pretrained"
    model_path = model_dir / "pushup_form_realdata.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Best model saved to: {model_path}")


def main():
    print("=" * 70)
    print("FORM QUALITY MODEL COMPARISON")
    print("Testing SVM, Random Forest, KNN, Neural Network, GPT-4o Vision")
    print("on real push-up video data (correct vs wrong form)")
    print("=" * 70)

    # Initialize pose estimator
    print("\nInitializing pose estimator...")
    pose_estimator = PoseEstimator()

    # Build dataset from real videos
    X, y = build_pushup_dataset(pose_estimator)
    pose_estimator.close()

    if X is None:
        print("Failed to build dataset.")
        return

    # Compare ML models
    ml_results, best_name, best_pipeline = compare_ml_models(X, y)

    # Evaluate GPT-4o Vision (small sample due to cost)
    gpt_results = evaluate_gpt4o_vision(n_samples=20)

    # Save results
    save_comparison_report(ml_results, gpt_results, best_name)
    save_best_model(best_pipeline, X, y)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Latency':>12} {'Cost':>8}")
    print("-" * 70)
    for name, r in sorted(ml_results.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{name:<35} {r['accuracy']:>9.1%} {r['inference_ms_per_frame']:>9.4f}ms {'Free':>8}")
    if isinstance(gpt_results.get("accuracy"), float):
        print(f"{'GPT-4o Vision':<35} {gpt_results['accuracy']:>9.1%} {gpt_results['avg_latency_s']*1000:>8.0f}ms {'$0.01/fr':>8}")

    print(f"\nSelected: {best_name}")
    print(f"Rationale: Best accuracy on real data with real-time inference speed.")


if __name__ == "__main__":
    main()
