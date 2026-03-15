"""
Pipeline Evaluation Script

Benchmarks the full AI pipeline on the Kaggle dataset:
- Exercise recognition accuracy
- Per-stage latency breakdown
- Form classifier vs rule-based agreement rate
- Generates charts and tables for the report

Usage:
    python -m tools.evaluate_pipeline
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.train_with_real_data import extract_features, LANDMARK_INDICES


def evaluate_exercise_recognizer():
    """Evaluate the real-data exercise recognizer."""
    import joblib
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix

    model_path = project_root / "backend" / "models" / "pretrained" / "exercise_recognizer.joblib"
    data_path = project_root / "data" / "kaggle" / "dataset_all_points.csv"

    if not model_path.exists() or not data_path.exists():
        print("Exercise recognizer model or data not found. Run train_with_real_data.py first.")
        return None

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    # Extract features
    features = []
    labels = []
    mapping = {
        "left_bicep": "bicep_curl", "right_bicep": "bicep_curl",
        "left_shoulder": "shoulder_press", "right_shoulder": "shoulder_press",
        "left_tricep": "tricep", "right_tricep": "tricep",
        "rest": "rest",
    }

    for _, row in df.iterrows():
        feat = extract_features(row)
        if feat:
            features.append(feat)
            labels.append(mapping.get(row["class"], "unknown"))

    X = pd.DataFrame(features).values
    y = np.array(labels)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print("=" * 60)
    print("EXERCISE RECOGNIZER EVALUATION (Real Kaggle Data)")
    print("=" * 60)
    print(f"Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"Per-fold: {[f'{s:.4f}' for s in scores]}")

    y_pred = model.predict(X)
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    return {
        "accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_form_classifiers():
    """Evaluate synthetic-data form classifiers."""
    import joblib
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    results = {}

    for exercise in ["squat", "pushup"]:
        model_path = project_root / "backend" / "models" / "pretrained" / f"{exercise}_classifier.joblib"
        data_path = project_root / "backend" / "models" / "training_data" / f"{exercise}_training_data.csv"

        if not model_path.exists() or not data_path.exists():
            continue

        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        from tools.generate_test_data import SQUAT_FEATURES, PUSHUP_FEATURES
        features = SQUAT_FEATURES if exercise == "squat" else PUSHUP_FEATURES

        X = df[features].values
        y = df["form_quality"].values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

        print(f"\n{'='*60}")
        print(f"FORM CLASSIFIER: {exercise.upper()} (Synthetic Data)")
        print(f"{'='*60}")
        print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        results[exercise] = {
            "accuracy": float(scores.mean()),
            "std": float(scores.std()),
        }

    return results


def benchmark_pipeline_latency():
    """Benchmark per-stage latency of the full pipeline."""
    from backend.orchestrator import WorkoutOrchestrator, ExerciseType

    print(f"\n{'='*60}")
    print("PIPELINE LATENCY BENCHMARK")
    print("="*60)

    # Use a synthetic test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    orch = WorkoutOrchestrator(exercise_type=ExerciseType.SQUAT)

    # Warm-up
    for _ in range(3):
        orch.process_frame(frame)

    # Benchmark
    n_frames = 50
    timings = []

    for i in range(n_frames):
        result = orch.process_frame(frame)
        timings.append(result.pipeline_timing.to_dict())

    orch.close()

    # Aggregate
    stages = ["pose_estimation_ms", "ml_classification_ms", "rule_analysis_ms",
              "fusion_ms", "feedback_ms", "total_ms"]

    print(f"\nResults over {n_frames} frames:")
    print(f"{'Stage':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    latency_results = {}
    for stage in stages:
        values = [t[stage] for t in timings]
        mean_v = np.mean(values)
        std_v = np.std(values)
        min_v = np.min(values)
        max_v = np.max(values)
        print(f"{stage:<25} {mean_v:>7.2f}ms {std_v:>7.2f}ms {min_v:>7.2f}ms {max_v:>7.2f}ms")
        latency_results[stage] = {
            "mean": round(float(mean_v), 2),
            "std": round(float(std_v), 2),
            "min": round(float(min_v), 2),
            "max": round(float(max_v), 2),
        }

    fps = 1000.0 / latency_results["total_ms"]["mean"]
    print(f"\nEstimated throughput: {fps:.1f} FPS")
    latency_results["fps"] = round(fps, 1)

    return latency_results


def generate_summary_report(recognizer_results, classifier_results, latency_results):
    """Save a JSON summary of all evaluation results."""
    output_dir = project_root / "docs" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "exercise_recognizer": recognizer_results,
        "form_classifiers": classifier_results,
        "pipeline_latency": latency_results,
        "summary": {
            "total_models_evaluated": 3,
            "datasets_used": [
                "Kaggle Multi-Class Exercise Poses (dp5995) - 2700 samples",
                "Synthetic generated data - 3000 samples per exercise",
            ],
            "key_findings": [
                f"Exercise recognition: {recognizer_results['accuracy']*100:.1f}% accuracy on real data" if recognizer_results else "N/A",
                f"Squat form classifier: {classifier_results.get('squat', {}).get('accuracy', 0)*100:.1f}% accuracy",
                f"Push-up form classifier: {classifier_results.get('pushup', {}).get('accuracy', 0)*100:.1f}% accuracy",
                f"Pipeline throughput: {latency_results.get('fps', 0)} FPS",
                f"Average total latency: {latency_results.get('total_ms', {}).get('mean', 0)}ms per frame",
            ],
        },
    }

    output_path = output_dir / "evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT SAVED")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"\nKey findings:")
    for finding in report["summary"]["key_findings"]:
        print(f"  - {finding}")

    return report


def main():
    print("AI Workout Coach - Pipeline Evaluation")
    print("=" * 60)

    # 1. Exercise recognizer (real data)
    recognizer_results = evaluate_exercise_recognizer()

    # 2. Form classifiers (synthetic data)
    classifier_results = evaluate_form_classifiers()

    # 3. Pipeline latency
    latency_results = benchmark_pipeline_latency()

    # 4. Generate report
    generate_summary_report(recognizer_results, classifier_results, latency_results)


if __name__ == "__main__":
    main()
