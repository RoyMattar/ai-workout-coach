"""
Video Dataset Evaluation Script

Processes real exercise videos through the full AI pipeline and evaluates:
- Form quality detection accuracy (correct vs wrong form)
- Pose detection rate (% of frames with valid pose)
- Per-stage latency on real video data
- Rep counting accuracy

Datasets supported:
- Push-up videos: data/evaluation/LSTM Exercise Classification - Push Up Videos/
    Correct sequence/ and Wrong sequence/ folders
- Workout videos: data/evaluation/workoutfitness-video/ (or similar)
    Organized by exercise type folders

Usage:
    python -m tools.evaluate_videos
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress MediaPipe/TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from backend.orchestrator import WorkoutOrchestrator, ExerciseType


def process_video(video_path: str, exercise_type: ExerciseType, max_frames: int = 150) -> dict:
    """
    Process a single video through the full pipeline.

    Args:
        video_path: path to MP4 file
        exercise_type: which exercise analyzer to use
        max_frames: max frames to process (for speed)

    Returns:
        dict with metrics: pose_detection_rate, form_scores, rep_count, latencies, errors
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    orch = WorkoutOrchestrator(exercise_type=exercise_type)

    total_frames = 0
    valid_frames = 0
    good_form_frames = 0
    error_counts: dict[str, int] = {}
    latencies = []
    in_position_frames = 0

    while total_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip every other frame for speed
        total_frames += 1
        if total_frames % 2 != 0:
            continue

        result = orch.process_frame(frame)

        if result.pose_result.is_valid:
            valid_frames += 1

        # Check if person is in exercise position (not "not_in_position" or "no_pose_detected")
        error_types = [e.error_type for e in result.analysis_result.errors]
        if "not_in_position" not in error_types and "no_pose_detected" not in error_types:
            in_position_frames += 1

            if result.analysis_result.is_good_form:
                good_form_frames += 1

            for e in result.analysis_result.errors:
                if e.error_type not in ("not_in_position", "no_pose_detected"):
                    error_counts[e.error_type] = error_counts.get(e.error_type, 0) + 1

        latencies.append(result.pipeline_timing.total_ms)

    cap.release()
    orch.close()

    processed = total_frames // 2  # we skip every other frame
    pose_rate = valid_frames / processed if processed > 0 else 0
    form_score = (good_form_frames / in_position_frames * 100) if in_position_frames > 0 else 0
    rep_count = orch.session_stats.total_reps

    return {
        "video": os.path.basename(video_path),
        "total_frames": total_frames,
        "processed_frames": processed,
        "valid_pose_frames": valid_frames,
        "in_position_frames": in_position_frames,
        "good_form_frames": good_form_frames,
        "pose_detection_rate": round(pose_rate, 3),
        "form_score": round(form_score, 1),
        "rep_count": rep_count,
        "errors": error_counts,
        "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
        "max_latency_ms": round(np.max(latencies), 2) if latencies else 0,
    }


def evaluate_pushup_dataset():
    """Evaluate on the push-up correct/wrong video dataset."""
    base = project_root / "data" / "evaluation" / "LSTM Exercise Classification - Push Up Videos"
    correct_dir = base / "Correct sequence"
    wrong_dir = base / "Wrong sequence"

    if not correct_dir.exists():
        print("Push-up video dataset not found. Skipping.")
        return None

    print("=" * 70)
    print("PUSH-UP VIDEO EVALUATION (Correct vs Wrong Form)")
    print("=" * 70)

    results = {"correct": [], "wrong": []}

    # Process correct form videos
    correct_videos = sorted(correct_dir.glob("*.mp4"))[:20]  # Limit for speed
    print(f"\nProcessing {len(correct_videos)} CORRECT form videos...")
    for i, video in enumerate(correct_videos):
        r = process_video(str(video), ExerciseType.PUSHUP)
        r["ground_truth"] = "correct"
        results["correct"].append(r)
        print(f"  [{i+1}/{len(correct_videos)}] {r['video']}: score={r['form_score']}%, reps={r['rep_count']}, pose_rate={r['pose_detection_rate']}")

    # Process wrong form videos
    wrong_videos = sorted(wrong_dir.glob("*.mp4"))[:20]
    print(f"\nProcessing {len(wrong_videos)} WRONG form videos...")
    for i, video in enumerate(wrong_videos):
        r = process_video(str(video), ExerciseType.PUSHUP)
        r["ground_truth"] = "wrong"
        results["wrong"].append(r)
        print(f"  [{i+1}/{len(wrong_videos)}] {r['video']}: score={r['form_score']}%, reps={r['rep_count']}, pose_rate={r['pose_detection_rate']}")

    # Aggregate results
    correct_scores = [r["form_score"] for r in results["correct"] if r["in_position_frames"] > 0]
    wrong_scores = [r["form_score"] for r in results["wrong"] if r["in_position_frames"] > 0]
    correct_pose_rates = [r["pose_detection_rate"] for r in results["correct"]]
    wrong_pose_rates = [r["pose_detection_rate"] for r in results["wrong"]]

    all_latencies = [r["avg_latency_ms"] for r in results["correct"] + results["wrong"] if r["avg_latency_ms"] > 0]

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    if correct_scores:
        print(f"\nCorrect form videos ({len(correct_scores)} analyzed):")
        print(f"  Average form score: {np.mean(correct_scores):.1f}%")
        print(f"  Median form score:  {np.median(correct_scores):.1f}%")
        print(f"  Avg pose detection: {np.mean(correct_pose_rates):.1%}")

    if wrong_scores:
        print(f"\nWrong form videos ({len(wrong_scores)} analyzed):")
        print(f"  Average form score: {np.mean(wrong_scores):.1f}%")
        print(f"  Median form score:  {np.median(wrong_scores):.1f}%")
        print(f"  Avg pose detection: {np.mean(wrong_pose_rates):.1%}")

    if correct_scores and wrong_scores:
        # Test multiple thresholds and find optimal
        print(f"\nForm quality classification by threshold:")
        best_acc, best_thresh = 0, 50
        for threshold in [50, 60, 70, 80, 85, 90, 95]:
            c_ok = sum(1 for s in correct_scores if s >= threshold)
            w_ok = sum(1 for s in wrong_scores if s < threshold)
            acc = (c_ok + w_ok) / (len(correct_scores) + len(wrong_scores))
            print(f"  {threshold}%: correct={c_ok}/{len(correct_scores)}, wrong={w_ok}/{len(wrong_scores)}, accuracy={acc:.1%}")
            if acc > best_acc:
                best_acc, best_thresh = acc, threshold

        print(f"\n  Best threshold: {best_thresh}% → {best_acc:.1%} accuracy")

        # Score separation
        score_diff = np.mean(correct_scores) - np.mean(wrong_scores)
        print(f"  Score separation (correct - wrong): {score_diff:+.1f}%")

    if all_latencies:
        print(f"\nPipeline latency on real video:")
        print(f"  Average: {np.mean(all_latencies):.1f}ms")
        print(f"  Max:     {np.max(all_latencies):.1f}ms")

    return results


def evaluate_workout_videos():
    """Evaluate any additional workout video folders under data/evaluation/."""
    eval_dir = project_root / "data" / "evaluation"
    results = {}

    # Look for folders with exercise videos (not the pushup dataset)
    exercise_map = {
        "squat": ExerciseType.SQUAT,
        "squats": ExerciseType.SQUAT,
        "pushup": ExerciseType.PUSHUP,
        "push-up": ExerciseType.PUSHUP,
        "push_up": ExerciseType.PUSHUP,
        "lunge": ExerciseType.LUNGE,
        "lunges": ExerciseType.LUNGE,
        "deadlift": ExerciseType.DEADLIFT,
        "romanian deadlift": ExerciseType.DEADLIFT,
        "bicep_curl": ExerciseType.BICEP_CURL,
        "bicep curl": ExerciseType.BICEP_CURL,
        "barbell bicep curl": ExerciseType.BICEP_CURL,
        "barbell biceps curl": ExerciseType.BICEP_CURL,
        "hammer curl": ExerciseType.BICEP_CURL,
        "shoulder_press": ExerciseType.SHOULDER_PRESS,
        "shoulder press": ExerciseType.SHOULDER_PRESS,
        "situp": ExerciseType.SITUP,
        "sit-up": ExerciseType.SITUP,
        "plank": ExerciseType.PLANK,
    }

    for folder in eval_dir.iterdir():
        if not folder.is_dir():
            continue
        if "LSTM" in folder.name:
            continue  # Skip pushup dataset (handled separately)

        # Check subfolders for exercise-named directories
        for subfolder in folder.iterdir():
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name.lower().strip()
            exercise_type = exercise_map.get(folder_name)

            if not exercise_type:
                # Try partial match
                for key, et in exercise_map.items():
                    if key in folder_name:
                        exercise_type = et
                        break

            if not exercise_type:
                continue

            videos = list(subfolder.glob("*.mp4")) + list(subfolder.glob("*.avi"))
            if not videos:
                continue

            print(f"\n{'='*70}")
            print(f"EVALUATING: {subfolder.name} ({len(videos)} videos) → {exercise_type.value}")
            print(f"{'='*70}")

            exercise_results = []
            for i, video in enumerate(videos[:15]):  # Limit for speed
                r = process_video(str(video), exercise_type)
                exercise_results.append(r)
                print(f"  [{i+1}/{min(len(videos),15)}] {r['video']}: score={r['form_score']}%, reps={r['rep_count']}, pose={r['pose_detection_rate']}")

            scores = [r["form_score"] for r in exercise_results if r["in_position_frames"] > 0]
            pose_rates = [r["pose_detection_rate"] for r in exercise_results]

            if scores:
                print(f"\n  Average form score: {np.mean(scores):.1f}%")
                print(f"  Average pose detection: {np.mean(pose_rates):.1%}")

            results[exercise_type.value] = exercise_results

    return results


def save_evaluation_report(pushup_results, workout_results):
    """Save comprehensive evaluation report."""
    output_dir = project_root / "docs" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pushup_video_evaluation": None,
        "workout_video_evaluation": {},
        "summary": {},
    }

    if pushup_results:
        correct_scores = [r["form_score"] for r in pushup_results["correct"] if r["in_position_frames"] > 0]
        wrong_scores = [r["form_score"] for r in pushup_results["wrong"] if r["in_position_frames"] > 0]

        report["pushup_video_evaluation"] = {
            "correct_videos": len(pushup_results["correct"]),
            "wrong_videos": len(pushup_results["wrong"]),
            "correct_avg_score": round(np.mean(correct_scores), 1) if correct_scores else 0,
            "wrong_avg_score": round(np.mean(wrong_scores), 1) if wrong_scores else 0,
            "score_separation": round(np.mean(correct_scores) - np.mean(wrong_scores), 1) if correct_scores and wrong_scores else 0,
            "correct_details": pushup_results["correct"],
            "wrong_details": pushup_results["wrong"],
        }

    for exercise, vids in workout_results.items():
        scores = [r["form_score"] for r in vids if r["in_position_frames"] > 0]
        report["workout_video_evaluation"][exercise] = {
            "videos_analyzed": len(vids),
            "avg_form_score": round(np.mean(scores), 1) if scores else 0,
            "details": vids,
        }

    output_path = output_dir / "video_evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"VIDEO EVALUATION REPORT SAVED: {output_path}")
    print(f"{'='*70}")

    return report


def main():
    print("AI Workout Coach — Video Dataset Evaluation")
    print("Processing real exercise videos through the full AI pipeline...")
    print()

    # 1. Push-up correct/wrong dataset
    pushup_results = evaluate_pushup_dataset()

    # 2. Any other workout video folders
    workout_results = evaluate_workout_videos()

    # 3. Save report
    save_evaluation_report(pushup_results, workout_results)


if __name__ == "__main__":
    main()
