"""
Tests for the 6 new exercise analyzers:
lunge, plank, deadlift, bicep_curl, shoulder_press, situp
"""
import pytest
from backend.exercises.lunge import LungeAnalyzer
from backend.exercises.plank import PlankAnalyzer
from backend.exercises.deadlift import DeadliftAnalyzer
from backend.exercises.bicep_curl import BicepCurlAnalyzer
from backend.exercises.shoulder_press import ShoulderPressAnalyzer
from backend.exercises.situp import SitupAnalyzer
from backend.exercises.base import ExercisePhase
from backend.pose_estimator import PoseResult


# ── Lunge ──

def test_lunge_exercise_name():
    assert LungeAnalyzer().exercise_name == "lunge"

def test_lunge_invalid_pose(invalid_pose):
    analyzer = LungeAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)

def test_lunge_reset():
    analyzer = LungeAnalyzer()
    analyzer.rep_counter.count = 5
    analyzer.reset()
    assert analyzer.rep_counter.count == 0


# ── Plank ──

def test_plank_exercise_name():
    assert PlankAnalyzer().exercise_name == "plank"

def test_plank_invalid_pose(invalid_pose):
    analyzer = PlankAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)

def test_plank_no_rep_counting():
    """Plank is isometric — should not count reps."""
    analyzer = PlankAnalyzer()
    # Even after processing, rep count should stay 0
    assert analyzer.rep_counter.count == 0


# ── Deadlift ──

def test_deadlift_exercise_name():
    assert DeadliftAnalyzer().exercise_name == "deadlift"

def test_deadlift_invalid_pose(invalid_pose):
    analyzer = DeadliftAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)

def test_deadlift_reset():
    analyzer = DeadliftAnalyzer()
    analyzer.rep_counter.count = 3
    analyzer.reset()
    assert analyzer.rep_counter.count == 0


# ── Bicep Curl ──

def test_bicep_curl_exercise_name():
    assert BicepCurlAnalyzer().exercise_name == "bicep_curl"

def test_bicep_curl_invalid_pose(invalid_pose):
    analyzer = BicepCurlAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)

def test_bicep_curl_upper_body_only():
    """Bicep curl should NOT require leg landmarks for _is_exercise_pose."""
    from tests.conftest import make_landmark
    # Upper body only — no knees or ankles
    landmarks = {
        "nose": make_landmark(0.50, 0.15),
        "left_shoulder": make_landmark(0.55, 0.30),
        "right_shoulder": make_landmark(0.45, 0.30),
        "left_elbow": make_landmark(0.58, 0.45),
        "right_elbow": make_landmark(0.42, 0.45),
        "left_wrist": make_landmark(0.58, 0.55),
        "right_wrist": make_landmark(0.42, 0.55),
        "left_hip": make_landmark(0.53, 0.55),
        "right_hip": make_landmark(0.47, 0.55),
    }
    pose = PoseResult(
        landmarks=landmarks,
        angles={
            "left_elbow_angle": 160.0,
            "right_elbow_angle": 158.0,
            "shoulder_hip_alignment": 0.01,
        },
        is_valid=True,
        confidence=0.85,
    )
    analyzer = BicepCurlAnalyzer()
    result = analyzer.analyze(pose)
    # Should NOT return "not_in_position" since it only needs upper body
    error_types = [e.error_type for e in result.errors]
    assert "not_in_position" not in error_types


# ── Shoulder Press ──

def test_shoulder_press_exercise_name():
    assert ShoulderPressAnalyzer().exercise_name == "shoulder_press"

def test_shoulder_press_invalid_pose(invalid_pose):
    analyzer = ShoulderPressAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)


# ── Situp ──

def test_situp_exercise_name():
    assert SitupAnalyzer().exercise_name == "situp"

def test_situp_invalid_pose(invalid_pose):
    analyzer = SitupAnalyzer()
    result = analyzer.analyze(invalid_pose)
    assert any(e.error_type == "no_pose_detected" for e in result.errors)


# ── Integration: All exercises exist in orchestrator ──

def test_all_exercises_in_orchestrator():
    """All 8 exercise types should be registered in the orchestrator."""
    from backend.orchestrator import WorkoutOrchestrator, ExerciseType
    import numpy as np

    for et in ExerciseType:
        orch = WorkoutOrchestrator(exercise_type=et)
        assert orch.current_exercise == et
        assert et.value in orch.analyzers
        orch.close()
