"""
Tests for backend/exercises/squat.py and backend/exercises/pushup.py

Covers form detection, phase detection, rep counting, symmetry checks,
and graceful handling of invalid poses for both SquatAnalyzer and PushupAnalyzer.
"""
import pytest

from backend.exercises.squat import SquatAnalyzer
from backend.exercises.pushup import PushupAnalyzer
from backend.exercises.base import ExercisePhase
from backend.pose_estimator import PoseResult, Landmark


# ═══════════════════════════════════════════════════════
# Squat Analyzer Tests
# ═══════════════════════════════════════════════════════

def test_squat_good_form(squat_bottom_pose):
    """Good squat form (85-degree knees, 25-degree torso) should produce no warning/error errors."""
    analyzer = SquatAnalyzer()
    result = analyzer.analyze(squat_bottom_pose)

    # Good form means either is_good_form is True or there are no warning/error severity errors
    severe_errors = [
        e for e in result.errors
        if e.severity.value in ("warning", "error", "critical")
    ]
    assert len(severe_errors) == 0, f"Unexpected severe errors: {[e.error_type for e in severe_errors]}"


def test_squat_insufficient_depth(squat_shallow_pose):
    """A shallow squat (knee angle ~119) should detect 'insufficient_depth' error."""
    analyzer = SquatAnalyzer()
    result = analyzer.analyze(squat_shallow_pose)

    error_types = [e.error_type for e in result.errors]
    assert "insufficient_depth" in error_types, (
        f"Expected 'insufficient_depth' in errors, got {error_types}"
    )


def test_squat_forward_lean(squat_forward_lean_pose):
    """Excessive forward lean (torso angle 55) should detect 'excessive_forward_lean'."""
    analyzer = SquatAnalyzer()
    result = analyzer.analyze(squat_forward_lean_pose)

    error_types = [e.error_type for e in result.errors]
    assert "excessive_forward_lean" in error_types, (
        f"Expected 'excessive_forward_lean' in errors, got {error_types}"
    )


def test_squat_phase_standing(standing_pose):
    """Standing pose (knee angle ~171) should be detected as STANDING phase."""
    analyzer = SquatAnalyzer()
    phase = analyzer.detect_phase(standing_pose)
    assert phase == ExercisePhase.STANDING


def test_squat_phase_bottom(squat_bottom_pose):
    """Squat bottom pose (knee angle ~84) should be BOTTOM or DESCENDING."""
    analyzer = SquatAnalyzer()
    phase = analyzer.detect_phase(squat_bottom_pose)
    assert phase in (ExercisePhase.BOTTOM, ExercisePhase.DESCENDING)


def test_squat_invalid_pose(invalid_pose):
    """Invalid pose should be handled gracefully with 'no_pose_detected'."""
    analyzer = SquatAnalyzer()
    result = analyzer.analyze(invalid_pose)

    error_types = [e.error_type for e in result.errors]
    assert "no_pose_detected" in error_types
    assert result.is_good_form is False


def test_squat_rep_counting(standing_pose, squat_bottom_pose):
    """Simulating a full rep cycle (standing -> bottom -> standing) should increment rep count."""
    analyzer = SquatAnalyzer()

    # Start standing
    result = analyzer.analyze(standing_pose)
    assert result.rep_count == 0

    # Create intermediate descending poses to go through proper phase transitions
    # Descending pose (knee angle ~130)
    descending_pose = PoseResult(
        landmarks=standing_pose.landmarks,
        angles={
            **standing_pose.angles,
            "left_knee_angle": 130.0,
            "right_knee_angle": 128.0,
        },
        is_valid=True,
        confidence=0.9,
    )
    analyzer.analyze(descending_pose)

    # Bottom pose
    analyzer.analyze(squat_bottom_pose)

    # Ascending pose (knee angle ~130)
    ascending_pose = PoseResult(
        landmarks=standing_pose.landmarks,
        angles={
            **standing_pose.angles,
            "left_knee_angle": 130.0,
            "right_knee_angle": 128.0,
        },
        is_valid=True,
        confidence=0.9,
    )
    analyzer.analyze(ascending_pose)

    # Return to standing - should complete the rep
    result = analyzer.analyze(standing_pose)
    assert result.rep_count >= 1, f"Expected at least 1 rep, got {result.rep_count}"


def test_squat_symmetry_check():
    """Large knee angle difference (>15 degrees) should detect 'asymmetric_depth'."""
    analyzer = SquatAnalyzer()

    asymmetric_pose = PoseResult(
        landmarks={
            "nose": Landmark(x=0.50, y=0.25, z=0.0, visibility=0.95),
            "left_shoulder": Landmark(x=0.55, y=0.35, z=0.0, visibility=0.95),
            "right_shoulder": Landmark(x=0.45, y=0.35, z=0.0, visibility=0.95),
            "left_elbow": Landmark(x=0.60, y=0.50, z=0.0, visibility=0.95),
            "right_elbow": Landmark(x=0.40, y=0.50, z=0.0, visibility=0.95),
            "left_wrist": Landmark(x=0.58, y=0.55, z=0.0, visibility=0.95),
            "right_wrist": Landmark(x=0.42, y=0.55, z=0.0, visibility=0.95),
            "left_hip": Landmark(x=0.53, y=0.55, z=0.0, visibility=0.95),
            "right_hip": Landmark(x=0.47, y=0.55, z=0.0, visibility=0.95),
            "left_knee": Landmark(x=0.56, y=0.72, z=0.0, visibility=0.95),
            "right_knee": Landmark(x=0.44, y=0.72, z=0.0, visibility=0.95),
            "left_ankle": Landmark(x=0.53, y=0.90, z=0.0, visibility=0.95),
            "right_ankle": Landmark(x=0.47, y=0.90, z=0.0, visibility=0.95),
        },
        angles={
            "left_knee_angle": 75.0,   # Deep on left
            "right_knee_angle": 100.0,  # Shallow on right (>15 diff)
            "left_hip_angle": 80.0,
            "right_hip_angle": 78.0,
            "left_elbow_angle": 155.0,
            "right_elbow_angle": 153.0,
            "torso_angle": 25.0,
            "shoulder_hip_alignment": 0.02,
        },
        is_valid=True,
        confidence=0.9,
    )

    result = analyzer.analyze(asymmetric_pose)
    error_types = [e.error_type for e in result.errors]
    assert "asymmetric_depth" in error_types, (
        f"Expected 'asymmetric_depth' in errors, got {error_types}"
    )


# ═══════════════════════════════════════════════════════
# Pushup Analyzer Tests
# ═══════════════════════════════════════════════════════

def test_pushup_plank_detection(pushup_plank_pose):
    """Pushup plank pose (elbow angle ~169) should be detected as PLANK phase."""
    analyzer = PushupAnalyzer()
    phase = analyzer.detect_phase(pushup_plank_pose)
    assert phase == ExercisePhase.PLANK


def test_pushup_invalid_pose(invalid_pose):
    """Invalid pose should be handled gracefully for pushup analyzer."""
    analyzer = PushupAnalyzer()
    result = analyzer.analyze(invalid_pose)

    error_types = [e.error_type for e in result.errors]
    assert "no_pose_detected" in error_types
    assert result.is_good_form is False


def test_pushup_good_form(pushup_plank_pose):
    """Plank pose with good alignment should produce no errors (or only info-level)."""
    analyzer = PushupAnalyzer()
    result = analyzer.analyze(pushup_plank_pose)

    severe_errors = [
        e for e in result.errors
        if e.severity.value in ("warning", "error", "critical")
    ]
    assert result.is_good_form is True or len(severe_errors) == 0


def test_pushup_analyzer_reset():
    """Calling reset() should zero out rep count and clear error history."""
    analyzer = PushupAnalyzer()

    # Simulate some activity
    analyzer.rep_counter.count = 5
    analyzer.error_history.append([])

    analyzer.reset()

    assert analyzer.rep_counter.count == 0
    assert len(analyzer.error_history) == 0


# ═══════════════════════════════════════════════════════
# Pose Validation Tests (_is_exercise_pose)
# ═══════════════════════════════════════════════════════

def test_squat_rejects_sitting_pose():
    """When ankles are in the upper half of frame (sitting), should return 'not_in_position'."""
    from tests.conftest import make_landmark
    sitting_landmarks = {
        "nose": make_landmark(0.50, 0.15),
        "left_shoulder": make_landmark(0.55, 0.25),
        "right_shoulder": make_landmark(0.45, 0.25),
        "left_elbow": make_landmark(0.60, 0.35),
        "right_elbow": make_landmark(0.40, 0.35),
        "left_wrist": make_landmark(0.60, 0.45),
        "right_wrist": make_landmark(0.40, 0.45),
        "left_hip": make_landmark(0.53, 0.38),
        "right_hip": make_landmark(0.47, 0.38),
        "left_knee": make_landmark(0.53, 0.38),  # Knees at same height as hips (sitting)
        "right_knee": make_landmark(0.47, 0.38),
        "left_ankle": make_landmark(0.53, 0.38),  # Ankles in upper frame
        "right_ankle": make_landmark(0.47, 0.38),
    }
    sitting_pose = PoseResult(
        landmarks=sitting_landmarks,
        angles={"left_knee_angle": 90.0, "right_knee_angle": 88.0, "torso_angle": 10.0},
        is_valid=True,
        confidence=0.85,
    )

    analyzer = SquatAnalyzer()
    result = analyzer.analyze(sitting_pose)
    error_types = [e.error_type for e in result.errors]
    assert "not_in_position" in error_types


def test_squat_accepts_standing_pose(standing_pose):
    """A valid standing pose with full body visible should pass _is_exercise_pose."""
    analyzer = SquatAnalyzer()
    result = analyzer.analyze(standing_pose)
    error_types = [e.error_type for e in result.errors]
    assert "not_in_position" not in error_types


def test_pushup_rejects_upright_pose():
    """A clearly upright standing pose should not pass push-up _is_exercise_pose."""
    from tests.conftest import make_landmark
    # Create a clearly upright pose: shoulders at 0.20, hips at 0.55 → diff = 0.35 > 0.3
    upright_landmarks = {
        "nose": make_landmark(0.50, 0.10),
        "left_shoulder": make_landmark(0.55, 0.20),
        "right_shoulder": make_landmark(0.45, 0.20),
        "left_elbow": make_landmark(0.60, 0.35),
        "right_elbow": make_landmark(0.40, 0.35),
        "left_wrist": make_landmark(0.60, 0.50),
        "right_wrist": make_landmark(0.40, 0.50),
        "left_hip": make_landmark(0.53, 0.55),
        "right_hip": make_landmark(0.47, 0.55),
        "left_knee": make_landmark(0.53, 0.73),
        "right_knee": make_landmark(0.47, 0.73),
        "left_ankle": make_landmark(0.53, 0.90),
        "right_ankle": make_landmark(0.47, 0.90),
    }
    upright_pose = PoseResult(
        landmarks=upright_landmarks,
        angles={"left_elbow_angle": 165.0, "right_elbow_angle": 163.0, "torso_angle": 5.0},
        is_valid=True,
        confidence=0.9,
    )
    analyzer = PushupAnalyzer()
    result = analyzer.analyze(upright_pose)
    error_types = [e.error_type for e in result.errors]
    assert "not_in_position" in error_types
