"""
Shared test fixtures for the AI Workout Coach test suite.

Provides reusable sample pose data, mock objects, and test utilities
used across all test modules.
"""
import numpy as np
import pytest

from backend.pose_estimator import Landmark, PoseResult
from backend.exercises.base import (
    AnalysisResult,
    ExercisePhase,
    FormError,
    ErrorSeverity,
)


# ── Landmark Factories ──

def make_landmark(x=0.5, y=0.5, z=0.0, visibility=0.95):
    """Create a Landmark with defaults."""
    return Landmark(x=x, y=y, z=z, visibility=visibility)


def make_standing_landmarks():
    """Landmarks representing a person standing upright."""
    return {
        "nose": make_landmark(0.50, 0.15),
        "left_shoulder": make_landmark(0.55, 0.30),
        "right_shoulder": make_landmark(0.45, 0.30),
        "left_elbow": make_landmark(0.60, 0.45),
        "right_elbow": make_landmark(0.40, 0.45),
        "left_wrist": make_landmark(0.60, 0.60),
        "right_wrist": make_landmark(0.40, 0.60),
        "left_hip": make_landmark(0.53, 0.55),
        "right_hip": make_landmark(0.47, 0.55),
        "left_knee": make_landmark(0.53, 0.73),
        "right_knee": make_landmark(0.47, 0.73),
        "left_ankle": make_landmark(0.53, 0.90),
        "right_ankle": make_landmark(0.47, 0.90),
    }


def make_squat_bottom_landmarks():
    """Landmarks at squat bottom position (good form, ~85° knee angle).
    Knees are aligned with ankles (no cave)."""
    return {
        "nose": make_landmark(0.50, 0.25),
        "left_shoulder": make_landmark(0.55, 0.35),
        "right_shoulder": make_landmark(0.45, 0.35),
        "left_elbow": make_landmark(0.60, 0.50),
        "right_elbow": make_landmark(0.40, 0.50),
        "left_wrist": make_landmark(0.58, 0.55),
        "right_wrist": make_landmark(0.42, 0.55),
        "left_hip": make_landmark(0.53, 0.55),
        "right_hip": make_landmark(0.47, 0.55),
        "left_knee": make_landmark(0.53, 0.72),
        "right_knee": make_landmark(0.47, 0.72),
        "left_ankle": make_landmark(0.53, 0.90),
        "right_ankle": make_landmark(0.47, 0.90),
    }


def make_pushup_plank_landmarks():
    """Landmarks in push-up plank position (arms straight)."""
    return {
        "nose": make_landmark(0.20, 0.35),
        "left_shoulder": make_landmark(0.25, 0.40),
        "right_shoulder": make_landmark(0.25, 0.38),
        "left_elbow": make_landmark(0.25, 0.50),
        "right_elbow": make_landmark(0.25, 0.48),
        "left_wrist": make_landmark(0.25, 0.60),
        "right_wrist": make_landmark(0.25, 0.58),
        "left_hip": make_landmark(0.55, 0.42),
        "right_hip": make_landmark(0.55, 0.40),
        "left_knee": make_landmark(0.75, 0.44),
        "right_knee": make_landmark(0.75, 0.42),
        "left_ankle": make_landmark(0.90, 0.45),
        "right_ankle": make_landmark(0.90, 0.43),
    }


# ── PoseResult Fixtures ──

@pytest.fixture
def standing_pose():
    """A valid standing PoseResult."""
    landmarks = make_standing_landmarks()
    # Standing: knee angles ~170°, torso upright
    return PoseResult(
        landmarks=landmarks,
        angles={
            "left_knee_angle": 172.0,
            "right_knee_angle": 170.0,
            "left_hip_angle": 175.0,
            "right_hip_angle": 173.0,
            "left_elbow_angle": 165.0,
            "right_elbow_angle": 163.0,
            "torso_angle": 5.0,
            "shoulder_hip_alignment": 0.01,
        },
        is_valid=True,
        confidence=0.92,
    )


@pytest.fixture
def squat_bottom_pose():
    """A valid squat at bottom position with good form."""
    landmarks = make_squat_bottom_landmarks()
    return PoseResult(
        landmarks=landmarks,
        angles={
            "left_knee_angle": 85.0,
            "right_knee_angle": 83.0,
            "left_hip_angle": 80.0,
            "right_hip_angle": 78.0,
            "left_elbow_angle": 155.0,
            "right_elbow_angle": 153.0,
            "torso_angle": 25.0,
            "shoulder_hip_alignment": 0.02,
        },
        is_valid=True,
        confidence=0.90,
    )


@pytest.fixture
def squat_shallow_pose():
    """A squat at BOTTOM phase but too shallow (insufficient depth).
    Knee angle 115° is below BOTTOM_THRESHOLD (120) so it enters BOTTOM phase,
    but above DEPTH_THRESHOLD (110) so it triggers insufficient_depth."""
    landmarks = make_squat_bottom_landmarks()
    return PoseResult(
        landmarks=landmarks,
        angles={
            "left_knee_angle": 115.0,
            "right_knee_angle": 113.0,
            "left_hip_angle": 120.0,
            "right_hip_angle": 118.0,
            "left_elbow_angle": 160.0,
            "right_elbow_angle": 158.0,
            "torso_angle": 15.0,
            "shoulder_hip_alignment": 0.01,
        },
        is_valid=True,
        confidence=0.88,
    )


@pytest.fixture
def squat_forward_lean_pose():
    """A squat with excessive forward lean."""
    landmarks = make_squat_bottom_landmarks()
    return PoseResult(
        landmarks=landmarks,
        angles={
            "left_knee_angle": 88.0,
            "right_knee_angle": 86.0,
            "left_hip_angle": 60.0,
            "right_hip_angle": 58.0,
            "left_elbow_angle": 155.0,
            "right_elbow_angle": 153.0,
            "torso_angle": 55.0,
            "shoulder_hip_alignment": 0.03,
        },
        is_valid=True,
        confidence=0.85,
    )


@pytest.fixture
def invalid_pose():
    """An invalid pose (no person detected)."""
    return PoseResult(
        landmarks={},
        angles={},
        is_valid=False,
        confidence=0.0,
    )


@pytest.fixture
def pushup_plank_pose():
    """A valid push-up in plank position."""
    landmarks = make_pushup_plank_landmarks()
    return PoseResult(
        landmarks=landmarks,
        angles={
            "left_elbow_angle": 170.0,
            "right_elbow_angle": 168.0,
            "left_knee_angle": 175.0,
            "right_knee_angle": 173.0,
            "left_hip_angle": 172.0,
            "right_hip_angle": 170.0,
            "torso_angle": 8.0,
            "shoulder_hip_alignment": 0.01,
        },
        is_valid=True,
        confidence=0.91,
    )


@pytest.fixture
def good_form_analysis():
    """AnalysisResult representing good form."""
    return AnalysisResult(
        exercise_type="squat",
        phase=ExercisePhase.BOTTOM,
        errors=[],
        rep_count=1,
        is_good_form=True,
        angles={"left_knee_angle": 85.0, "right_knee_angle": 83.0},
        feedback_priority=[],
    )


@pytest.fixture
def bad_form_analysis():
    """AnalysisResult with errors."""
    return AnalysisResult(
        exercise_type="squat",
        phase=ExercisePhase.BOTTOM,
        errors=[
            FormError(
                error_type="insufficient_depth",
                message="Go deeper!",
                severity=ErrorSeverity.WARNING,
                body_part="knees",
                current_value=120.0,
                target_value=90.0,
            ),
            FormError(
                error_type="excessive_forward_lean",
                message="Chest up!",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=55.0,
                target_value=45.0,
            ),
        ],
        rep_count=1,
        is_good_form=False,
        angles={"left_knee_angle": 120.0, "torso_angle": 55.0},
        feedback_priority=["excessive_forward_lean", "insufficient_depth"],
    )
