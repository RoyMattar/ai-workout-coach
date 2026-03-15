"""
Tests for backend/pose_estimator.py

Covers angle calculation, pose processing on blank frames, PoseResult
structure, and confidence calculation.
"""
import numpy as np
import pytest

from backend.pose_estimator import PoseEstimator, Landmark, PoseResult


# ── Angle calculation tests (static method, no model needed) ──

def test_calculate_angle_straight_line():
    """Three collinear points should give approximately 180 degrees."""
    p1 = Landmark(x=0.0, y=0.0, z=0.0, visibility=0.9)
    p2 = Landmark(x=0.5, y=0.0, z=0.0, visibility=0.9)
    p3 = Landmark(x=1.0, y=0.0, z=0.0, visibility=0.9)

    angle = PoseEstimator._calculate_angle(p1, p2, p3)
    assert abs(angle - 180.0) < 1.0, f"Expected ~180, got {angle}"


def test_calculate_angle_right_angle():
    """Known right-angle geometry should give approximately 90 degrees."""
    p1 = Landmark(x=1.0, y=0.0, z=0.0, visibility=0.9)
    p2 = Landmark(x=0.0, y=0.0, z=0.0, visibility=0.9)
    p3 = Landmark(x=0.0, y=1.0, z=0.0, visibility=0.9)

    angle = PoseEstimator._calculate_angle(p1, p2, p3)
    assert abs(angle - 90.0) < 1.0, f"Expected ~90, got {angle}"


def test_calculate_angle_zero_distance():
    """Coincident points should not crash (returns some finite value)."""
    p1 = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)
    p2 = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)
    p3 = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.9)

    # Should not raise an exception due to the epsilon in the denominator
    angle = PoseEstimator._calculate_angle(p1, p2, p3)
    assert isinstance(angle, float)
    assert np.isfinite(angle)


# ── Process frame tests (requires model download) ──

def test_process_frame_blank_image():
    """An all-black frame should return PoseResult with is_valid=False."""
    estimator = PoseEstimator()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = estimator.process_frame(blank_frame)

    assert isinstance(result, PoseResult)
    assert result.is_valid is False
    assert result.confidence == 0.0
    assert result.landmarks == {}
    assert result.angles == {}

    estimator.close()


# ── PoseResult structure tests ──

def test_pose_result_structure():
    """Verify PoseResult dataclass has the expected fields and types."""
    landmarks = {"nose": Landmark(x=0.5, y=0.15, z=0.0, visibility=0.95)}
    angles = {"left_knee_angle": 170.0}

    result = PoseResult(
        landmarks=landmarks,
        angles=angles,
        is_valid=True,
        confidence=0.9,
    )

    assert result.is_valid is True
    assert "nose" in result.landmarks
    assert "left_knee_angle" in result.angles
    assert result.confidence == 0.9
    assert result.raw_landmarks is None  # default


def test_calculate_confidence():
    """_calculate_confidence should return the average of visibilities."""
    estimator = PoseEstimator()

    landmarks = {
        "a": Landmark(x=0.0, y=0.0, z=0.0, visibility=0.8),
        "b": Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0),
        "c": Landmark(x=0.0, y=0.0, z=0.0, visibility=0.6),
    }

    confidence = estimator._calculate_confidence(landmarks)
    expected = (0.8 + 1.0 + 0.6) / 3.0
    assert abs(confidence - expected) < 1e-6

    # Empty landmarks should return 0.0
    assert estimator._calculate_confidence({}) == 0.0

    estimator.close()
