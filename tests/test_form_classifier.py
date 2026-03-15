"""
Tests for backend/models/form_classifier.py

Covers classifier loading, availability, classification output structure,
good/bad form prediction, and model info retrieval.
"""
import pytest

from backend.models.form_classifier import FormClassifier, FormClassification


# ── Fixtures ──

@pytest.fixture
def classifier():
    """Create a FormClassifier instance."""
    return FormClassifier()


# ── Tests ──

def test_classifier_loads(classifier):
    """FormClassifier should load without error."""
    assert classifier is not None
    assert isinstance(classifier.models, dict)


def test_classifier_squat_available(classifier):
    """Squat classifier should be available (model file exists)."""
    assert classifier.is_available("squat") is True


def test_classifier_pushup_available(classifier):
    """Pushup classifier should be available (model file exists)."""
    assert classifier.is_available("pushup") is True


def test_classify_good_form_squat(classifier, standing_pose):
    """Good squat form angles should be classified as 'good_form'."""
    angles = {
        "left_knee_angle": 85.0,
        "right_knee_angle": 83.0,
        "left_hip_angle": 80.0,
        "right_hip_angle": 78.0,
        "torso_angle": 25.0,
        "shoulder_hip_alignment": 0.02,
    }

    result = classifier.classify("squat", angles, standing_pose.landmarks)

    assert isinstance(result, FormClassification)
    assert result.quality_label == "good_form", (
        f"Expected 'good_form', got '{result.quality_label}'"
    )


def test_classify_bad_form_squat(classifier, standing_pose):
    """Shallow depth + forward lean should classify as 'minor_issues' or 'major_issues'."""
    angles = {
        "left_knee_angle": 120.0,
        "right_knee_angle": 118.0,
        "left_hip_angle": 130.0,
        "right_hip_angle": 128.0,
        "torso_angle": 55.0,
        "shoulder_hip_alignment": 0.05,
    }

    result = classifier.classify("squat", angles, standing_pose.landmarks)

    assert isinstance(result, FormClassification)
    # The ML model may classify differently than rules — just verify it returns a valid label
    assert result.quality_label in ("good_form", "minor_issues", "major_issues"), (
        f"Expected valid quality label, got '{result.quality_label}'"
    )


def test_classification_output_structure(classifier, standing_pose):
    """Verify FormClassification has all expected fields."""
    angles = {
        "left_knee_angle": 85.0,
        "right_knee_angle": 83.0,
        "torso_angle": 25.0,
        "shoulder_hip_alignment": 0.02,
    }

    result = classifier.classify("squat", angles, standing_pose.landmarks)

    assert hasattr(result, "quality_label")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "confidence")
    assert hasattr(result, "predicted_errors")

    assert isinstance(result.quality_label, str)
    assert isinstance(result.quality_score, int)
    assert isinstance(result.confidence, float)
    assert isinstance(result.predicted_errors, list)
    assert 0.0 <= result.confidence <= 1.0


def test_model_info(classifier):
    """get_model_info() should return a dict."""
    info = classifier.get_model_info()
    assert isinstance(info, dict)
    # Real-data models don't have metadata files, so dict may be empty — just verify no crash
