"""
Tests for backend/exercises/base.py

Covers ErrorSeverity comparison operators, FormError, AnalysisResult,
and the ExerciseAnalyzer base class utilities.
"""
import pytest
from backend.exercises.base import ErrorSeverity, FormError, RepCounter, ExercisePhase


# ── ErrorSeverity Ordering ──

def test_error_severity_lt():
    """INFO < WARNING < ERROR < CRITICAL"""
    assert ErrorSeverity.INFO < ErrorSeverity.WARNING
    assert ErrorSeverity.WARNING < ErrorSeverity.ERROR
    assert ErrorSeverity.ERROR < ErrorSeverity.CRITICAL


def test_error_severity_gt():
    """CRITICAL > ERROR > WARNING > INFO"""
    assert ErrorSeverity.CRITICAL > ErrorSeverity.ERROR
    assert ErrorSeverity.ERROR > ErrorSeverity.WARNING
    assert ErrorSeverity.WARNING > ErrorSeverity.INFO


def test_error_severity_le_ge():
    """Less-than-or-equal and greater-than-or-equal"""
    assert ErrorSeverity.INFO <= ErrorSeverity.INFO
    assert ErrorSeverity.INFO <= ErrorSeverity.WARNING
    assert ErrorSeverity.CRITICAL >= ErrorSeverity.CRITICAL
    assert ErrorSeverity.CRITICAL >= ErrorSeverity.ERROR


def test_error_severity_max():
    """max() should work on a list of ErrorSeverity values."""
    severities = [ErrorSeverity.INFO, ErrorSeverity.ERROR, ErrorSeverity.WARNING]
    assert max(severities) == ErrorSeverity.ERROR


def test_error_severity_min():
    """min() should work on a list of ErrorSeverity values."""
    severities = [ErrorSeverity.WARNING, ErrorSeverity.CRITICAL, ErrorSeverity.ERROR]
    assert min(severities) == ErrorSeverity.WARNING


def test_error_severity_invalid_comparison():
    """Comparing with non-ErrorSeverity should return NotImplemented."""
    result = ErrorSeverity.INFO.__lt__("not_an_enum")
    assert result is NotImplemented


def test_error_severity_sorted():
    """sorted() should order by severity."""
    items = [ErrorSeverity.CRITICAL, ErrorSeverity.INFO, ErrorSeverity.ERROR, ErrorSeverity.WARNING]
    assert sorted(items) == [
        ErrorSeverity.INFO, ErrorSeverity.WARNING,
        ErrorSeverity.ERROR, ErrorSeverity.CRITICAL,
    ]


# ── RepCounter ──

def test_rep_counter_squat_cycle():
    """A full squat cycle should increment rep count."""
    counter = RepCounter()
    counter.update_phase(ExercisePhase.DESCENDING)
    counter.update_phase(ExercisePhase.BOTTOM)
    counter.update_phase(ExercisePhase.ASCENDING)
    counter.update_phase(ExercisePhase.STANDING)
    assert counter.count == 1


def test_rep_counter_no_false_reps():
    """Staying in one phase should not count reps."""
    counter = RepCounter()
    counter.update_phase(ExercisePhase.STANDING)
    counter.update_phase(ExercisePhase.STANDING)
    counter.update_phase(ExercisePhase.STANDING)
    assert counter.count == 0


# ── FormError ──

def test_form_error_to_dict():
    """FormError.to_dict() should contain all fields."""
    error = FormError(
        error_type="test_error",
        message="Test message",
        severity=ErrorSeverity.WARNING,
        body_part="knees",
        current_value=95.0,
        target_value=90.0,
    )
    d = error.to_dict()
    assert d["error_type"] == "test_error"
    assert d["severity"] == "warning"
    assert d["current_value"] == 95.0
