"""
Tests for backend/orchestrator.py

Covers orchestrator initialization, blank frame processing, result structure,
initial session stats, exercise switching, and model info retrieval.
"""
import numpy as np
import pytest

from backend.orchestrator import (
    WorkoutOrchestrator,
    ExerciseType,
    OrchestrationResult,
    SessionStats,
    PipelineTiming,
)


# ── Fixtures ──

@pytest.fixture
def orchestrator():
    """Create and yield a WorkoutOrchestrator, then close it."""
    orch = WorkoutOrchestrator(exercise_type=ExerciseType.SQUAT)
    yield orch
    orch.close()


# ── Tests ──

def test_orchestrator_init(orchestrator):
    """WorkoutOrchestrator should initialize without error."""
    assert orchestrator is not None
    assert orchestrator.current_exercise == ExerciseType.SQUAT
    assert orchestrator.pose_estimator is not None
    assert orchestrator.form_classifier is not None
    assert orchestrator.feedback_generator is not None
    assert orchestrator.tts_engine is not None


def test_process_frame_blank(orchestrator):
    """Processing a blank frame should return a result with is_valid=False."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = orchestrator.process_frame(blank_frame)

    assert isinstance(result, OrchestrationResult)
    assert result.pose_result.is_valid is False


def test_process_frame_result_structure(orchestrator):
    """OrchestrationResult should have all expected fields."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = orchestrator.process_frame(blank_frame)

    assert hasattr(result, "pose_result")
    assert hasattr(result, "analysis_result")
    assert hasattr(result, "ml_classification")
    assert hasattr(result, "feedback")
    assert hasattr(result, "session_stats")
    assert hasattr(result, "pipeline_timing")

    assert isinstance(result.session_stats, SessionStats)
    assert isinstance(result.pipeline_timing, PipelineTiming)

    # Verify to_dict works without error
    result_dict = result.to_dict()
    assert "pose" in result_dict
    assert "analysis" in result_dict
    assert "feedback" in result_dict
    assert "session" in result_dict
    assert "pipeline_timing" in result_dict


def test_session_stats_initial(orchestrator):
    """Initial session stats should have 0 reps and 0 errors."""
    stats = orchestrator.session_stats

    assert stats.total_reps == 0
    assert stats.good_form_reps == 0
    assert stats.errors_by_type == {}
    assert stats.frames_processed == 0


def test_exercise_switching(orchestrator):
    """Switching exercise type should update the current exercise and reset state."""
    orchestrator.set_exercise(ExerciseType.PUSHUP)

    assert orchestrator.current_exercise == ExerciseType.PUSHUP
    assert orchestrator.session_stats.exercise_type == "pushup"
    assert orchestrator.session_stats.total_reps == 0


def test_get_model_info(orchestrator):
    """get_model_info() should return a dict with 5 model entries."""
    info = orchestrator.get_model_info()

    assert isinstance(info, dict)
    assert "pose_estimation" in info
    assert "form_classifier" in info
    assert "feedback_generator" in info
    assert "vision_analyzer" in info
    assert "tts_engine" in info
    assert len(info) == 5


def test_last_angles_updated_after_processing(orchestrator):
    """After processing a frame, _last_angles should be populated."""
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    orchestrator.process_frame(blank_frame)
    # Even for blank frame, _last_angles should be set (empty dict if no pose)
    assert isinstance(orchestrator._last_angles, dict)


def test_vision_cooldown_initial(orchestrator):
    """_last_vision_time should start at 0."""
    assert orchestrator._last_vision_time == 0.0


def test_reset_clears_state(orchestrator):
    """reset_session should clear all state."""
    orchestrator._last_errors = ["test"]
    orchestrator._last_tts_text = "test"
    orchestrator.reset_session()
    assert orchestrator._last_errors == []
    assert orchestrator._last_tts_text == ""
    assert orchestrator.session_stats.total_reps == 0
