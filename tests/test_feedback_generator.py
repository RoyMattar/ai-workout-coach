"""
Tests for backend/feedback_generator.py

Covers quick feedback generation for good and bad form, unknown error
fallback, response parsing, API key absence, and encouragement rotation.
"""
import pytest

from backend.feedback_generator import FeedbackGenerator, FeedbackResult
from backend.exercises.base import (
    AnalysisResult,
    ExercisePhase,
    FormError,
    ErrorSeverity,
)


# ── Fixtures ──

@pytest.fixture
def generator():
    """Create a FeedbackGenerator with no API key (template-only mode)."""
    return FeedbackGenerator(api_key=None)


# ── Tests ──

def test_quick_feedback_good_form(generator, good_form_analysis):
    """Good form analysis should produce positive spoken feedback."""
    result = generator.generate_quick_feedback(good_form_analysis)

    assert isinstance(result, FeedbackResult)
    assert "perfect" in result.spoken_feedback.lower() or "great" in result.spoken_feedback.lower()
    assert result.is_cached is True


def test_quick_feedback_known_error(generator, bad_form_analysis):
    """Bad form analysis with known error types should return template feedback."""
    result = generator.generate_quick_feedback(bad_form_analysis)

    assert isinstance(result, FeedbackResult)
    assert len(result.spoken_feedback) > 0
    # The first error is "insufficient_depth" and feedback should match its template
    assert result.spoken_feedback == FeedbackGenerator.QUICK_FEEDBACK["insufficient_depth"]["spoken"]


def test_quick_feedback_unknown_error(generator):
    """An analysis with an unknown error type should produce fallback feedback."""
    analysis = AnalysisResult(
        exercise_type="squat",
        phase=ExercisePhase.BOTTOM,
        errors=[
            FormError(
                error_type="some_unknown_error",
                message="Something unusual happened.",
                severity=ErrorSeverity.WARNING,
                body_part="full_body",
            )
        ],
        rep_count=1,
        is_good_form=False,
        angles={},
        feedback_priority=["some_unknown_error"],
    )

    result = generator.generate_quick_feedback(analysis)

    assert isinstance(result, FeedbackResult)
    assert "check your form" in result.spoken_feedback.lower()


def test_parse_feedback_response(generator):
    """_parse_feedback_response should correctly parse a well-formatted response."""
    response_text = (
        "SPOKEN: Keep your chest up!\n"
        "DETAILED: You are leaning too far forward.\n"
        "TIP: Focus on looking slightly upward.\n"
        "ENCOURAGEMENT: You've got this!"
    )

    result = generator._parse_feedback_response(response_text)

    assert isinstance(result, FeedbackResult)
    assert result.spoken_feedback == "Keep your chest up!"
    assert result.detailed_feedback == "You are leaning too far forward."
    assert result.tip == "Focus on looking slightly upward."
    assert result.encouragement == "You've got this!"
    assert result.is_cached is False


def test_no_api_key_fallback(good_form_analysis, monkeypatch):
    """FeedbackGenerator with empty API key should still generate quick feedback."""
    # Temporarily clear any API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from backend.config import get_settings
    get_settings.cache_clear()

    generator = FeedbackGenerator(api_key="")
    assert generator.client is None

    result = generator.generate_quick_feedback(good_form_analysis)
    assert isinstance(result, FeedbackResult)
    assert len(result.spoken_feedback) > 0

    get_settings.cache_clear()  # Reset for other tests


def test_encouragement_rotation(generator):
    """Calling _get_encouragement multiple times should cycle through messages."""
    seen = set()
    num_messages = len(FeedbackGenerator.ENCOURAGEMENTS)

    for _ in range(num_messages):
        msg = generator._get_encouragement()
        seen.add(msg)

    # Should have seen multiple different messages
    assert len(seen) > 1, "Expected multiple different encouragement messages"
    assert len(seen) == num_messages, (
        f"Expected {num_messages} unique messages, got {len(seen)}"
    )


# ═══════════════════════════════════════════════════════
# Vision Analysis Tests
# ═══════════════════════════════════════════════════════

def test_parse_vision_response_valid(generator):
    """A well-formatted vision response should parse correctly."""
    response = (
        "ASSESSMENT: minor_issues\n"
        "CONFIDENCE: high\n"
        "OBSERVATIONS: knees caving slightly, good depth achieved\n"
        "SUGGESTIONS: push knees outward, maintain chest position"
    )
    result = generator._parse_vision_response(response)

    assert result.form_assessment == "minor_issues"
    assert result.confidence == "high"
    assert len(result.observations) == 2
    assert len(result.suggestions) == 2
    assert "knees caving slightly" in result.observations


def test_parse_vision_response_good_form(generator):
    """Assessment 'good' should parse correctly."""
    response = (
        "ASSESSMENT: good\n"
        "CONFIDENCE: high\n"
        "OBSERVATIONS: excellent form throughout\n"
        "SUGGESTIONS: maintain current technique"
    )
    result = generator._parse_vision_response(response)
    assert result.form_assessment == "good"


def test_parse_vision_response_malformed(generator):
    """A malformed response should use defaults, not crash."""
    response = "This is just random text with no structured format."
    result = generator._parse_vision_response(response)

    assert result.form_assessment == "minor_issues"  # default
    assert result.confidence == "medium"  # default
    assert isinstance(result.observations, list)
    assert isinstance(result.suggestions, list)


def test_vision_analysis_no_client():
    """Vision analysis should return None when no API client."""
    generator = FeedbackGenerator(api_key="")
    assert generator.client is None


def test_vision_system_prompt_exists(generator):
    """Vision system prompt should be non-empty."""
    prompt = generator._get_vision_system_prompt()
    assert len(prompt) > 50
    assert "ASSESSMENT" in prompt
    assert "OBSERVATIONS" in prompt


def test_vision_analysis_result_to_dict(generator):
    """VisionAnalysisResult.to_dict() should contain expected keys."""
    from backend.feedback_generator import VisionAnalysisResult
    result = VisionAnalysisResult(
        form_assessment="good",
        observations=["great depth"],
        suggestions=["keep it up"],
        confidence="high",
    )
    d = result.to_dict()
    assert d["form_assessment"] == "good"
    assert d["confidence"] == "high"
    assert len(d["observations"]) == 1
