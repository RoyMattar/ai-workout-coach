"""
Tests for backend/coach_personas.py

Covers persona definitions, system prompt modification, voice mapping,
and the persona registry.
"""
import pytest
from backend.coach_personas import (
    CoachPersona,
    PERSONAS,
    get_persona,
    list_personas,
    DEFAULT_COACH,
)


def test_personas_registry_not_empty():
    """At least one persona should be registered."""
    assert len(PERSONAS) >= 5


def test_all_persona_ids():
    """Expected persona IDs should exist."""
    expected = {"coach_pro", "drill_sergeant", "zen_master", "hype_beast", "pop_diva"}
    assert expected.issubset(set(PERSONAS.keys()))


def test_default_coach_is_coach_pro():
    """Default coach should be Coach Pro."""
    assert DEFAULT_COACH.id == "coach_pro"


def test_get_persona_valid():
    """get_persona with valid ID returns correct persona."""
    persona = get_persona("drill_sergeant")
    assert persona.id == "drill_sergeant"
    assert persona.name == "Drill Sergeant"


def test_get_persona_invalid_returns_default():
    """get_persona with invalid ID returns default coach."""
    persona = get_persona("nonexistent_coach")
    assert persona.id == DEFAULT_COACH.id


def test_each_persona_has_unique_voice():
    """Each persona should use a different OpenAI voice."""
    voices = [p.voice_id for p in PERSONAS.values()]
    assert len(voices) == len(set(voices)), "Duplicate voices found"


def test_each_persona_has_system_prompt():
    """Each persona should have a non-empty system prompt modifier."""
    for persona_id, persona in PERSONAS.items():
        assert persona.system_prompt_modifier, f"{persona_id} missing system_prompt_modifier"


def test_each_persona_has_encouragements():
    """Each persona should have at least 3 encouragements."""
    for persona_id, persona in PERSONAS.items():
        assert len(persona.encouragements) >= 3, (
            f"{persona_id} has only {len(persona.encouragements)} encouragements"
        )


def test_each_persona_has_theme_color():
    """Each persona should have a theme color."""
    for persona_id, persona in PERSONAS.items():
        assert persona.theme_color.startswith("#"), f"{persona_id} bad theme_color"


def test_list_personas_returns_dicts():
    """list_personas should return a list of dicts with expected keys."""
    result = list_personas()
    assert isinstance(result, list)
    assert len(result) >= 5
    for p in result:
        assert "id" in p
        assert "name" in p
        assert "voice_id" in p


def test_persona_to_dict():
    """CoachPersona.to_dict() should contain expected keys."""
    persona = get_persona("zen_master")
    d = persona.to_dict()
    assert d["id"] == "zen_master"
    assert d["name"] == "Zen Master"
    assert d["voice_id"] == "shimmer"
    assert d["theme_color"] == "#8b5cf6"


def test_persona_voice_ids_are_valid_openai_voices():
    """All persona voice IDs should be valid OpenAI TTS voices."""
    valid_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    for persona_id, persona in PERSONAS.items():
        assert persona.voice_id in valid_voices, (
            f"{persona_id} has invalid voice: {persona.voice_id}"
        )


def test_pop_diva_personality():
    """Pop Diva should have pop-star-related content."""
    persona = get_persona("pop_diva")
    all_text = " ".join(persona.catchphrases + persona.encouragements)
    # Should contain pop-star-ish language
    assert any(word in all_text.lower() for word in ["star", "fabulous", "encore", "honey"])


def test_drill_sergeant_personality():
    """Drill Sergeant should have military-style content."""
    persona = get_persona("drill_sergeant")
    all_text = " ".join(persona.catchphrases + persona.encouragements)
    assert any(word in all_text.lower() for word in ["soldier", "excuses", "move"])
