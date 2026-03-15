"""
Tests for backend/tts_engine.py

Covers initialization, availability check, empty string handling,
and cache key determinism.
"""
import pytest

from backend.tts_engine import TTSEngine


# ── Fixtures ──

@pytest.fixture
def engine():
    """Create a TTSEngine instance."""
    return TTSEngine()


# ── Tests ──

def test_tts_engine_init(engine):
    """TTSEngine should initialize without error."""
    assert engine is not None
    assert engine.cache_dir.exists()


def test_tts_available(engine):
    """engine.is_available should be True when gTTS is installed."""
    assert engine.is_available is True


def test_tts_empty_string(engine):
    """Synthesizing an empty string should return None without error."""
    result = engine.synthesize("")
    assert result is None

    result = engine.synthesize("   ")
    assert result is None


def test_tts_cache_key(engine):
    """Identical texts should produce the same cache key; different texts should differ."""
    key1 = engine._cache_key("hello world")
    key2 = engine._cache_key("hello world")
    key3 = engine._cache_key("goodbye world")

    assert key1 == key2, "Same text should produce the same cache key"
    assert key1 != key3, "Different texts should produce different cache keys"
