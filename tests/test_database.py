"""
Tests for backend/database.py

Covers CRUD operations, session stats, achievements, and workout plans.
"""
import json
import tempfile
import time
import pytest

from backend.database import Database, SessionRecord, ACHIEVEMENT_DEFS


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(db_path=f.name)
        yield db


@pytest.fixture
def sample_session():
    return SessionRecord(
        exercise_type="squat",
        coach_persona="drill_sergeant",
        start_time=time.time() - 300,
        end_time=time.time(),
        total_reps=15,
        good_form_reps=12,
        form_score=80.0,
        errors_json=json.dumps({"insufficient_depth": 3, "knee_cave": 1}),
        pipeline_avg_latency_ms=45.2,
    )


# ── Session CRUD ──

def test_save_and_retrieve_session(db, sample_session):
    """Save a session and retrieve it."""
    session_id = db.save_session(sample_session)
    assert session_id > 0

    sessions = db.get_sessions(limit=1)
    assert len(sessions) == 1
    assert sessions[0]["exercise_type"] == "squat"
    assert sessions[0]["total_reps"] == 15


def test_session_count(db, sample_session):
    """Session count should increment."""
    assert db.get_session_count() == 0
    db.save_session(sample_session)
    assert db.get_session_count() == 1
    db.save_session(sample_session)
    assert db.get_session_count() == 2


def test_session_stats_empty(db):
    """Stats on empty database should return zeros."""
    stats = db.get_session_stats()
    assert stats["total_sessions"] == 0
    assert stats["total_reps"] == 0
    assert stats["avg_form_score"] == 0


def test_session_stats_with_data(db, sample_session):
    """Stats should aggregate correctly."""
    db.save_session(sample_session)
    db.save_session(sample_session)

    stats = db.get_session_stats()
    assert stats["total_sessions"] == 2
    assert stats["total_reps"] == 30
    assert stats["avg_form_score"] == 80.0


def test_common_errors(db, sample_session):
    """Common errors should aggregate across sessions."""
    db.save_session(sample_session)
    db.save_session(sample_session)

    errors = db.get_common_errors()
    error_types = [e["error_type"] for e in errors]
    assert "insufficient_depth" in error_types


def test_session_to_dict(sample_session):
    """SessionRecord.to_dict() should contain expected fields."""
    d = sample_session.to_dict()
    assert "exercise_type" in d
    assert "duration_seconds" in d
    assert "form_score" in d
    assert "errors" in d
    assert isinstance(d["errors"], dict)


# ── Achievements ──

def test_unlock_achievement(db):
    """Unlocking a new achievement should return True."""
    assert db.unlock_achievement("first_rep") is True


def test_duplicate_achievement(db):
    """Unlocking the same achievement twice should return False."""
    db.unlock_achievement("first_rep")
    assert db.unlock_achievement("first_rep") is False


def test_get_achievements(db):
    """Retrieved achievements should include definition metadata."""
    db.unlock_achievement("first_rep")
    achievements = db.get_achievements()
    assert len(achievements) == 1
    assert achievements[0]["achievement_type"] == "first_rep"
    assert "name" in achievements[0]  # From ACHIEVEMENT_DEFS


def test_check_achievements_first_session(db, sample_session):
    """First session should unlock 'first_session' and 'first_rep'."""
    db.save_session(sample_session)
    unlocked = db.check_and_unlock_achievements(sample_session)
    assert "first_session" in unlocked
    assert "first_rep" in unlocked


def test_achievement_defs_complete():
    """All achievement definitions should have required fields."""
    for key, defn in ACHIEVEMENT_DEFS.items():
        assert "name" in defn, f"{key} missing 'name'"
        assert "description" in defn, f"{key} missing 'description'"
        assert "icon" in defn, f"{key} missing 'icon'"


# ── Workout Plans ──

def test_save_and_get_plan(db):
    """Save a plan and retrieve it."""
    plan = {"plan_name": "Test Plan", "exercises": []}
    plan_id = db.save_plan(json.dumps(plan))
    assert plan_id > 0

    current = db.get_current_plan()
    assert current is not None
    assert current["plan"]["plan_name"] == "Test Plan"
    assert current["status"] == "active"


def test_new_plan_archives_old(db):
    """Saving a new plan should archive the old one."""
    db.save_plan(json.dumps({"plan_name": "Plan 1"}))
    db.save_plan(json.dumps({"plan_name": "Plan 2"}))

    current = db.get_current_plan()
    assert current["plan"]["plan_name"] == "Plan 2"


def test_no_current_plan(db):
    """When no plans exist, get_current_plan returns None."""
    assert db.get_current_plan() is None
