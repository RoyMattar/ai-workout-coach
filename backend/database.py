"""
SQLite Database for Persistent Storage

Stores workout sessions, AI-generated plans, achievements, and user
preferences. Uses Python's built-in sqlite3 module (no extra dependencies).

Tables:
- workout_sessions: completed workout data (reps, form score, errors)
- workout_plans: AI-generated adaptive workout plans
- achievements: unlocked badges and milestones
"""
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "workout_coach.db"


@dataclass
class SessionRecord:
    """A completed workout session."""
    id: Optional[int] = None
    exercise_type: str = ""
    coach_persona: str = "coach_pro"
    start_time: float = 0.0
    end_time: float = 0.0
    total_reps: int = 0
    good_form_reps: int = 0
    form_score: float = 0.0
    errors_json: str = "{}"
    pipeline_avg_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "exercise_type": self.exercise_type,
            "coach_persona": self.coach_persona,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.end_time - self.start_time, 1),
            "total_reps": self.total_reps,
            "good_form_reps": self.good_form_reps,
            "form_score": round(self.form_score, 1),
            "errors": json.loads(self.errors_json),
            "pipeline_avg_latency_ms": round(self.pipeline_avg_latency_ms, 1),
        }


@dataclass
class WorkoutPlan:
    """An AI-generated workout plan."""
    id: Optional[int] = None
    plan_json: str = "{}"
    generated_at: float = 0.0
    status: str = "active"  # active, completed, archived

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plan": json.loads(self.plan_json),
            "generated_at": self.generated_at,
            "status": self.status,
        }


@dataclass
class Achievement:
    """An unlocked achievement."""
    id: Optional[int] = None
    achievement_type: str = ""
    earned_at: float = 0.0
    details_json: str = "{}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "achievement_type": self.achievement_type,
            "earned_at": self.earned_at,
            "details": json.loads(self.details_json),
        }


# Achievement definitions
ACHIEVEMENT_DEFS = {
    "first_rep": {"name": "First Rep", "description": "Complete your first rep", "icon": "🎯"},
    "perfect_ten": {"name": "Perfect Ten", "description": "10 reps with perfect form", "icon": "💯"},
    "iron_will": {"name": "Iron Will", "description": "3 sessions in one week", "icon": "🔥"},
    "form_master": {"name": "Form Master", "description": "90%+ form score 5 sessions straight", "icon": "🏆"},
    "century_club": {"name": "Century Club", "description": "100 total reps across sessions", "icon": "💪"},
    "variety_pack": {"name": "Variety Pack", "description": "Try all exercise types", "icon": "🎨"},
    "first_session": {"name": "Getting Started", "description": "Complete your first session", "icon": "⭐"},
    "streak_3": {"name": "On a Roll", "description": "3 sessions on consecutive days", "icon": "🔥"},
}


class Database:
    """SQLite database manager for the workout coach."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()
        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workout_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exercise_type TEXT NOT NULL,
                    coach_persona TEXT DEFAULT 'coach_pro',
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    total_reps INTEGER DEFAULT 0,
                    good_form_reps INTEGER DEFAULT 0,
                    form_score REAL DEFAULT 0.0,
                    errors_json TEXT DEFAULT '{}',
                    pipeline_avg_latency_ms REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS workout_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_json TEXT NOT NULL,
                    generated_at REAL NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS achievements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    achievement_type TEXT NOT NULL UNIQUE,
                    earned_at REAL NOT NULL,
                    details_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    # ── Sessions ──

    def save_session(self, session: SessionRecord) -> int:
        """Save a completed workout session. Returns the session ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO workout_sessions
                   (exercise_type, coach_persona, start_time, end_time,
                    total_reps, good_form_reps, form_score,
                    errors_json, pipeline_avg_latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.exercise_type,
                    session.coach_persona,
                    session.start_time,
                    session.end_time,
                    session.total_reps,
                    session.good_form_reps,
                    session.form_score,
                    session.errors_json,
                    session.pipeline_avg_latency_ms,
                ),
            )
            session_id = cursor.lastrowid
            logger.info(f"Session saved: id={session_id}, exercise={session.exercise_type}, reps={session.total_reps}")
            return session_id

    def get_sessions(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Get recent workout sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM workout_sessions
                   ORDER BY start_time DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            return [self._row_to_session(r).to_dict() for r in rows]

    def get_session_count(self) -> int:
        """Get total number of sessions."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM workout_sessions").fetchone()
            return row["cnt"]

    def get_session_stats(self) -> dict:
        """Get aggregate stats across all sessions."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_sessions,
                    COALESCE(SUM(total_reps), 0) as total_reps,
                    COALESCE(SUM(good_form_reps), 0) as total_good_reps,
                    COALESCE(AVG(form_score), 0) as avg_form_score,
                    COALESCE(SUM(end_time - start_time), 0) as total_duration,
                    COALESCE(AVG(pipeline_avg_latency_ms), 0) as avg_latency
                FROM workout_sessions
            """).fetchone()

            # Get per-exercise stats
            exercise_rows = conn.execute("""
                SELECT exercise_type,
                       COUNT(*) as sessions,
                       SUM(total_reps) as reps,
                       AVG(form_score) as avg_score
                FROM workout_sessions
                GROUP BY exercise_type
            """).fetchall()

            # Get recent form score trend (last 10 sessions)
            trend_rows = conn.execute("""
                SELECT form_score, start_time, exercise_type
                FROM workout_sessions
                ORDER BY start_time DESC LIMIT 10
            """).fetchall()

            return {
                "total_sessions": row["total_sessions"],
                "total_reps": row["total_reps"],
                "total_good_reps": row["total_good_reps"],
                "avg_form_score": round(row["avg_form_score"], 1),
                "total_duration_minutes": round(row["total_duration"] / 60, 1),
                "avg_pipeline_latency_ms": round(row["avg_latency"], 1),
                "by_exercise": {
                    r["exercise_type"]: {
                        "sessions": r["sessions"],
                        "reps": r["reps"],
                        "avg_score": round(r["avg_score"], 1),
                    }
                    for r in exercise_rows
                },
                "score_trend": [
                    {"score": r["form_score"], "time": r["start_time"], "exercise": r["exercise_type"]}
                    for r in reversed(list(trend_rows))
                ],
            }

    def get_common_errors(self, limit: int = 5) -> list[dict]:
        """Get most common errors across all sessions."""
        with self._connect() as conn:
            rows = conn.execute("SELECT errors_json FROM workout_sessions").fetchall()

        error_counts: dict[str, int] = {}
        for row in rows:
            errors = json.loads(row["errors_json"])
            for error_type, count in errors.items():
                error_counts[error_type] = error_counts.get(error_type, 0) + count

        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])
        return [{"error_type": e, "count": c} for e, c in sorted_errors[:limit]]

    # ── Workout Plans ──

    def save_plan(self, plan_json: str) -> int:
        """Save an AI-generated workout plan."""
        # Archive any existing active plans
        with self._connect() as conn:
            conn.execute(
                "UPDATE workout_plans SET status = 'archived' WHERE status = 'active'"
            )
            cursor = conn.execute(
                "INSERT INTO workout_plans (plan_json, generated_at, status) VALUES (?, ?, 'active')",
                (plan_json, time.time()),
            )
            return cursor.lastrowid

    def get_current_plan(self) -> Optional[dict]:
        """Get the current active workout plan."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM workout_plans WHERE status = 'active' ORDER BY generated_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return WorkoutPlan(
                    id=row["id"],
                    plan_json=row["plan_json"],
                    generated_at=row["generated_at"],
                    status=row["status"],
                ).to_dict()
            return None

    # ── Achievements ──

    def unlock_achievement(self, achievement_type: str, details: dict = None) -> bool:
        """Unlock an achievement. Returns True if newly unlocked, False if already had."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM achievements WHERE achievement_type = ?",
                (achievement_type,),
            ).fetchone()

            if existing:
                return False

            conn.execute(
                "INSERT INTO achievements (achievement_type, earned_at, details_json) VALUES (?, ?, ?)",
                (achievement_type, time.time(), json.dumps(details or {})),
            )
            logger.info(f"Achievement unlocked: {achievement_type}")
            return True

    def get_achievements(self) -> list[dict]:
        """Get all unlocked achievements."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM achievements ORDER BY earned_at DESC"
            ).fetchall()

        result = []
        for row in rows:
            ach = Achievement(
                id=row["id"],
                achievement_type=row["achievement_type"],
                earned_at=row["earned_at"],
                details_json=row["details_json"],
            ).to_dict()
            # Add definition metadata
            defn = ACHIEVEMENT_DEFS.get(row["achievement_type"], {})
            ach.update(defn)
            result.append(ach)
        return result

    def check_and_unlock_achievements(self, session: SessionRecord) -> list[str]:
        """Check if the latest session unlocks any achievements. Returns newly unlocked types."""
        newly_unlocked = []
        stats = self.get_session_stats()

        # First session
        if stats["total_sessions"] == 1:
            if self.unlock_achievement("first_session"):
                newly_unlocked.append("first_session")

        # First rep
        if session.total_reps > 0:
            if self.unlock_achievement("first_rep"):
                newly_unlocked.append("first_rep")

        # Perfect Ten: 10+ good form reps in a session
        if session.good_form_reps >= 10:
            if self.unlock_achievement("perfect_ten"):
                newly_unlocked.append("perfect_ten")

        # Century Club: 100+ total reps
        if stats["total_reps"] >= 100:
            if self.unlock_achievement("century_club"):
                newly_unlocked.append("century_club")

        # Form Master: 90%+ form score for current session (simplified from 5-session streak)
        if session.form_score >= 90 and session.total_reps >= 5:
            if self.unlock_achievement("form_master"):
                newly_unlocked.append("form_master")

        # Variety Pack: tried multiple exercise types
        if len(stats.get("by_exercise", {})) >= 2:
            if self.unlock_achievement("variety_pack"):
                newly_unlocked.append("variety_pack")

        return newly_unlocked

    # ── Helpers ──

    @staticmethod
    def _row_to_session(row) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            exercise_type=row["exercise_type"],
            coach_persona=row["coach_persona"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            total_reps=row["total_reps"],
            good_form_reps=row["good_form_reps"],
            form_score=row["form_score"],
            errors_json=row["errors_json"],
            pipeline_avg_latency_ms=row["pipeline_avg_latency_ms"],
        )
