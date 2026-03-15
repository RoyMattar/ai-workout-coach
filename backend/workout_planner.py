"""
AI Workout Plan Generator

Uses GPT-4o-mini to generate personalized, adaptive workout plans
based on the user's session history. Plans adapt over time:

- Improving form scores → increase difficulty/reps
- Consistent errors → add corrective focus notes
- New personal bests → celebrate + push harder
- Declining scores → suggest rest or reduce volume
"""
import json
import logging
import time
from typing import Optional

import openai

from .config import get_settings
from .database import Database

logger = logging.getLogger(__name__)


class WorkoutPlanner:
    """Generates adaptive workout plans using LLM analysis of session history."""

    def __init__(self, db: Database, api_key: Optional[str] = None):
        settings = get_settings()
        self.db = db
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = settings.openai_model
        self.client = None

        if self.api_key:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def generate_plan(self, difficulty: str = "intermediate") -> Optional[dict]:
        """
        Generate a personalized workout plan based on session history.

        Args:
            difficulty: beginner, intermediate, or advanced

        Returns:
            Structured workout plan dict, or None if generation fails
        """
        if not self.client:
            return self._generate_fallback_plan(difficulty)

        # Gather context from database
        stats = self.db.get_session_stats()
        recent_sessions = self.db.get_sessions(limit=10)
        common_errors = self.db.get_common_errors(limit=5)

        prompt = self._build_plan_prompt(stats, recent_sessions, common_errors, difficulty)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_planner_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.7,
                response_format={"type": "json_object"},
            )

            plan_text = response.choices[0].message.content
            plan = json.loads(plan_text)

            # Save to database
            self.db.save_plan(json.dumps(plan))
            logger.info(f"AI workout plan generated: {plan.get('plan_name', 'unnamed')}")

            return plan

        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return self._generate_fallback_plan(difficulty)

    def _get_planner_system_prompt(self) -> str:
        return """You are an expert fitness coach creating personalized workout plans.
Analyze the user's workout history and generate an adaptive plan.

You MUST respond with a valid JSON object in this exact format:
{
    "plan_name": "descriptive name for this plan",
    "difficulty": "beginner/intermediate/advanced",
    "exercises": [
        {
            "exercise": "squat",
            "sets": 3,
            "target_reps": 12,
            "rest_seconds": 60,
            "focus": "specific focus area based on their history",
            "notes": "personalized note about their progress"
        }
    ],
    "estimated_duration_minutes": 20,
    "ai_notes": "explanation of why this plan was designed this way",
    "motivation": "personalized motivational message"
}

Guidelines:
- Base the plan on their actual performance data
- If they have common errors, add notes about fixing them
- If form scores are improving, increase challenge
- If scores are declining, reduce volume and focus on form
- Always be encouraging and specific
- Include 2-4 exercises per plan
- Supported exercises: squat, pushup (more coming soon)"""

    def _build_plan_prompt(
        self,
        stats: dict,
        recent_sessions: list[dict],
        common_errors: list[dict],
        difficulty: str,
    ) -> str:
        prompt = f"Difficulty level: {difficulty}\n\n"

        if stats["total_sessions"] == 0:
            prompt += "This is a NEW user with no workout history. Create a beginner-friendly starter plan.\n"
            return prompt

        prompt += f"""Workout History Summary:
- Total sessions: {stats['total_sessions']}
- Total reps: {stats['total_reps']}
- Average form score: {stats['avg_form_score']}%
- Total workout time: {stats['total_duration_minutes']} minutes

Per-exercise breakdown:
"""
        for exercise, data in stats.get("by_exercise", {}).items():
            prompt += f"  {exercise}: {data['sessions']} sessions, {data['reps']} reps, avg score {data['avg_score']}%\n"

        if common_errors:
            prompt += "\nMost common form errors:\n"
            for error in common_errors:
                prompt += f"  - {error['error_type']}: {error['count']} occurrences\n"

        if stats.get("score_trend"):
            scores = [s["score"] for s in stats["score_trend"]]
            if len(scores) >= 3:
                recent_avg = sum(scores[-3:]) / 3
                older_avg = sum(scores[:3]) / 3
                if recent_avg > older_avg + 5:
                    prompt += "\nTrend: Form is IMPROVING! Consider increasing difficulty.\n"
                elif recent_avg < older_avg - 5:
                    prompt += "\nTrend: Form is DECLINING. Consider reducing volume and focusing on technique.\n"
                else:
                    prompt += "\nTrend: Form is stable. Maintain current level or slightly increase.\n"

        prompt += "\nGenerate an appropriate workout plan based on this data."
        return prompt

    def _generate_fallback_plan(self, difficulty: str) -> dict:
        """Generate a basic plan without LLM (when no API key)."""
        reps = {"beginner": 8, "intermediate": 12, "advanced": 15}.get(difficulty, 12)
        sets = {"beginner": 2, "intermediate": 3, "advanced": 4}.get(difficulty, 3)

        plan = {
            "plan_name": f"{difficulty.title()} Bodyweight Workout",
            "difficulty": difficulty,
            "exercises": [
                {
                    "exercise": "squat",
                    "sets": sets,
                    "target_reps": reps,
                    "rest_seconds": 60,
                    "focus": "Full depth with proper knee tracking",
                    "notes": "Focus on controlled movement",
                },
                {
                    "exercise": "pushup",
                    "sets": sets,
                    "target_reps": max(reps - 2, 5),
                    "rest_seconds": 60,
                    "focus": "Keep body straight, full range of motion",
                    "notes": "Modify on knees if needed",
                },
            ],
            "estimated_duration_minutes": 15 + (sets * 3),
            "ai_notes": "Standard bodyweight plan. Complete workout sessions to get AI-personalized plans.",
            "motivation": "Every rep is progress. Let's get started!",
        }

        return plan
