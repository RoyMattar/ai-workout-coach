"""
LLM-based Feedback Generator

Generates natural, encouraging, and personalized coaching feedback
using two LLM modes:

1. Text Feedback (GPT-4o-mini): Converts structured error data into
   natural language coaching cues. Fast and cheap.

2. Vision Analysis (GPT-4o): Analyzes actual video frames to provide
   independent form assessment. This is a fundamentally different
   approach — direct visual understanding vs. skeleton-based analysis.
"""
import asyncio
import base64
import logging
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
import openai

from .exercises.base import FormError, AnalysisResult, ErrorSeverity
from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FeedbackResult:
    """Result from feedback generation"""
    spoken_feedback: str
    detailed_feedback: str
    encouragement: str
    tip: str
    is_cached: bool = False


@dataclass
class VisionAnalysisResult:
    """Result from GPT-4o Vision analysis of a video frame."""
    form_assessment: str       # "good", "minor_issues", "major_issues"
    observations: list[str]    # What the model sees
    suggestions: list[str]     # Improvement suggestions
    confidence: str            # "high", "medium", "low"
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "form_assessment": self.form_assessment,
            "observations": self.observations,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }


class FeedbackGenerator:
    """
    Generates personalized coaching feedback using LLM models.

    Two modes:
    1. Text feedback (GPT-4o-mini): structured errors → coaching text
    2. Vision analysis (GPT-4o): video frame → independent form assessment
    """

    QUICK_FEEDBACK = {
        "insufficient_depth": {
            "spoken": "Go deeper! Get those thighs parallel.",
            "detailed": "Your squat depth is too shallow. Aim to get your thighs parallel to the ground.",
            "tip": "Try widening your stance slightly or pointing your toes out more."
        },
        "excessive_forward_lean": {
            "spoken": "Chest up! You're leaning too far forward.",
            "detailed": "You're bending too far forward at the hips, stressing your lower back.",
            "tip": "Focus on keeping your chest proud and looking slightly upward."
        },
        "knee_cave": {
            "spoken": "Push those knees out!",
            "detailed": "Your knees are collapsing inward, which can lead to knee injuries.",
            "tip": "Think about spreading the floor with your feet and pushing knees over pinky toes."
        },
        "asymmetric_depth": {
            "spoken": "Even it out! One side is going deeper.",
            "detailed": "You're squatting asymmetrically, with one side going deeper than the other.",
            "tip": "Try using a mirror or filming yourself from behind to check symmetry."
        },
        "hip_sag": {
            "spoken": "Squeeze your core! Hips are dropping.",
            "detailed": "Your hips are sagging below the shoulder-ankle line, straining your lower back.",
            "tip": "Engage your abs like bracing for a punch, and squeeze your glutes."
        },
        "hip_pike": {
            "spoken": "Lower those hips! You're piking up.",
            "detailed": "Your hips are rising too high, forming an inverted V shape.",
            "tip": "Think about making your body a straight plank from head to heels."
        },
        "elbow_flare": {
            "spoken": "Tuck your elbows in!",
            "detailed": "Your elbows are flaring out too wide, which can strain shoulders.",
            "tip": "Keep elbows at about a 45-degree angle from your body."
        },
        # Lunge errors
        "front_knee_too_far": {
            "spoken": "Keep your front knee behind your toes!",
            "detailed": "Your front knee is going past your toes, stressing the knee joint.",
            "tip": "Take a slightly longer step forward so your shin stays vertical."
        },
        "rear_knee_high": {
            "spoken": "Drop that back knee lower!",
            "detailed": "Your rear knee isn't bending enough for full range of motion.",
            "tip": "Lower until your back knee nearly touches the ground."
        },
        "torso_lean": {
            "spoken": "Stay upright! Don't lean forward.",
            "detailed": "You're leaning your torso too far forward during the lunge.",
            "tip": "Engage your core and keep your chest tall throughout the movement."
        },
        # Deadlift errors
        "rounded_back": {
            "spoken": "Straighten your back! Keep it flat.",
            "detailed": "Your back is rounding during the lift, risking a spinal injury.",
            "tip": "Think about pushing your chest forward and squeezing your shoulder blades."
        },
        "knees_too_bent": {
            "spoken": "This is a hip hinge, not a squat!",
            "detailed": "Your knees are bending too much — deadlift is driven by the hips.",
            "tip": "Push your hips back while keeping a slight knee bend."
        },
        "lockout_incomplete": {
            "spoken": "Stand all the way up! Full lockout.",
            "detailed": "You're not fully extending at the top of the lift.",
            "tip": "Squeeze your glutes at the top and stand completely straight."
        },
        # Bicep curl errors
        "swinging": {
            "spoken": "No swinging! Control the weight.",
            "detailed": "You're using momentum instead of your biceps to lift.",
            "tip": "Keep your elbows pinned to your sides and move only your forearms."
        },
        "incomplete_rom": {
            "spoken": "Full range of motion! Curl higher.",
            "detailed": "You're not curling the weight high enough for full bicep activation.",
            "tip": "Bring the weight all the way up until your forearms are vertical."
        },
        "elbow_drift": {
            "spoken": "Keep your elbows still!",
            "detailed": "Your elbows are drifting forward, reducing bicep tension.",
            "tip": "Lock your elbows at your sides throughout the entire curl."
        },
        # Shoulder press errors
        "excessive_arch": {
            "spoken": "Don't arch your back!",
            "detailed": "Your lower back is arching excessively during the press.",
            "tip": "Brace your core and squeeze your glutes to keep your back neutral."
        },
        "asymmetric_press": {
            "spoken": "Press evenly! One side is ahead.",
            "detailed": "Your arms are pressing unevenly, with one side moving faster.",
            "tip": "Use a mirror and focus on pressing both arms at the same speed."
        },
        "incomplete_lockout": {
            "spoken": "Lock out overhead! Full extension.",
            "detailed": "You're not fully extending your arms at the top of the press.",
            "tip": "Press until your arms are straight and biceps are near your ears."
        },
        # Situp errors
        "neck_pull": {
            "spoken": "Don't pull on your neck!",
            "detailed": "You're pulling your head forward instead of using your abs.",
            "tip": "Cross your arms on your chest or place fingertips behind ears lightly."
        },
        "incomplete_range": {
            "spoken": "Come up higher! Full crunch.",
            "detailed": "You're not sitting up far enough for full ab engagement.",
            "tip": "Focus on curling your shoulder blades off the ground fully."
        },
        # Plank errors (reuse hip_sag/hip_pike, add head_drop)
        "head_drop": {
            "spoken": "Look down! Keep your head neutral.",
            "detailed": "Your head is dropping too low, straining your neck.",
            "tip": "Look at a spot on the floor about a foot in front of your hands."
        },
        "not_in_position": {
            "spoken": "Step back so the camera can see you.",
            "detailed": "Position yourself so the camera can see your torso and legs.",
            "tip": "Step back 1-2 meters. Side angle works best for most exercises."
        },
    }

    ENCOURAGEMENTS = [
        "You've got this!",
        "Keep pushing!",
        "Great effort! Stay focused!",
        "You're doing amazing!",
        "Every rep makes you stronger!",
        "Excellent work! Keep it up!",
        "Focus on form, power will follow!",
        "You're crushing it!",
    ]

    def __init__(self, api_key: Optional[str] = None, persona=None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.model = settings.openai_model
        self.vision_model = "gpt-4o"
        self.client = None
        self.persona = persona  # CoachPersona or None

        if self.api_key:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"LLM feedback: {self.model} (text), {self.vision_model} (vision)")
        else:
            logger.warning("No OpenAI API key — LLM feedback disabled")

        self._encouragement_index = 0
        self._feedback_cache = {}
        self._vision_cache = {}

    def generate_quick_feedback(self, analysis: AnalysisResult) -> FeedbackResult:
        """Generate quick template-based feedback (no API call)."""
        if analysis.is_good_form or not analysis.errors:
            return FeedbackResult(
                spoken_feedback="Perfect form! Keep it up!",
                detailed_feedback="Your form looks great. Maintain this technique.",
                encouragement=self._get_encouragement(),
                tip="Focus on controlled movement and breathing.",
                is_cached=True,
            )

        priority_error = analysis.errors[0] if analysis.errors else None

        if priority_error and priority_error.error_type in self.QUICK_FEEDBACK:
            template = self.QUICK_FEEDBACK[priority_error.error_type]
            return FeedbackResult(
                spoken_feedback=template["spoken"],
                detailed_feedback=template["detailed"],
                encouragement=self._get_encouragement(),
                tip=template["tip"],
                is_cached=True,
            )

        return FeedbackResult(
            spoken_feedback="Check your form!",
            detailed_feedback=priority_error.message if priority_error else "Focus on your technique.",
            encouragement=self._get_encouragement(),
            tip="Try watching yourself in a mirror.",
            is_cached=True,
        )

    async def generate_personalized_feedback(
        self,
        analysis: AnalysisResult,
        user_context: Optional[dict] = None,
    ) -> FeedbackResult:
        """Generate personalized LLM feedback (text mode, GPT-4o-mini)."""
        if not self.client:
            return self.generate_quick_feedback(analysis)

        error_key = tuple(sorted(e.error_type for e in analysis.errors))
        cache_key = (analysis.exercise_type, error_key)

        if cache_key in self._feedback_cache:
            cached = self._feedback_cache[cache_key]
            cached.is_cached = True
            return cached

        prompt = self._build_feedback_prompt(analysis, user_context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )

            feedback_text = response.choices[0].message.content
            result = self._parse_feedback_response(feedback_text)
            self._feedback_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"LLM feedback failed: {e}")
            return self.generate_quick_feedback(analysis)

    async def analyze_frame_with_vision(
        self,
        frame: np.ndarray,
        exercise_type: str,
        current_angles: dict,
    ) -> Optional[VisionAnalysisResult]:
        """
        Analyze a video frame using GPT-4o Vision.

        This is a fundamentally different approach to form analysis:
        instead of extracting skeleton landmarks and computing angles,
        the vision model directly interprets the image to assess form.

        Args:
            frame: BGR image frame from camera
            exercise_type: "squat" or "pushup"
            current_angles: angles from pose estimation (for context)

        Returns:
            VisionAnalysisResult with independent form assessment
        """
        if not self.client:
            return None

        try:
            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            # Build vision prompt
            angle_context = ", ".join(
                f"{k}: {v:.0f} deg" for k, v in current_angles.items()
                if isinstance(v, (int, float))
            )

            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_vision_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Analyze this {exercise_type} form. "
                                    f"Pose estimation measured: {angle_context}. "
                                    f"Provide your independent visual assessment."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=400,
                temperature=0.3,
            )

            raw = response.choices[0].message.content
            return self._parse_vision_response(raw)

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return None

    def _get_system_prompt(self) -> str:
        base = """You are an expert fitness coach providing real-time form feedback.
Your feedback should be:
1. CONCISE - Users are exercising, keep spoken feedback under 10 words
2. ENCOURAGING - Always positive and motivating
3. SPECIFIC - Reference the exact body part and correction needed
4. ACTIONABLE - Tell them exactly what to do differently

Format your response as:
SPOKEN: [Short cue for voice feedback, max 10 words]
DETAILED: [One sentence explanation of the error]
TIP: [One practical tip to fix the issue]
ENCOURAGEMENT: [Short motivational message]"""

        # Inject persona personality
        if self.persona and self.persona.system_prompt_modifier:
            base += f"\n\nPERSONALITY: {self.persona.system_prompt_modifier}"
            if self.persona.feedback_style:
                base += f"\nSTYLE: {self.persona.feedback_style}"

        return base

    def _get_vision_system_prompt(self) -> str:
        return """You are an expert fitness coach analyzing exercise form from a video frame.
Assess the person's exercise form based on what you see in the image.

You must respond in this exact format:
ASSESSMENT: [good / minor_issues / major_issues]
CONFIDENCE: [high / medium / low]
OBSERVATIONS: [comma-separated list of what you observe about their form]
SUGGESTIONS: [comma-separated list of specific corrections]

Be specific about body positioning. If you cannot clearly see the person exercising, set CONFIDENCE to low."""

    def _build_feedback_prompt(
        self,
        analysis: AnalysisResult,
        user_context: Optional[dict] = None,
    ) -> str:
        errors_desc = []
        for error in analysis.errors:
            desc = f"- {error.error_type}: {error.message}"
            if error.current_value is not None:
                desc += f" (current: {error.current_value:.1f}, target: {error.target_value:.1f})"
            errors_desc.append(desc)

        prompt = f"""Exercise: {analysis.exercise_type.upper()}
Current phase: {analysis.phase.value}
Rep count: {analysis.rep_count}

Form errors detected:
{chr(10).join(errors_desc) if errors_desc else "Good form!"}

Provide coaching feedback for these errors."""

        if user_context:
            prompt += f"\n\nUser context: {user_context}"

        return prompt

    def _parse_feedback_response(self, response: str) -> FeedbackResult:
        lines = response.strip().split('\n')
        spoken = ""
        detailed = ""
        tip = ""
        encouragement = ""

        for line in lines:
            line = line.strip()
            if line.startswith("SPOKEN:"):
                spoken = line.replace("SPOKEN:", "").strip()
            elif line.startswith("DETAILED:"):
                detailed = line.replace("DETAILED:", "").strip()
            elif line.startswith("TIP:"):
                tip = line.replace("TIP:", "").strip()
            elif line.startswith("ENCOURAGEMENT:"):
                encouragement = line.replace("ENCOURAGEMENT:", "").strip()

        if not spoken:
            spoken = detailed[:50] if detailed else "Check your form!"
        if not encouragement:
            encouragement = self._get_encouragement()

        return FeedbackResult(
            spoken_feedback=spoken,
            detailed_feedback=detailed,
            encouragement=encouragement,
            tip=tip,
            is_cached=False,
        )

    def _parse_vision_response(self, response: str) -> VisionAnalysisResult:
        """Parse GPT-4o Vision response into structured result."""
        assessment = "minor_issues"
        confidence = "medium"
        observations = []
        suggestions = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith("ASSESSMENT:"):
                val = line.replace("ASSESSMENT:", "").strip().lower()
                if val in ("good", "good_form"):
                    assessment = "good"
                elif val in ("major_issues", "major"):
                    assessment = "major_issues"
                else:
                    assessment = "minor_issues"
            elif line.startswith("CONFIDENCE:"):
                val = line.replace("CONFIDENCE:", "").strip().lower()
                confidence = val if val in ("high", "medium", "low") else "medium"
            elif line.startswith("OBSERVATIONS:"):
                val = line.replace("OBSERVATIONS:", "").strip()
                observations = [o.strip() for o in val.split(",") if o.strip()]
            elif line.startswith("SUGGESTIONS:"):
                val = line.replace("SUGGESTIONS:", "").strip()
                suggestions = [s.strip() for s in val.split(",") if s.strip()]

        return VisionAnalysisResult(
            form_assessment=assessment,
            observations=observations,
            suggestions=suggestions,
            confidence=confidence,
            raw_response=response,
        )

    def _get_encouragement(self) -> str:
        # Use persona-specific encouragements if available
        pool = self.ENCOURAGEMENTS
        if self.persona and self.persona.encouragements:
            pool = self.persona.encouragements

        msg = pool[self._encouragement_index % len(pool)]
        self._encouragement_index += 1
        return msg

    def clear_cache(self):
        self._feedback_cache = {}
        self._vision_cache = {}
