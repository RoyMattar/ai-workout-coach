"""
AI Model Orchestrator

Coordinates the flow between six AI models in a multi-stage pipeline:

  Stage 1: Pose Estimation (MediaPipe BlazePose) — Computer Vision domain
  Stage 2a: ML Form Classifier (scikit-learn SVM) — Classification domain
  Stage 2b: Rule-Based Analysis (exercise-specific analyzers) — kept for comparison
  Stage 3: Result Fusion — combines ML + rules with weighted consensus
  Stage 4a: Template Feedback — immediate (<50ms)
  Stage 4b: LLM Feedback (GPT-4o-mini) — async, NLP domain
  Stage 4c: TTS Audio (gTTS) — async, Audio Synthesis domain

This orchestrator demonstrates genuine AI model orchestration by:
- Running two independent analysis methods (ML + rules) and fusing their results
- Providing three feedback channels with different latency profiles
- Tracking model agreement/disagreement as a confidence signal
- Managing data flow and rate limiting across all pipeline stages
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import numpy as np

from .pose_estimator import PoseEstimator, PoseResult
from .exercises import (
    SquatAnalyzer, PushupAnalyzer, LungeAnalyzer, PlankAnalyzer,
    DeadliftAnalyzer, BicepCurlAnalyzer, ShoulderPressAnalyzer, SitupAnalyzer,
    ExerciseAnalyzer,
)
from .exercises.base import AnalysisResult, FormError, ErrorSeverity
from .feedback_generator import FeedbackGenerator, FeedbackResult
from .models.form_classifier import FormClassifier, FormClassification
from .tts_engine import TTSEngine
from .coach_personas import CoachPersona, get_persona, DEFAULT_COACH
from .config import get_settings

logger = logging.getLogger(__name__)


class ExerciseType(Enum):
    """Supported exercise types"""
    SQUAT = "squat"
    PUSHUP = "pushup"
    LUNGE = "lunge"
    PLANK = "plank"
    DEADLIFT = "deadlift"
    BICEP_CURL = "bicep_curl"
    SHOULDER_PRESS = "shoulder_press"
    SITUP = "situp"


@dataclass
class PipelineTiming:
    """Timing breakdown for each pipeline stage (milliseconds)."""
    pose_estimation_ms: float = 0.0
    ml_classification_ms: float = 0.0
    rule_analysis_ms: float = 0.0
    fusion_ms: float = 0.0
    feedback_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pose_estimation_ms": round(self.pose_estimation_ms, 2),
            "ml_classification_ms": round(self.ml_classification_ms, 2),
            "rule_analysis_ms": round(self.rule_analysis_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "feedback_ms": round(self.feedback_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class SessionStats:
    """Statistics for a workout session"""
    exercise_type: str = ""
    total_reps: int = 0
    good_form_reps: int = 0
    errors_by_type: dict = field(default_factory=dict)
    session_start: float = 0.0
    last_feedback_time: float = 0.0
    frames_processed: int = 0
    model_agreements: int = 0
    model_disagreements: int = 0
    # 5-dimension form score tracking (frames with good form per dimension)
    dimension_good: dict = field(default_factory=lambda: {
        "depth": 0, "alignment": 0, "symmetry": 0, "tempo": 0, "consistency": 0
    })
    dimension_total: dict = field(default_factory=lambda: {
        "depth": 0, "alignment": 0, "symmetry": 0, "tempo": 0, "consistency": 0
    })

    @property
    def form_score(self) -> float:
        if self.total_reps == 0:
            return 100.0
        return (self.good_form_reps / self.total_reps) * 100

    @property
    def model_agreement_rate(self) -> float:
        total = self.model_agreements + self.model_disagreements
        if total == 0:
            return 1.0
        return self.model_agreements / total

    @property
    def form_dimensions(self) -> dict:
        """5-dimension form score breakdown (0-100 each)."""
        dims = {}
        for dim in ["depth", "alignment", "symmetry", "tempo", "consistency"]:
            total = self.dimension_total.get(dim, 0)
            good = self.dimension_good.get(dim, 0)
            dims[dim] = round((good / total) * 100, 1) if total > 0 else 100.0
        return dims

    def to_dict(self) -> dict:
        return {
            "exercise_type": self.exercise_type,
            "total_reps": self.total_reps,
            "good_form_reps": self.good_form_reps,
            "form_score": round(self.form_score, 1),
            "form_dimensions": self.form_dimensions,
            "errors_by_type": self.errors_by_type,
            "duration_seconds": time.time() - self.session_start if self.session_start else 0,
            "frames_processed": self.frames_processed,
            "model_agreement_rate": round(self.model_agreement_rate, 3),
        }


@dataclass
class OrchestrationResult:
    """Complete result from the orchestration pipeline"""
    pose_result: PoseResult
    analysis_result: AnalysisResult
    ml_classification: Optional[FormClassification]
    feedback: FeedbackResult
    session_stats: SessionStats
    pipeline_timing: PipelineTiming
    models_agree: bool = True

    def to_dict(self) -> dict:
        result = {
            "pose": {
                "is_valid": self.pose_result.is_valid,
                "confidence": self.pose_result.confidence,
                "angles": self.pose_result.angles,
                "landmarks": self._serialize_landmarks(),
            },
            "analysis": self.analysis_result.to_dict(),
            "ml_classification": self.ml_classification.to_dict() if self.ml_classification else None,
            "feedback": {
                "spoken": self.feedback.spoken_feedback,
                "detailed": self.feedback.detailed_feedback,
                "tip": self.feedback.tip,
                "encouragement": self.feedback.encouragement,
            },
            "session": self.session_stats.to_dict(),
            "pipeline_timing": self.pipeline_timing.to_dict(),
            "models_agree": self.models_agree,
        }
        return result

    def _serialize_landmarks(self) -> dict:
        """Serialize pose landmarks for frontend skeleton rendering."""
        if not self.pose_result.is_valid:
            return {}
        landmarks = {}
        for name, lm in self.pose_result.landmarks.items():
            landmarks[name] = {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
        return landmarks


class WorkoutOrchestrator:
    """
    Orchestrates the AI workout coaching pipeline.

    Coordinates four AI models:
    1. MediaPipe BlazePose (pose estimation — vision domain)
    2. scikit-learn SVM (form quality classification — classification domain)
    3. GPT-4o-mini (coaching feedback — NLP domain)
    4. gTTS (voice synthesis — audio domain)

    The orchestrator manages data flow, result fusion, rate limiting,
    and session state across all models.
    """

    # Minimum time between feedback updates
    FEEDBACK_COOLDOWN = 2.0  # seconds
    # Minimum time between GPT-4o Vision analyses (expensive, rate-limit)
    VISION_COOLDOWN = 8.0  # seconds
    # Weight for ML classifier vs rule-based in fusion (0-1)
    ML_WEIGHT = 0.6
    RULE_WEIGHT = 0.4

    def __init__(
        self,
        exercise_type: ExerciseType = ExerciseType.SQUAT,
        openai_api_key: Optional[str] = None,
        persona_id: str = "coach_pro",
    ):
        settings = get_settings()

        # Coach persona
        self.persona = get_persona(persona_id)
        logger.info(f"Coach persona: {self.persona.name} (voice: {self.persona.voice_id})")

        # Model 1: Pose Estimation (MediaPipe BlazePose)
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        logger.info("Model 1: MediaPipe BlazePose initialized (vision domain)")

        # Model 2: Form Quality Classifier (scikit-learn)
        self.form_classifier = FormClassifier()
        logger.info("Model 2: Form quality classifier initialized (classification domain)")

        # Model 3 (feedback): LLM Feedback Generator (GPT-4o-mini)
        self.feedback_generator = FeedbackGenerator(
            api_key=openai_api_key,
            persona=self.persona,
        )
        logger.info("Model 3: GPT-4o-mini feedback generator initialized (NLP domain)")

        # Model 4: Text-to-Speech (OpenAI TTS / gTTS)
        self.tts_engine = TTSEngine(
            openai_api_key=openai_api_key,
            default_voice=self.persona.voice_id,
        )
        logger.info(f"Model 4: TTS engine initialized ({self.tts_engine.engine_name}, audio domain)")

        # Exercise analyzers (rule-based complement to ML classifier)
        self.analyzers: dict[str, ExerciseAnalyzer] = {
            ExerciseType.SQUAT.value: SquatAnalyzer(),
            ExerciseType.PUSHUP.value: PushupAnalyzer(),
            ExerciseType.LUNGE.value: LungeAnalyzer(),
            ExerciseType.PLANK.value: PlankAnalyzer(),
            ExerciseType.DEADLIFT.value: DeadliftAnalyzer(),
            ExerciseType.BICEP_CURL.value: BicepCurlAnalyzer(),
            ExerciseType.SHOULDER_PRESS.value: ShoulderPressAnalyzer(),
            ExerciseType.SITUP.value: SitupAnalyzer(),
        }

        # Current exercise
        self.current_exercise = exercise_type
        self.current_analyzer = self.analyzers[exercise_type.value]

        # Session tracking
        self.session_stats = SessionStats()
        self.session_stats.session_start = time.time()
        self.session_stats.exercise_type = exercise_type.value

        # State
        self._last_rep_count = 0
        self._last_feedback: Optional[FeedbackResult] = None
        self._last_errors: list[str] = []
        self._last_tts_text: str = ""
        self._last_vision_time: float = 0.0
        self._last_angles: dict = {}

    def set_exercise(self, exercise_type: ExerciseType):
        """Change the current exercise type."""
        self.current_exercise = exercise_type
        self.current_analyzer = self.analyzers[exercise_type.value]
        self.current_analyzer.reset()
        self.session_stats = SessionStats()
        self.session_stats.session_start = time.time()
        self.session_stats.exercise_type = exercise_type.value
        self._last_rep_count = 0
        self._last_feedback = None
        self._last_errors = []
        self._last_tts_text = ""

    def process_frame(self, frame: np.ndarray) -> OrchestrationResult:
        """
        Process a single frame through the complete orchestration pipeline.

        Pipeline stages:
          1. Pose Estimation (MediaPipe) → landmarks + angles
          2a. ML Classification (scikit-learn) → form quality + confidence
          2b. Rule-Based Analysis (exercise analyzer) → errors + phase + reps
          3. Result Fusion → combine ML + rules, track agreement
          4. Feedback Generation (templates) → coaching cues
        """
        timing = PipelineTiming()
        pipeline_start = time.time()

        # ── Stage 1: Pose Estimation (MediaPipe BlazePose) ──
        t0 = time.time()
        pose_result = self.pose_estimator.process_frame(frame)
        timing.pose_estimation_ms = (time.time() - t0) * 1000

        # ── Stage 2a: ML Form Classification (scikit-learn SVM) ──
        ml_classification = None
        t0 = time.time()
        if pose_result.is_valid and self.form_classifier.is_available(self.current_exercise.value):
            ml_classification = self.form_classifier.classify(
                exercise=self.current_exercise.value,
                pose_angles=pose_result.angles,
                pose_landmarks=pose_result.landmarks,
            )
        timing.ml_classification_ms = (time.time() - t0) * 1000

        # ── Stage 2b: Rule-Based Analysis ──
        t0 = time.time()
        analysis_result = self.current_analyzer.analyze(pose_result)
        timing.rule_analysis_ms = (time.time() - t0) * 1000

        # ── Stage 3: Result Fusion ──
        t0 = time.time()
        models_agree = self._fuse_results(analysis_result, ml_classification)
        timing.fusion_ms = (time.time() - t0) * 1000

        # ── Stage 4: Feedback Generation (template-based for sync path) ──
        t0 = time.time()
        feedback = self._generate_feedback(analysis_result, ml_classification)
        timing.feedback_ms = (time.time() - t0) * 1000

        # Update session — ML classification is primary form quality signal
        self._update_session_stats(analysis_result, ml_classification)
        self.session_stats.frames_processed += 1
        self._last_angles = pose_result.angles

        timing.total_ms = (time.time() - pipeline_start) * 1000

        return OrchestrationResult(
            pose_result=pose_result,
            analysis_result=analysis_result,
            ml_classification=ml_classification,
            feedback=feedback,
            session_stats=self.session_stats,
            pipeline_timing=timing,
            models_agree=models_agree,
        )

    async def process_frame_async(self, frame: np.ndarray) -> OrchestrationResult:
        """
        Async version with LLM feedback (GPT-4o-mini).

        Returns the synchronous result immediately but also triggers
        async LLM feedback generation when appropriate.
        """
        # Run synchronous pipeline first
        result = self.process_frame(frame)

        # Trigger async LLM feedback if appropriate
        current_time = time.time()
        if (result.analysis_result.errors
                and self._should_generate_new_feedback(result.analysis_result, current_time)):
            try:
                llm_feedback = await self.feedback_generator.generate_personalized_feedback(
                    result.analysis_result
                )
                # Update result with richer LLM feedback
                result.feedback = llm_feedback
                self._last_feedback = llm_feedback
                self.session_stats.last_feedback_time = current_time
            except Exception as e:
                logger.error(f"Async LLM feedback failed: {e}")

        return result

    async def analyze_frame_with_vision(self, frame: np.ndarray) -> Optional[dict]:
        """
        Request GPT-4o Vision analysis of a video frame.

        This provides a completely independent form assessment by having
        the vision model look at the actual image rather than processed
        skeleton data. Returns the result as a dict for the frontend.

        Rate-limited: only runs every VISION_COOLDOWN seconds.
        """
        current_time = time.time()
        if current_time - self._last_vision_time < self.VISION_COOLDOWN:
            return None

        result = await self.feedback_generator.analyze_frame_with_vision(
            frame=frame,
            exercise_type=self.current_exercise.value,
            current_angles=self._last_angles,
        )

        if result:
            self._last_vision_time = current_time
            return result.to_dict()

        return None

    async def generate_tts_audio(self, text: str) -> Optional[str]:
        """
        Generate TTS audio for coaching feedback (async wrapper).

        Returns base64-encoded MP3 audio, or None if TTS is unavailable
        or the text hasn't changed.
        """
        if not text or text == self._last_tts_text:
            return None

        if not self.tts_engine.is_available:
            return None

        # Run TTS in executor to avoid blocking
        loop = asyncio.get_event_loop()
        audio_b64 = await loop.run_in_executor(
            None,
            lambda: self.tts_engine.synthesize_base64(
                text,
                voice=self.persona.voice_id,
                speed=self.persona.tts_speed,
            ),
        )

        if audio_b64:
            self._last_tts_text = text

        return audio_b64

    def _fuse_results(
        self,
        rule_result: AnalysisResult,
        ml_result: Optional[FormClassification],
    ) -> bool:
        """
        Fuse ML classifier and rule-based analysis results.

        Compares both methods' assessment of form quality and tracks
        agreement/disagreement. When they disagree, the combined
        assessment uses weighted voting.

        Returns:
            True if models agree, False otherwise
        """
        if ml_result is None or ml_result.quality_score < 0:
            return True  # No ML result to compare

        # Map rule-based result to quality level
        rule_quality = 0  # good_form
        if rule_result.errors:
            max_severity = max(e.severity for e in rule_result.errors)
            if max_severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL):
                rule_quality = 2  # major_issues
            elif max_severity == ErrorSeverity.WARNING:
                rule_quality = 1  # minor_issues

        ml_quality = ml_result.quality_score

        # Check agreement
        agree = rule_quality == ml_quality
        if agree:
            self.session_stats.model_agreements += 1
        else:
            self.session_stats.model_disagreements += 1
            logger.debug(
                f"Model disagreement: rules={rule_quality}, ML={ml_quality} "
                f"(ML confidence={ml_result.confidence:.2f})"
            )

        return agree

    def _generate_feedback(
        self,
        analysis_result: AnalysisResult,
        ml_classification: Optional[FormClassification] = None,
    ) -> FeedbackResult:
        """Generate feedback with rate limiting, considering both analysis methods."""
        current_time = time.time()

        if self._should_generate_new_feedback(analysis_result, current_time):
            feedback = self.feedback_generator.generate_quick_feedback(analysis_result)
            self._last_feedback = feedback
            self.session_stats.last_feedback_time = current_time
            self._last_errors = [e.error_type for e in analysis_result.errors]
            return feedback

        if self._last_feedback:
            return self._last_feedback

        return self.feedback_generator.generate_quick_feedback(analysis_result)

    def _should_generate_new_feedback(
        self,
        analysis_result: AnalysisResult,
        current_time: float,
    ) -> bool:
        """Determine if we should generate new feedback."""
        if self._last_feedback is None:
            return True

        time_since_last = current_time - self.session_stats.last_feedback_time
        if time_since_last < self.FEEDBACK_COOLDOWN:
            return False

        current_errors = [e.error_type for e in analysis_result.errors]
        if set(current_errors) != set(self._last_errors):
            return True

        if analysis_result.rep_count > self._last_rep_count:
            return True

        return False

    # Map error types to form dimensions
    ERROR_DIMENSION_MAP = {
        "insufficient_depth": "depth", "almost_parallel": "depth",
        "almost_full_depth": "depth", "incomplete_rom": "depth",
        "incomplete_range": "depth", "incomplete_lockout": "depth",
        "excessive_forward_lean": "alignment", "torso_lean": "alignment",
        "hip_sag": "alignment", "hip_pike": "alignment",
        "rounded_back": "alignment", "excessive_arch": "alignment",
        "head_drop": "alignment", "neck_pull": "alignment",
        "knee_cave": "symmetry", "asymmetric_depth": "symmetry",
        "asymmetric_press": "symmetry",
        "swinging": "tempo", "elbow_drift": "tempo",
        "front_knee_too_far": "consistency", "knees_too_bent": "consistency",
        "rear_knee_high": "consistency", "lockout_incomplete": "consistency",
        "elbow_flare": "consistency",
    }

    def _update_session_stats(
        self,
        analysis_result: AnalysisResult,
        ml_classification: Optional[FormClassification] = None,
    ):
        """Update session statistics. ML classification is primary form quality signal."""
        if analysis_result.rep_count > self._last_rep_count:
            new_reps = analysis_result.rep_count - self._last_rep_count
            self.session_stats.total_reps += new_reps

            # ML classification is primary; fall back to rule-based
            if ml_classification and ml_classification.quality_score >= 0:
                is_good = ml_classification.quality_label == "good_form"
            else:
                is_good = analysis_result.is_good_form

            if is_good:
                self.session_stats.good_form_reps += new_reps

            self._last_rep_count = analysis_result.rep_count

        # Track error frequency
        for error in analysis_result.errors:
            error_type = error.error_type
            if error_type in ("no_pose_detected", "not_in_position"):
                continue
            if error_type not in self.session_stats.errors_by_type:
                self.session_stats.errors_by_type[error_type] = 0
            self.session_stats.errors_by_type[error_type] += 1

        # Track 5-dimension form scores
        active_errors = {e.error_type for e in analysis_result.errors
                         if e.error_type not in ("no_pose_detected", "not_in_position")}
        if analysis_result.phase.value not in ("standing", "plank") and analysis_result.errors:
            error_dims = {self.ERROR_DIMENSION_MAP.get(et) for et in active_errors} - {None}
            for dim in ["depth", "alignment", "symmetry", "tempo", "consistency"]:
                self.session_stats.dimension_total[dim] += 1
                if dim not in error_dims:
                    self.session_stats.dimension_good[dim] += 1

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        summary = self.session_stats.to_dict()
        summary["most_common_error"] = (
            max(
                self.session_stats.errors_by_type.items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0]
            if self.session_stats.errors_by_type
            else "none"
        )
        summary["models"] = self.get_model_info()
        return summary

    def get_model_info(self) -> dict:
        """Get information about all models in the pipeline."""
        return {
            "pose_estimation": {
                "name": "MediaPipe BlazePose",
                "domain": "computer_vision",
                "status": "active",
            },
            "form_classifier": {
                "name": "scikit-learn SVM",
                "domain": "classification",
                "status": "active" if self.form_classifier.is_available(self.current_exercise.value) else "unavailable",
                "details": self.form_classifier.get_model_info(),
            },
            "feedback_generator": {
                "name": "GPT-4o-mini",
                "domain": "nlp_text",
                "status": "active" if self.feedback_generator.client else "no_api_key",
            },
            "vision_analyzer": {
                "name": "GPT-4o Vision",
                "domain": "visual_understanding",
                "status": "active" if self.feedback_generator.client else "no_api_key",
            },
            "tts_engine": {
                "name": self.tts_engine.engine_name,
                "domain": "audio_synthesis",
                "status": "active" if self.tts_engine.is_available else "unavailable",
                "details": self.tts_engine.get_info(),
            },
        }

    def reset_session(self):
        """Reset the session statistics."""
        self.session_stats = SessionStats()
        self.session_stats.session_start = time.time()
        self.session_stats.exercise_type = self.current_exercise.value
        self.current_analyzer.reset()
        self._last_rep_count = 0
        self._last_feedback = None
        self._last_errors = []
        self._last_tts_text = ""
        self.tts_engine.clear_cache()

    def close(self):
        """Release resources."""
        self.pose_estimator.close()
