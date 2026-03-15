"""
Plank Exercise Form Analyzer

Analyzes plank form for common errors:
- Hip sag (hips dropping below shoulder-ankle line)
- Hip pike (hips rising above shoulder-ankle line)
- Head drop (head drooping below shoulders)

This is an isometric exercise: no reps are counted.
Instead, hold duration is tracked.
"""
import time

from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class PlankAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for the plank hold (isometric exercise).

    Tracks body alignment from shoulder to ankle and head position.
    Does not count reps — tracks hold duration instead.
    Designed for side-view camera angle.
    """

    # Body alignment thresholds (normalized coordinate deviation)
    # Relaxed for webcam variability — plank form varies naturally
    HIP_SAG_THRESHOLD = 0.09  # Hip below shoulder-ankle line
    HIP_PIKE_THRESHOLD = 0.09  # Hip above shoulder-ankle line
    HEAD_DROP_THRESHOLD = 0.10  # Nose y much lower than shoulder y

    def __init__(self):
        super().__init__()
        self.hold_start_time: float | None = None
        self.hold_duration: float = 0.0

    @property
    def exercise_name(self) -> str:
        return "plank"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Plank is a static hold — always returns PLANK phase."""
        return ExercisePhase.PLANK

    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a plank position.

        Body should be roughly horizontal (shoulder and hip at similar y-levels).
        Need shoulders, hips, and at least one wrist visible.
        """
        landmarks = pose_result.landmarks

        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for name in required:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.4:
                return False

        # Need at least one wrist visible (hands/forearms on ground)
        has_wrist = (
            ("left_wrist" in landmarks and landmarks["left_wrist"].visibility > 0.3)
            or ("right_wrist" in landmarks and landmarks["right_wrist"].visibility > 0.3)
        )
        if not has_wrist:
            return False

        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2

        # In plank, body is roughly horizontal (not upright)
        if abs(mid_hip_y - mid_shoulder_y) > 0.3:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze plank form and track hold duration."""
        errors = []

        if not pose_result.is_valid:
            # Lost tracking — pause hold timer
            self.hold_start_time = None
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="no_pose_detected",
                    message="Cannot detect your body. Try a side angle view.",
                    severity=ErrorSeverity.INFO,
                    body_part="full_body"
                )],
                rep_count=self.rep_counter.count,
                is_good_form=False,
                angles={},
                feedback_priority=["no_pose_detected"]
            )

        angles = pose_result.angles
        landmarks = pose_result.landmarks

        # Check if the pose looks like a plank position
        if not self._is_exercise_pose(pose_result):
            self.hold_start_time = None
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Get into plank position. Side view works best.",
                    severity=ErrorSeverity.INFO,
                    body_part="full_body"
                )],
                rep_count=self.rep_counter.count,
                is_good_form=False,
                angles=angles,
                feedback_priority=["not_in_position"]
            )

        # Set phase to PLANK (no rep counting for isometric holds)
        new_phase = self.detect_phase(pose_result)
        # Don't call rep_counter.update_phase to avoid counting reps
        self.rep_counter.phase = new_phase

        # Track hold duration
        now = time.time()
        if self.hold_start_time is None:
            self.hold_start_time = now
        self.hold_duration = now - self.hold_start_time

        # Check body alignment (hip sag/pike)
        alignment_error = self._check_body_alignment(landmarks)
        if alignment_error:
            errors.append(alignment_error)

        # Check head position
        head_error = self._check_head_position(landmarks)
        if head_error:
            errors.append(head_error)

        # Track error history
        self.error_history.append(errors)

        # Determine if form is good
        is_good_form = all(e.severity == ErrorSeverity.INFO for e in errors)

        return AnalysisResult(
            exercise_type=self.exercise_name,
            phase=new_phase,
            errors=errors,
            rep_count=self.rep_counter.count,
            is_good_form=is_good_form or len(errors) == 0,
            angles=angles,
            feedback_priority=self._prioritize_errors(errors)
        )

    def reset(self):
        """Reset analyzer state including hold timer."""
        super().reset()
        self.hold_start_time = None
        self.hold_duration = 0.0

    def _check_body_alignment(self, landmarks: dict) -> FormError | None:
        """Check for hip sag or pike relative to the shoulder-ankle line."""
        required = ["left_shoulder", "right_shoulder", "left_hip",
                     "right_hip", "left_ankle", "right_ankle"]
        if not all(k in landmarks for k in required):
            return None

        # Calculate mid-points
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2
        mid_ankle_y = (landmarks["left_ankle"].y + landmarks["right_ankle"].y) / 2

        mid_shoulder_x = (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2
        mid_hip_x = (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2
        mid_ankle_x = (landmarks["left_ankle"].x + landmarks["right_ankle"].x) / 2

        # Linear interpolation for expected hip position on shoulder-ankle line
        if abs(mid_ankle_x - mid_shoulder_x) < 0.01:
            return None  # Avoid division by zero

        t = (mid_hip_x - mid_shoulder_x) / (mid_ankle_x - mid_shoulder_x)
        expected_hip_y = mid_shoulder_y + t * (mid_ankle_y - mid_shoulder_y)

        # Deviation from straight line (positive = sag, negative = pike)
        deviation = mid_hip_y - expected_hip_y

        if deviation > self.HIP_SAG_THRESHOLD:
            return FormError(
                error_type="hip_sag",
                message="Squeeze your core! Your hips are sagging down.",
                severity=ErrorSeverity.ERROR,
                body_part="hips",
                current_value=deviation,
                target_value=0,
                confidence=0.85
            )
        elif deviation < -self.HIP_PIKE_THRESHOLD:
            return FormError(
                error_type="hip_pike",
                message="Lower your hips! They're piking up too high.",
                severity=ErrorSeverity.WARNING,
                body_part="hips",
                current_value=deviation,
                target_value=0,
                confidence=0.85
            )

        return None

    def _check_head_position(self, landmarks: dict) -> FormError | None:
        """Check for excessive head drop (nose well below shoulder line)."""
        if "nose" not in landmarks:
            return None

        required = ["left_shoulder", "right_shoulder"]
        if not all(k in landmarks for k in required):
            return None

        if landmarks["nose"].visibility < 0.3:
            return None

        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        nose_y = landmarks["nose"].y

        # In image coordinates, y increases downward
        # Head drop: nose y is significantly greater than shoulder y
        if nose_y - mid_shoulder_y > self.HEAD_DROP_THRESHOLD:
            return FormError(
                error_type="head_drop",
                message="Keep your head in line with your spine! Don't let it drop.",
                severity=ErrorSeverity.WARNING,
                body_part="head",
                current_value=nose_y - mid_shoulder_y,
                target_value=0,
                confidence=0.8
            )

        return None
