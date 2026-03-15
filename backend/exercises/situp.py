"""
Sit-up Exercise Form Analyzer

Analyzes sit-up form for common errors:
- Neck pull (pulling on the neck instead of using abs)
- Incomplete range of motion (not sitting up fully)

Tracks torso angle relative to horizontal via the hip angle.
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class SitupAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for sit-ups and crunches.

    Tracks hip angle to measure how far the torso curls up from lying.
    Designed for side-view camera angle.

    Phase mapping (reusing push-up phase enums):
    - PLANK: lying flat on back
    - PUSHING: curling up (ascending)
    - LOWEST: top of sit-up (torso most upright)
    - LOWERING: lowering back down
    """

    # Form thresholds (in degrees)
    INCOMPLETE_RANGE_THRESHOLD = 100  # Hip angle at top (should be < 100)
    NECK_PULL_DISTANCE_THRESHOLD = 0.04  # Nose-to-shoulder distance change

    # Phase detection thresholds (based on hip angle)
    # Hip angle: 180 = lying flat, decreases as you sit up
    LYING_THRESHOLD = 150  # Roughly lying flat
    CURLING_THRESHOLD = 120  # Mid-curl
    TOP_THRESHOLD = 100  # At or near top of sit-up

    def __init__(self):
        super().__init__()
        self._prev_nose_shoulder_dist: float | None = None

    @property
    def exercise_name(self) -> str:
        return "situp"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect sit-up phase based on hip angle."""
        if not pose_result.is_valid:
            return self.rep_counter.phase

        angles = pose_result.angles

        # Use average hip angle
        hip_angles = []
        if "left_hip_angle" in angles:
            hip_angles.append(angles["left_hip_angle"])
        if "right_hip_angle" in angles:
            hip_angles.append(angles["right_hip_angle"])

        if not hip_angles:
            return self.rep_counter.phase

        avg_hip_angle = sum(hip_angles) / len(hip_angles)
        current_phase = self.rep_counter.phase

        # State machine for phase detection
        # Hip angle decreases as the person sits up
        if avg_hip_angle > self.LYING_THRESHOLD:
            return ExercisePhase.PLANK  # Lying flat
        elif avg_hip_angle > self.CURLING_THRESHOLD:
            if current_phase in [ExercisePhase.PLANK, ExercisePhase.PUSHING]:
                return ExercisePhase.PUSHING  # Curling up
            else:
                return ExercisePhase.LOWERING  # Lowering down
        elif avg_hip_angle > self.TOP_THRESHOLD:
            if current_phase in [ExercisePhase.PLANK, ExercisePhase.PUSHING]:
                return ExercisePhase.PUSHING
            elif current_phase == ExercisePhase.LOWEST:
                return ExercisePhase.LOWERING
            else:
                return ExercisePhase.LOWERING
        else:
            return ExercisePhase.LOWEST  # Top of sit-up

    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a sit-up position.

        Need shoulders and hips visible. Shoulders and hips should be at
        a similar y-level (lying down or crunching), not fully upright.
        """
        landmarks = pose_result.landmarks

        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for name in required:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.4:
                return False

        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2

        # For sit-ups the body is roughly horizontal or at a moderate angle
        # Shoulders should not be far above hips (that would be standing)
        # Allow some difference since during the curl shoulders rise
        if mid_hip_y - mid_shoulder_y > 0.35:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze sit-up form and detect errors."""
        errors = []

        if not pose_result.is_valid:
            self._prev_nose_shoulder_dist = None
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

        # Check if the pose looks like a sit-up position
        if not self._is_exercise_pose(pose_result):
            self._prev_nose_shoulder_dist = None
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Lie on your back with your body visible. Side view works best.",
                    severity=ErrorSeverity.INFO,
                    body_part="full_body"
                )],
                rep_count=self.rep_counter.count,
                is_good_form=False,
                angles=angles,
                feedback_priority=["not_in_position"]
            )

        # Detect current phase
        new_phase = self.detect_phase(pose_result)
        self.rep_counter.update_phase(new_phase)

        # Check form during active phases
        if new_phase not in [ExercisePhase.PLANK]:
            # Check for neck pulling
            neck_error = self._check_neck_pull(landmarks)
            if neck_error:
                errors.append(neck_error)

            # Check range of motion at top
            if new_phase == ExercisePhase.LOWEST:
                rom_error = self._check_incomplete_range(angles)
                if rom_error:
                    errors.append(rom_error)

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
        """Reset analyzer state including tracking variables."""
        super().reset()
        self._prev_nose_shoulder_dist = None

    def _check_neck_pull(self, landmarks: dict) -> FormError | None:
        """
        Check for neck pulling (nose moving forward much faster than shoulders).

        Approximated by tracking the nose-to-shoulder distance change between
        frames. A rapid decrease suggests the head is being pulled forward.
        """
        if "nose" not in landmarks:
            self._prev_nose_shoulder_dist = None
            return None

        required = ["left_shoulder", "right_shoulder"]
        if not all(k in landmarks for k in required):
            self._prev_nose_shoulder_dist = None
            return None

        if landmarks["nose"].visibility < 0.3:
            self._prev_nose_shoulder_dist = None
            return None

        mid_shoulder_x = (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        nose_x = landmarks["nose"].x
        nose_y = landmarks["nose"].y

        # Distance from nose to mid-shoulder
        current_dist = ((nose_x - mid_shoulder_x) ** 2 + (nose_y - mid_shoulder_y) ** 2) ** 0.5

        result = None
        if self._prev_nose_shoulder_dist is not None:
            dist_change = self._prev_nose_shoulder_dist - current_dist
            # Rapid decrease in distance = head pulling toward chest
            if dist_change > self.NECK_PULL_DISTANCE_THRESHOLD:
                result = FormError(
                    error_type="neck_pull",
                    message="Don't pull on your neck! Use your abs to lift your torso.",
                    severity=ErrorSeverity.ERROR,
                    body_part="head",
                    current_value=dist_change,
                    target_value=0,
                    confidence=0.75
                )

        self._prev_nose_shoulder_dist = current_dist
        return result

    def _check_incomplete_range(self, angles: dict) -> FormError | None:
        """Check for incomplete range of motion at the top of the sit-up."""
        hip_angles = []
        if "left_hip_angle" in angles:
            hip_angles.append(angles["left_hip_angle"])
        if "right_hip_angle" in angles:
            hip_angles.append(angles["right_hip_angle"])

        if not hip_angles:
            return None

        avg_hip_angle = sum(hip_angles) / len(hip_angles)

        if avg_hip_angle > self.INCOMPLETE_RANGE_THRESHOLD:
            return FormError(
                error_type="incomplete_range",
                message="Sit up higher! Bring your torso closer to your knees.",
                severity=ErrorSeverity.WARNING,
                body_part="torso",
                current_value=avg_hip_angle,
                target_value=80.0,
                confidence=0.85
            )

        return None
