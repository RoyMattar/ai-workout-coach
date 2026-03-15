"""
Shoulder Press Exercise Form Analyzer

Analyzes shoulder press form for common errors:
- Excessive back arch (leaning backward during press)
- Asymmetric press (one arm pressing higher than the other)
- Incomplete lockout (not fully extending arms overhead)
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class ShoulderPressAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for standing or seated overhead shoulder press.

    Tracks elbow angle (pressing motion), shoulder position, and back arch.
    Upper-body only exercise, so leg landmarks are not required.

    Phase mapping (reusing available enums):
    - STANDING: elbows bent at sides (~90 degrees)
    - ASCENDING: pressing up, elbow angle increasing
    - BOTTOM: arms overhead (elbow > 160)
    - DESCENDING: lowering back down
    """

    # Form thresholds (in degrees, relaxed for webcam variability)
    EXCESSIVE_ARCH_THRESHOLD = 25  # Max torso angle (back arch)
    ASYMMETRIC_THRESHOLD = 30  # Max left-right elbow angle difference
    LOCKOUT_THRESHOLD = 150  # Elbow angle for full lockout overhead

    # Phase detection thresholds (based on elbow angle)
    STANDING_THRESHOLD = 100  # Elbows bent at sides (start position)
    ASCENDING_THRESHOLD = 130  # Mid-press
    TOP_THRESHOLD = 160  # Arms overhead (fully pressed)

    @property
    def exercise_name(self) -> str:
        return "shoulder_press"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect shoulder press phase based on elbow angle."""
        if not pose_result.is_valid:
            return self.rep_counter.phase

        angles = pose_result.angles

        # Use average elbow angle
        elbow_angles = []
        if "left_elbow_angle" in angles:
            elbow_angles.append(angles["left_elbow_angle"])
        if "right_elbow_angle" in angles:
            elbow_angles.append(angles["right_elbow_angle"])

        if not elbow_angles:
            return self.rep_counter.phase

        avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)
        current_phase = self.rep_counter.phase

        # State machine for phase detection
        # For shoulder press, elbow angle INCREASES as you press up
        if avg_elbow_angle < self.STANDING_THRESHOLD:
            return ExercisePhase.STANDING
        elif avg_elbow_angle < self.ASCENDING_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.ASCENDING]:
                return ExercisePhase.ASCENDING
            else:
                return ExercisePhase.DESCENDING
        elif avg_elbow_angle < self.TOP_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.ASCENDING]:
                return ExercisePhase.ASCENDING
            elif current_phase == ExercisePhase.BOTTOM:
                return ExercisePhase.DESCENDING
            else:
                return ExercisePhase.DESCENDING
        else:
            return ExercisePhase.BOTTOM

    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a shoulder press position.

        Only upper body is needed: shoulders, elbows, and wrists.
        Legs are not required.
        """
        landmarks = pose_result.landmarks

        required = ["left_shoulder", "right_shoulder"]
        for name in required:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.4:
                return False

        # Need at least one elbow and one wrist visible
        has_elbow = (
            ("left_elbow" in landmarks and landmarks["left_elbow"].visibility > 0.3)
            or ("right_elbow" in landmarks and landmarks["right_elbow"].visibility > 0.3)
        )
        has_wrist = (
            ("left_wrist" in landmarks and landmarks["left_wrist"].visibility > 0.3)
            or ("right_wrist" in landmarks and landmarks["right_wrist"].visibility > 0.3)
        )

        if not has_elbow or not has_wrist:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze shoulder press form and detect errors."""
        errors = []

        if not pose_result.is_valid:
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="no_pose_detected",
                    message="Cannot detect your body. Please step into frame.",
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

        # Check if the pose looks like a shoulder press position
        if not self._is_exercise_pose(pose_result):
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Make sure your upper body is visible in the camera.",
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
        if new_phase != ExercisePhase.STANDING:
            # Check for excessive back arch
            arch_error = self._check_excessive_arch(angles)
            if arch_error:
                errors.append(arch_error)

            # Check for asymmetric press
            asym_error = self._check_asymmetric_press(angles)
            if asym_error:
                errors.append(asym_error)

            # Check lockout at the top
            if new_phase == ExercisePhase.BOTTOM:
                lockout_error = self._check_incomplete_lockout(angles)
                if lockout_error:
                    errors.append(lockout_error)

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

    def _check_excessive_arch(self, angles: dict) -> FormError | None:
        """Check for excessive back arch during the press."""
        if "torso_angle" not in angles:
            return None

        torso_angle = angles["torso_angle"]

        if torso_angle > self.EXCESSIVE_ARCH_THRESHOLD:
            return FormError(
                error_type="excessive_arch",
                message="Brace your core! You're arching your back too much.",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=torso_angle,
                target_value=self.EXCESSIVE_ARCH_THRESHOLD,
                confidence=0.85
            )

        return None

    def _check_asymmetric_press(self, angles: dict) -> FormError | None:
        """Check for asymmetric pressing (one arm higher than the other)."""
        if "left_elbow_angle" not in angles or "right_elbow_angle" not in angles:
            return None

        diff = abs(angles["left_elbow_angle"] - angles["right_elbow_angle"])

        if diff > self.ASYMMETRIC_THRESHOLD:
            higher_side = "left" if angles["left_elbow_angle"] > angles["right_elbow_angle"] else "right"
            lower_side = "right" if higher_side == "left" else "left"
            return FormError(
                error_type="asymmetric_press",
                message=f"Your {lower_side} arm is lagging behind. Press both arms evenly.",
                severity=ErrorSeverity.WARNING,
                body_part="shoulders",
                current_value=diff,
                target_value=0,
                confidence=0.8
            )

        return None

    def _check_incomplete_lockout(self, angles: dict) -> FormError | None:
        """Check for incomplete lockout at the top of the press."""
        elbow_angles = []
        if "left_elbow_angle" in angles:
            elbow_angles.append(angles["left_elbow_angle"])
        if "right_elbow_angle" in angles:
            elbow_angles.append(angles["right_elbow_angle"])

        if not elbow_angles:
            return None

        avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)

        if avg_elbow_angle < self.LOCKOUT_THRESHOLD:
            return FormError(
                error_type="incomplete_lockout",
                message="Press all the way up! Fully extend your arms overhead.",
                severity=ErrorSeverity.WARNING,
                body_part="elbows",
                current_value=avg_elbow_angle,
                target_value=self.LOCKOUT_THRESHOLD,
                confidence=0.85
            )

        return None
