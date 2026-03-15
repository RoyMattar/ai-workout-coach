"""
Deadlift Exercise Form Analyzer

Analyzes deadlift form for common errors:
- Rounded back (excessive torso forward lean during hip hinge)
- Knees too bent (should be a hip hinge, not a squat)
- Incomplete lockout (not fully extending hips at the top)
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class DeadliftAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for conventional and Romanian deadlifts.

    Tracks hip angle (hip hinge), knee angle, and torso angle (spine).
    Phase detection is driven by the hip angle since the deadlift is
    primarily a hip-hinge movement.
    """

    # Form thresholds (in degrees)
    ROUNDED_BACK_THRESHOLD = 60  # Max torso angle during descent/bottom
    KNEES_TOO_BENT_THRESHOLD = 90   # Knee angle floor (not a squat)
    LOCKOUT_HIP_THRESHOLD = 160  # Hip angle for full lockout (relaxed)

    # Phase detection thresholds (based on hip angle)
    STANDING_THRESHOLD = 160  # Hips fully extended
    DESCENDING_THRESHOLD = 130  # Starting to hinge
    BOTTOM_THRESHOLD = 130  # At or near bottom of hinge

    @property
    def exercise_name(self) -> str:
        return "deadlift"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect deadlift phase based on hip angle (hip hinge)."""
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

        # State machine for phase detection (based on hip angle)
        if avg_hip_angle > self.STANDING_THRESHOLD:
            return ExercisePhase.STANDING
        elif avg_hip_angle > self.DESCENDING_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.DESCENDING]:
                return ExercisePhase.DESCENDING
            else:
                return ExercisePhase.ASCENDING
        else:
            # Below BOTTOM_THRESHOLD
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.DESCENDING]:
                return ExercisePhase.BOTTOM
            elif current_phase == ExercisePhase.BOTTOM:
                return ExercisePhase.BOTTOM
            else:
                return ExercisePhase.ASCENDING

    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a deadlift position.

        Need shoulders, hips, and knees visible. The person should be
        in a roughly upright orientation (shoulders above hips).
        """
        landmarks = pose_result.landmarks

        core = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                "left_knee", "right_knee"]

        for name in core:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.45:
                return False

        # Shoulders should be above or near hip level (not lying down)
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2

        # Allow for hinge — shoulders can be somewhat below hip level
        # but not extremely (that would mean lying down)
        if mid_shoulder_y > mid_hip_y + 0.2:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze deadlift form and detect errors."""
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

        # Check if the pose looks like a deadlift position
        if not self._is_exercise_pose(pose_result):
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Stand with your full body visible. Side view works best.",
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

        # Only check form during active phases (not standing)
        if new_phase != ExercisePhase.STANDING:
            # Check for rounded back during descent and bottom
            if new_phase in [ExercisePhase.DESCENDING, ExercisePhase.BOTTOM]:
                back_error = self._check_rounded_back(angles)
                if back_error:
                    errors.append(back_error)

            # Check for excessive knee bend
            knee_error = self._check_knees_too_bent(angles)
            if knee_error:
                errors.append(knee_error)
        else:
            # Check lockout when standing
            lockout_error = self._check_lockout(angles)
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

    def _check_rounded_back(self, angles: dict) -> FormError | None:
        """Check for excessive torso forward lean (rounded back)."""
        if "torso_angle" not in angles:
            return None

        torso_angle = angles["torso_angle"]

        if torso_angle > self.ROUNDED_BACK_THRESHOLD:
            return FormError(
                error_type="rounded_back",
                message="Keep your back straight! You're rounding your spine.",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=torso_angle,
                target_value=self.ROUNDED_BACK_THRESHOLD,
                confidence=0.85
            )

        return None

    def _check_knees_too_bent(self, angles: dict) -> FormError | None:
        """Check if knees are bending too much (should be a hip hinge)."""
        knee_angles = []
        if "left_knee_angle" in angles:
            knee_angles.append(angles["left_knee_angle"])
        if "right_knee_angle" in angles:
            knee_angles.append(angles["right_knee_angle"])

        if not knee_angles:
            return None

        avg_knee_angle = sum(knee_angles) / len(knee_angles)

        if avg_knee_angle < self.KNEES_TOO_BENT_THRESHOLD:
            return FormError(
                error_type="knees_too_bent",
                message="Straighten your legs more! This is a hip hinge, not a squat.",
                severity=ErrorSeverity.WARNING,
                body_part="knees",
                current_value=avg_knee_angle,
                target_value=self.KNEES_TOO_BENT_THRESHOLD,
                confidence=0.85
            )

        return None

    def _check_lockout(self, angles: dict) -> FormError | None:
        """Check for incomplete hip lockout at the top."""
        hip_angles = []
        if "left_hip_angle" in angles:
            hip_angles.append(angles["left_hip_angle"])
        if "right_hip_angle" in angles:
            hip_angles.append(angles["right_hip_angle"])

        if not hip_angles:
            return None

        avg_hip_angle = sum(hip_angles) / len(hip_angles)

        if avg_hip_angle < self.LOCKOUT_HIP_THRESHOLD:
            return FormError(
                error_type="lockout_incomplete",
                message="Squeeze your glutes and stand tall! Fully extend your hips.",
                severity=ErrorSeverity.INFO,
                body_part="hips",
                current_value=avg_hip_angle,
                target_value=self.LOCKOUT_HIP_THRESHOLD,
                confidence=0.8
            )

        return None
