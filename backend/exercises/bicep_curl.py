"""
Bicep Curl Exercise Form Analyzer

Analyzes bicep curl form for common errors:
- Swinging (using momentum instead of strict form)
- Incomplete range of motion (not curling fully)
- Elbow drift (elbows moving away from the body)
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class BicepCurlAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for bicep curls (dumbbell or barbell).

    Tracks elbow angle and shoulder stability. Upper-body only exercise,
    so leg landmarks are not required.

    Phase mapping (reusing available enums):
    - STANDING: arms down, elbow > 150
    - ASCENDING: curling up, elbow angle decreasing
    - BOTTOM: top of curl (elbow most bent, < 60)
    - DESCENDING: lowering back down
    """

    # Form thresholds (relaxed for webcam variability)
    SWING_THRESHOLD = 0.06  # Max shoulder-hip alignment shift
    INCOMPLETE_ROM_THRESHOLD = 75  # Elbow angle at top of curl
    ELBOW_DRIFT_THRESHOLD = 0.10  # Max elbow x drift from hip x

    # Phase detection thresholds (relaxed for webcam front-view angle compression)
    # From a front-facing webcam, elbow angles appear much smaller than actual
    STANDING_THRESHOLD = 130  # Arms at sides (real data shows 128-145)
    ASCENDING_THRESHOLD = 118  # Mid-curl
    BOTTOM_THRESHOLD = 108  # Top of curl (real data shows 99-112)

    @property
    def exercise_name(self) -> str:
        return "bicep_curl"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect bicep curl phase based on elbow angle."""
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
        if avg_elbow_angle > self.STANDING_THRESHOLD:
            return ExercisePhase.STANDING
        elif avg_elbow_angle > self.ASCENDING_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.ASCENDING]:
                return ExercisePhase.ASCENDING
            else:
                return ExercisePhase.DESCENDING
        elif avg_elbow_angle > self.BOTTOM_THRESHOLD:
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
        Check if the pose is consistent with a bicep curl position.

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
        """Analyze bicep curl form and detect errors."""
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

        # Check if the pose looks like a bicep curl position
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
            # Check for body swinging
            swing_error = self._check_swinging(angles)
            if swing_error:
                errors.append(swing_error)

            # Check ROM at the top of curl
            if new_phase == ExercisePhase.BOTTOM:
                rom_error = self._check_incomplete_rom(angles)
                if rom_error:
                    errors.append(rom_error)

            # Check elbow drift
            elbow_error = self._check_elbow_drift(landmarks)
            if elbow_error:
                errors.append(elbow_error)

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

    def _check_swinging(self, angles: dict) -> FormError | None:
        """Check for body swinging (momentum use) via shoulder-hip alignment."""
        if "shoulder_hip_alignment" not in angles:
            return None

        alignment = angles["shoulder_hip_alignment"]

        if alignment > self.SWING_THRESHOLD:
            return FormError(
                error_type="swinging",
                message="Stop swinging! Keep your torso still and use only your arms.",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=alignment,
                target_value=0,
                confidence=0.8
            )

        return None

    def _check_incomplete_rom(self, angles: dict) -> FormError | None:
        """Check for incomplete range of motion at top of curl."""
        elbow_angles = []
        if "left_elbow_angle" in angles:
            elbow_angles.append(angles["left_elbow_angle"])
        if "right_elbow_angle" in angles:
            elbow_angles.append(angles["right_elbow_angle"])

        if not elbow_angles:
            return None

        avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)

        if avg_elbow_angle > self.INCOMPLETE_ROM_THRESHOLD:
            return FormError(
                error_type="incomplete_rom",
                message="Curl all the way up! Squeeze your biceps at the top.",
                severity=ErrorSeverity.WARNING,
                body_part="elbows",
                current_value=avg_elbow_angle,
                target_value=45.0,
                confidence=0.85
            )

        return None

    def _check_elbow_drift(self, landmarks: dict) -> FormError | None:
        """Check if elbows are drifting away from the body."""
        sides_checked = []

        if all(k in landmarks for k in ["left_elbow", "left_hip"]):
            left_drift = abs(landmarks["left_elbow"].x - landmarks["left_hip"].x)
            sides_checked.append(("left", left_drift))

        if all(k in landmarks for k in ["right_elbow", "right_hip"]):
            right_drift = abs(landmarks["right_elbow"].x - landmarks["right_hip"].x)
            sides_checked.append(("right", right_drift))

        if not sides_checked:
            return None

        for side, drift in sides_checked:
            if drift > self.ELBOW_DRIFT_THRESHOLD:
                return FormError(
                    error_type="elbow_drift",
                    message=f"Pin your {side} elbow to your side! It's drifting out.",
                    severity=ErrorSeverity.WARNING,
                    body_part=f"{side}_elbow",
                    current_value=drift,
                    target_value=0,
                    confidence=0.75
                )

        return None
