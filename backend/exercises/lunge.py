"""
Lunge Exercise Form Analyzer

Analyzes lunge form for common errors:
- Front knee too far (knee past toes)
- Insufficient depth (front knee angle too open)
- Torso lean (excessive forward lean)
- Rear knee high (rear knee not bending enough)
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)


class LungeAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for forward and stationary lunges.

    Tracks front knee angle, rear knee angle, and torso angle.
    Designed for side-view camera angle for best results.
    """

    # Form thresholds (in degrees)
    DEPTH_THRESHOLD = 110  # Front knee angle for acceptable depth
    GOOD_DEPTH_THRESHOLD = 90  # Ideal lunge depth
    TORSO_LEAN_THRESHOLD = 50  # Max acceptable torso forward lean
    REAR_KNEE_THRESHOLD = 130  # Rear knee should bend below this
    KNEE_OVER_TOE_THRESHOLD = 0.03  # Normalized x-distance threshold

    # Phase detection thresholds (based on average knee angle)
    STANDING_THRESHOLD = 150  # Roughly straight legs
    DESCENDING_THRESHOLD = 135  # Starting to descend
    BOTTOM_THRESHOLD = 120  # At or near bottom

    @property
    def exercise_name(self) -> str:
        return "lunge"

    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect lunge phase based on average available knee angle."""
        if not pose_result.is_valid:
            return self.rep_counter.phase

        angles = pose_result.angles

        # Use average of available knee angles
        knee_angles = []
        if "left_knee_angle" in angles:
            knee_angles.append(angles["left_knee_angle"])
        if "right_knee_angle" in angles:
            knee_angles.append(angles["right_knee_angle"])

        if not knee_angles:
            return self.rep_counter.phase

        avg_knee_angle = sum(knee_angles) / len(knee_angles)
        current_phase = self.rep_counter.phase

        # State machine for phase detection
        if avg_knee_angle > self.STANDING_THRESHOLD:
            return ExercisePhase.STANDING
        elif avg_knee_angle > self.DESCENDING_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.DESCENDING]:
                return ExercisePhase.DESCENDING
            else:
                return ExercisePhase.ASCENDING
        elif avg_knee_angle > self.BOTTOM_THRESHOLD:
            if current_phase in [ExercisePhase.STANDING, ExercisePhase.DESCENDING]:
                return ExercisePhase.DESCENDING
            elif current_phase == ExercisePhase.BOTTOM:
                return ExercisePhase.ASCENDING
            else:
                return ExercisePhase.ASCENDING
        else:
            return ExercisePhase.BOTTOM

    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a lunge position.

        For lunges we need shoulders, hips, and at least one knee visible.
        The body should be roughly upright (shoulders above hips).
        """
        landmarks = pose_result.landmarks

        # Core landmarks needed
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        for name in required:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.45:
                return False

        # Need at least one knee visible
        has_knee = (
            ("left_knee" in landmarks and landmarks["left_knee"].visibility > 0.4)
            or ("right_knee" in landmarks and landmarks["right_knee"].visibility > 0.4)
        )
        if not has_knee:
            return False

        # Check body is upright: shoulders above hips
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2

        if not (mid_shoulder_y < mid_hip_y):
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze lunge form and detect errors."""
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

        # Check if the pose looks like a lunge position
        if not self._is_exercise_pose(pose_result):
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Step back so your full body is visible in the camera.",
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
            # Check depth at bottom
            if new_phase == ExercisePhase.BOTTOM:
                depth_error = self._check_depth(angles)
                if depth_error:
                    errors.append(depth_error)

                rear_knee_error = self._check_rear_knee(angles)
                if rear_knee_error:
                    errors.append(rear_knee_error)

            # Check front knee past toes
            knee_error = self._check_front_knee(landmarks)
            if knee_error:
                errors.append(knee_error)

            # Check torso lean
            lean_error = self._check_torso_lean(angles)
            if lean_error:
                errors.append(lean_error)

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

    def _check_depth(self, angles: dict) -> FormError | None:
        """Check if lunge reaches proper depth based on front knee angle."""
        knee_angles = []
        if "left_knee_angle" in angles:
            knee_angles.append(angles["left_knee_angle"])
        if "right_knee_angle" in angles:
            knee_angles.append(angles["right_knee_angle"])

        if not knee_angles:
            return None

        # Front knee is the one with the smaller angle (more bent)
        front_knee_angle = min(knee_angles)

        if front_knee_angle > self.DEPTH_THRESHOLD:
            return FormError(
                error_type="insufficient_depth",
                message="Go deeper! Lower your hips until your front thigh is parallel.",
                severity=ErrorSeverity.WARNING,
                body_part="knees",
                current_value=front_knee_angle,
                target_value=self.GOOD_DEPTH_THRESHOLD,
                confidence=0.9
            )

        return None

    def _check_front_knee(self, landmarks: dict) -> FormError | None:
        """Check if front knee is tracking past the toes."""
        # Check both sides and detect which is the front leg
        sides = []
        if all(k in landmarks for k in ["left_knee", "left_ankle"]):
            sides.append(("left", landmarks["left_knee"], landmarks["left_ankle"]))
        if all(k in landmarks for k in ["right_knee", "right_ankle"]):
            sides.append(("right", landmarks["right_knee"], landmarks["right_ankle"]))

        if not sides:
            return None

        for side, knee, ankle in sides:
            # Knee x should not go far past ankle x (past toes)
            # Account for both left-facing and right-facing views
            knee_past_toes = abs(knee.x - ankle.x) > self.KNEE_OVER_TOE_THRESHOLD
            knee_forward = knee.y < ankle.y  # Knee is higher, so it's the front leg

            if knee_forward and knee_past_toes:
                return FormError(
                    error_type="front_knee_too_far",
                    message=f"Your {side} knee is going past your toes. Keep it over your ankle.",
                    severity=ErrorSeverity.ERROR,
                    body_part=f"{side}_knee",
                    confidence=0.8
                )

        return None

    def _check_torso_lean(self, angles: dict) -> FormError | None:
        """Check for excessive forward lean."""
        if "torso_angle" not in angles:
            return None

        torso_angle = angles["torso_angle"]

        if torso_angle > self.TORSO_LEAN_THRESHOLD:
            return FormError(
                error_type="torso_lean",
                message="Keep your torso upright! You're leaning too far forward.",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=torso_angle,
                target_value=self.TORSO_LEAN_THRESHOLD,
                confidence=0.85
            )

        return None

    def _check_rear_knee(self, angles: dict) -> FormError | None:
        """Check if rear knee is bending enough."""
        knee_angles = []
        if "left_knee_angle" in angles:
            knee_angles.append(angles["left_knee_angle"])
        if "right_knee_angle" in angles:
            knee_angles.append(angles["right_knee_angle"])

        if len(knee_angles) < 2:
            return None

        # Rear knee is the one with the larger angle (less bent)
        rear_knee_angle = max(knee_angles)

        if rear_knee_angle > self.REAR_KNEE_THRESHOLD:
            return FormError(
                error_type="rear_knee_high",
                message="Bend your rear knee more! Lower it toward the ground.",
                severity=ErrorSeverity.WARNING,
                body_part="knees",
                current_value=rear_knee_angle,
                target_value=90.0,
                confidence=0.8
            )

        return None
