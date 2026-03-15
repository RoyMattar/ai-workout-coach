"""
Squat Exercise Form Analyzer

Analyzes squat form for common errors:
- Insufficient depth (knees not reaching 90 degrees)
- Knee cave (knees collapsing inward)
- Forward lean (torso too far forward)
- Heel rise (weight shifting to toes)
- Asymmetric movement (one side lower than the other)
"""
from ..pose_estimator import PoseResult
from .base import (
    ExerciseAnalyzer, 
    AnalysisResult, 
    FormError, 
    ExercisePhase,
    ErrorSeverity
)


class SquatAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for bodyweight and barbell squats.
    
    Thresholds based on exercise science guidelines for proper squat form.
    """
    
    # Form thresholds (in degrees)
    DEPTH_THRESHOLD = 110  # Knee angle for acceptable depth
    GOOD_DEPTH_THRESHOLD = 95  # Ideal squat depth
    DEEP_SQUAT_THRESHOLD = 70  # ATG squat

    FORWARD_LEAN_THRESHOLD = 50  # Max acceptable torso forward lean
    KNEE_OVER_TOE_THRESHOLD = 0.05  # Normalized distance

    # Phase detection thresholds (relaxed for webcam accuracy)
    STANDING_THRESHOLD = 150  # Roughly straight legs
    DESCENDING_THRESHOLD = 135  # Starting to descend
    BOTTOM_THRESHOLD = 120  # At or near bottom
    
    @property
    def exercise_name(self) -> str:
        return "squat"
    
    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect squat phase based on knee angle"""
        if not pose_result.is_valid:
            return self.rep_counter.phase
        
        angles = pose_result.angles
        
        # Use average knee angle
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
        Check if the detected pose is consistent with a standing/squatting posture.

        For squats we need to see at least the torso and legs. We check:
        - Key landmarks (shoulders, hips, knees) are visible with sufficient confidence
        - The person appears to be upright (shoulders above hips, hips above knees)
        - Knees are in the lower portion of the frame (not just a face/upper body crop)

        Note: We don't strictly require ankles — a slightly cropped view where
        knees are visible is sufficient for basic squat analysis.
        """
        landmarks = pose_result.landmarks

        # Core landmarks needed for squat analysis
        core = ["left_hip", "right_hip", "left_knee", "right_knee",
                "left_shoulder", "right_shoulder"]

        for name in core:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.45:
                return False

        # Check body is upright: shoulders above hips above knees
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2
        mid_knee_y = (landmarks["left_knee"].y + landmarks["right_knee"].y) / 2

        if not (mid_shoulder_y < mid_hip_y < mid_knee_y):
            return False

        # Knees should be in the lower half of frame (filters out sitting at desk)
        if mid_knee_y < 0.35:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze squat form and detect errors"""
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

        # Check if the pose looks like the person is actually exercising
        if not self._is_exercise_pose(pose_result):
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Step back so your knees and torso are visible in the camera.",
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
            # Check depth (at bottom of squat)
            if new_phase == ExercisePhase.BOTTOM:
                depth_error = self._check_depth(angles)
                if depth_error:
                    errors.append(depth_error)

            # Check forward lean
            lean_error = self._check_forward_lean(angles)
            if lean_error:
                errors.append(lean_error)

            # Check knee tracking
            knee_error = self._check_knee_tracking(landmarks, angles)
            if knee_error:
                errors.append(knee_error)

            # Check symmetry
            symmetry_error = self._check_symmetry(angles)
            if symmetry_error:
                errors.append(symmetry_error)

        # Track error history
        self.error_history.append(errors)

        # Determine if form is good (no errors or only info-level)
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
        """Check if squat reaches proper depth"""
        knee_angles = []
        if "left_knee_angle" in angles:
            knee_angles.append(angles["left_knee_angle"])
        if "right_knee_angle" in angles:
            knee_angles.append(angles["right_knee_angle"])
        
        if not knee_angles:
            return None
        
        avg_knee_angle = sum(knee_angles) / len(knee_angles)
        
        if avg_knee_angle > self.DEPTH_THRESHOLD:
            return FormError(
                error_type="insufficient_depth",
                message="Go deeper! Try to get your thighs parallel to the ground.",
                severity=ErrorSeverity.WARNING,
                body_part="knees",
                current_value=avg_knee_angle,
                target_value=self.GOOD_DEPTH_THRESHOLD,
                confidence=0.9
            )
        elif avg_knee_angle > self.GOOD_DEPTH_THRESHOLD:
            return FormError(
                error_type="almost_parallel",
                message="Almost there! Just a bit deeper for full depth.",
                severity=ErrorSeverity.INFO,
                body_part="knees",
                current_value=avg_knee_angle,
                target_value=self.GOOD_DEPTH_THRESHOLD,
                confidence=0.8
            )
        
        return None
    
    def _check_forward_lean(self, angles: dict) -> FormError | None:
        """Check for excessive forward lean"""
        if "torso_angle" not in angles:
            return None
        
        torso_angle = angles["torso_angle"]
        
        if torso_angle > self.FORWARD_LEAN_THRESHOLD:
            return FormError(
                error_type="excessive_forward_lean",
                message="Keep your chest up! You're leaning too far forward.",
                severity=ErrorSeverity.ERROR,
                body_part="torso",
                current_value=torso_angle,
                target_value=self.FORWARD_LEAN_THRESHOLD,
                confidence=0.85
            )
        
        return None
    
    def _check_knee_tracking(self, landmarks: dict, angles: dict) -> FormError | None:
        """Check if knees are tracking over toes (not caving in)"""
        required = ["left_knee", "right_knee", "left_ankle", "right_ankle", 
                   "left_hip", "right_hip"]
        if not all(k in landmarks for k in required):
            return None
        
        # Calculate knee-ankle alignment
        left_knee_x = landmarks["left_knee"].x
        left_ankle_x = landmarks["left_ankle"].x
        left_hip_x = landmarks["left_hip"].x
        
        right_knee_x = landmarks["right_knee"].x
        right_ankle_x = landmarks["right_ankle"].x
        right_hip_x = landmarks["right_hip"].x
        
        # Knees should track between hip and ankle, slightly outside ankle
        # Knee cave = knee going significantly inside the ankle line
        # From side view, x-positions are unreliable, so only flag obvious cases
        # Also skip if shoulder width is small (side view)
        shoulder_width = abs(landmarks.get("left_shoulder", landmarks["left_hip"]).x
                            - landmarks.get("right_shoulder", landmarks["right_hip"]).x)
        if shoulder_width < 0.08:
            return None  # Side view — can't reliably detect knee cave

        left_cave = left_knee_x > left_ankle_x + 0.04  # Relaxed threshold
        right_cave = right_knee_x < right_ankle_x - 0.04
        
        if left_cave or right_cave:
            side = "left" if left_cave else "right"
            return FormError(
                error_type="knee_cave",
                message=f"Push your {side} knee out! Don't let it collapse inward.",
                severity=ErrorSeverity.ERROR,
                body_part=f"{side}_knee",
                confidence=0.8
            )
        
        return None
    
    def _check_symmetry(self, angles: dict) -> FormError | None:
        """Check for asymmetric squat pattern"""
        if "left_knee_angle" not in angles or "right_knee_angle" not in angles:
            return None
        
        diff = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
        
        if diff > 20:  # More than 20 degrees difference
            lower_side = "left" if angles["left_knee_angle"] < angles["right_knee_angle"] else "right"
            return FormError(
                error_type="asymmetric_depth",
                message=f"You're going deeper on your {lower_side} side. Try to keep both sides even.",
                severity=ErrorSeverity.WARNING,
                body_part="hips",
                current_value=diff,
                target_value=0,
                confidence=0.7
            )
        
        return None


