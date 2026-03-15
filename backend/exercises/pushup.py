"""
Push-up Exercise Form Analyzer

Analyzes push-up form for common errors:
- Insufficient depth (elbows not reaching 90 degrees)
- Hip sag (hips dropping below shoulder-ankle line)
- Hip pike (hips rising above shoulder-ankle line)
- Elbow flare (elbows going too wide)
- Head position (looking up or down excessively)
"""
from ..pose_estimator import PoseResult, Landmark
from .base import (
    ExerciseAnalyzer,
    AnalysisResult,
    FormError,
    ExercisePhase,
    ErrorSeverity
)
import numpy as np


class PushupAnalyzer(ExerciseAnalyzer):
    """
    Analyzer for push-ups.
    
    Designed for side-view camera angle (most common setup).
    """
    
    # Form thresholds (in degrees)
    DEPTH_THRESHOLD = 110  # Elbow angle for minimum depth
    GOOD_DEPTH_THRESHOLD = 90  # Ideal push-up depth
    
    HIP_SAG_THRESHOLD = 10  # Degrees below straight line
    HIP_PIKE_THRESHOLD = 15  # Degrees above straight line
    
    # Phase detection thresholds (relaxed for webcam accuracy)
    PLANK_THRESHOLD = 150  # Arms roughly straight
    LOWERING_THRESHOLD = 130  # Starting to lower
    BOTTOM_THRESHOLD = 110  # At or near bottom
    
    @property
    def exercise_name(self) -> str:
        return "pushup"
    
    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """Detect push-up phase based on elbow angle"""
        if not pose_result.is_valid:
            return self.rep_counter.phase
        
        angles = pose_result.angles
        
        # Use average elbow angle (works for both side angles)
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
        if avg_elbow_angle > self.PLANK_THRESHOLD:
            return ExercisePhase.PLANK
        elif avg_elbow_angle > self.LOWERING_THRESHOLD:
            if current_phase in [ExercisePhase.PLANK, ExercisePhase.LOWERING]:
                return ExercisePhase.LOWERING
            else:
                return ExercisePhase.PUSHING
        elif avg_elbow_angle > self.BOTTOM_THRESHOLD:
            if current_phase in [ExercisePhase.PLANK, ExercisePhase.LOWERING]:
                return ExercisePhase.LOWERING
            elif current_phase == ExercisePhase.LOWEST:
                return ExercisePhase.PUSHING
            else:
                return ExercisePhase.PUSHING
        else:
            return ExercisePhase.LOWEST
    
    def _is_exercise_pose(self, pose_result: PoseResult) -> bool:
        """
        Check if the pose is consistent with a push-up position.

        Push-ups are typically done from a side view. We check that:
        - Shoulders, hips, and at least one wrist are visible
        - The body is roughly horizontal (shoulder-hip y difference is small)

        We're lenient because camera angle matters a lot for push-ups.
        """
        landmarks = pose_result.landmarks
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

        for name in required:
            if name not in landmarks:
                return False
            if landmarks[name].visibility < 0.4:
                return False

        # Need at least one wrist visible (hands on ground)
        has_wrist = (
            ("left_wrist" in landmarks and landmarks["left_wrist"].visibility > 0.3)
            or ("right_wrist" in landmarks and landmarks["right_wrist"].visibility > 0.3)
        )
        if not has_wrist:
            return False

        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2

        # In push-up, body is roughly horizontal (not upright)
        # If hips are far below shoulders → standing/sitting, not push-up
        if abs(mid_hip_y - mid_shoulder_y) > 0.3:
            return False

        return True

    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """Analyze push-up form and detect errors"""
        errors = []

        if not pose_result.is_valid:
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

        # Check if the pose looks like a push-up position
        if not self._is_exercise_pose(pose_result):
            return AnalysisResult(
                exercise_type=self.exercise_name,
                phase=self.rep_counter.phase,
                errors=[FormError(
                    error_type="not_in_position",
                    message="Get into push-up position. Side view works best.",
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

        # Only check form during active phases (not resting in plank)
        if new_phase != ExercisePhase.PLANK:
            # Check depth at bottom
            if new_phase == ExercisePhase.LOWEST:
                depth_error = self._check_depth(angles)
                if depth_error:
                    errors.append(depth_error)

            # Check body alignment (hip sag/pike)
            alignment_error = self._check_body_alignment(landmarks)
            if alignment_error:
                errors.append(alignment_error)

            # Check elbow position (only reliable from front/back view)
            elbow_error = self._check_elbow_position(landmarks, angles)
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
    
    def _check_depth(self, angles: dict) -> FormError | None:
        """Check if push-up reaches proper depth"""
        elbow_angles = []
        if "left_elbow_angle" in angles:
            elbow_angles.append(angles["left_elbow_angle"])
        if "right_elbow_angle" in angles:
            elbow_angles.append(angles["right_elbow_angle"])
        
        if not elbow_angles:
            return None
        
        avg_elbow_angle = sum(elbow_angles) / len(elbow_angles)
        
        if avg_elbow_angle > self.DEPTH_THRESHOLD:
            return FormError(
                error_type="insufficient_depth",
                message="Go lower! Bend your elbows to at least 90 degrees.",
                severity=ErrorSeverity.WARNING,
                body_part="elbows",
                current_value=avg_elbow_angle,
                target_value=self.GOOD_DEPTH_THRESHOLD,
                confidence=0.9
            )
        elif avg_elbow_angle > self.GOOD_DEPTH_THRESHOLD:
            return FormError(
                error_type="almost_full_depth",
                message="Almost there! Just a bit lower for full range.",
                severity=ErrorSeverity.INFO,
                body_part="elbows",
                current_value=avg_elbow_angle,
                target_value=self.GOOD_DEPTH_THRESHOLD,
                confidence=0.8
            )
        
        return None
    
    def _check_body_alignment(self, landmarks: dict) -> FormError | None:
        """Check for hip sag or pike"""
        required = ["left_shoulder", "right_shoulder", "left_hip", 
                   "right_hip", "left_ankle", "right_ankle"]
        if not all(k in landmarks for k in required):
            return None
        
        # Calculate mid-points
        mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
        mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2
        mid_ankle_y = (landmarks["left_ankle"].y + landmarks["right_ankle"].y) / 2
        
        # In image coordinates, y increases downward
        # Calculate expected hip y position on line from shoulder to ankle
        mid_shoulder_x = (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2
        mid_hip_x = (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2
        mid_ankle_x = (landmarks["left_ankle"].x + landmarks["right_ankle"].x) / 2
        
        # Linear interpolation for expected hip position
        if abs(mid_ankle_x - mid_shoulder_x) < 0.01:
            return None  # Avoid division by zero
        
        t = (mid_hip_x - mid_shoulder_x) / (mid_ankle_x - mid_shoulder_x)
        expected_hip_y = mid_shoulder_y + t * (mid_ankle_y - mid_shoulder_y)
        
        # Deviation from straight line (normalized)
        deviation = mid_hip_y - expected_hip_y
        
        # Threshold scaled to normalized coordinates
        # Sag is the main differentiator between correct/wrong push-ups
        sag_threshold = 0.06
        pike_threshold = 0.07
        
        if deviation > sag_threshold:
            return FormError(
                error_type="hip_sag",
                message="Squeeze your core! Your hips are sagging down.",
                severity=ErrorSeverity.ERROR,
                body_part="hips",
                current_value=deviation,
                target_value=0,
                confidence=0.85
            )
        elif deviation < -pike_threshold:
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
    
    def _check_elbow_position(self, landmarks: dict, angles: dict) -> FormError | None:
        """Check for elbow flare (elbows going too wide).

        Only reliable when viewed from front/back — from side view,
        the x-axis compression makes elbows appear inside/outside
        shoulders regardless of actual position.
        """
        required = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]
        if not all(k in landmarks for k in required):
            return None

        shoulder_width = abs(landmarks["left_shoulder"].x - landmarks["right_shoulder"].x)

        # If shoulder width is very small, we're likely viewing from the side
        # where this check is unreliable — skip it
        if shoulder_width < 0.08:
            return None

        # Relaxed threshold: elbows must be VERY far outside shoulders
        left_elbow_outside = landmarks["left_elbow"].x < landmarks["left_shoulder"].x - shoulder_width * 0.6
        right_elbow_outside = landmarks["right_elbow"].x > landmarks["right_shoulder"].x + shoulder_width * 0.6

        if left_elbow_outside or right_elbow_outside:
            return FormError(
                error_type="elbow_flare",
                message="Tuck your elbows in! Keep them at about 45 degrees from your body.",
                severity=ErrorSeverity.WARNING,
                body_part="elbows",
                confidence=0.6,
            )

        return None


