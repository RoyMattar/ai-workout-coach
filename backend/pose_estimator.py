"""
Pose Estimation Module using MediaPipe PoseLandmarker (Tasks API)

This module handles real-time body pose detection and landmark extraction.
It provides normalized landmark coordinates and calculates joint angles.

Uses the MediaPipe Tasks API (PoseLandmarker) which is the current supported
API for MediaPipe >= 0.10.20.
"""
import logging
import math
import os
import urllib.request

# Suppress TensorFlow Lite delegate warnings from MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
    PoseLandmark,
)

logger = logging.getLogger(__name__)

# Model download URL and path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_DIR = Path(__file__).parent / "models" / "pretrained"
MODEL_PATH = MODEL_DIR / "pose_landmarker_heavy.task"


@dataclass
class Landmark:
    """Represents a single body landmark"""
    x: float
    y: float
    z: float
    visibility: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class PoseResult:
    """Result from pose estimation"""
    landmarks: dict[str, Landmark]
    angles: dict[str, float]
    is_valid: bool
    confidence: float
    raw_landmarks: Optional[list] = None


def _ensure_model():
    """Download the pose landmarker model if not present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading pose landmarker model to {MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(
            f"Could not download pose landmarker model. "
            f"Please manually download from {MODEL_URL} "
            f"and place at {MODEL_PATH}"
        ) from e

    return str(MODEL_PATH)


class PoseEstimator:
    """
    MediaPipe-based pose estimation for exercise form analysis.

    Uses PoseLandmarker (Tasks API) to extract 33 body landmarks
    and calculates relevant joint angles for exercise form analysis.
    """

    # Landmark indices we care about
    LANDMARK_MAP = {
        PoseLandmark.NOSE: "nose",
        PoseLandmark.LEFT_SHOULDER: "left_shoulder",
        PoseLandmark.RIGHT_SHOULDER: "right_shoulder",
        PoseLandmark.LEFT_ELBOW: "left_elbow",
        PoseLandmark.RIGHT_ELBOW: "right_elbow",
        PoseLandmark.LEFT_WRIST: "left_wrist",
        PoseLandmark.RIGHT_WRIST: "right_wrist",
        PoseLandmark.LEFT_HIP: "left_hip",
        PoseLandmark.RIGHT_HIP: "right_hip",
        PoseLandmark.LEFT_KNEE: "left_knee",
        PoseLandmark.RIGHT_KNEE: "right_knee",
        PoseLandmark.LEFT_ANKLE: "left_ankle",
        PoseLandmark.RIGHT_ANKLE: "right_ankle",
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = _ensure_model()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker initialized (Tasks API)")

    def process_frame(self, frame: np.ndarray) -> PoseResult:
        """
        Process a single frame and extract pose landmarks.

        Args:
            frame: BGR image frame from camera (numpy array)

        Returns:
            PoseResult with landmarks and calculated angles
        """
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1].copy() if frame.shape[2] == 3 else frame.copy()

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect pose
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PoseResult(
                landmarks={},
                angles={},
                is_valid=False,
                confidence=0.0,
            )

        # Extract landmarks from first detected pose
        raw_lms = result.pose_landmarks[0]
        landmarks = self._extract_landmarks(raw_lms)
        angles = self._calculate_angles(landmarks)
        confidence = self._calculate_confidence(landmarks)

        return PoseResult(
            landmarks=landmarks,
            angles=angles,
            is_valid=True,
            confidence=confidence,
            raw_landmarks=raw_lms,
        )

    def _extract_landmarks(self, raw_landmarks) -> dict[str, Landmark]:
        """Extract relevant landmarks from MediaPipe output."""
        landmarks = {}

        for idx, name in self.LANDMARK_MAP.items():
            lm = raw_landmarks[idx]
            landmarks[name] = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else lm.presence,
            )

        return landmarks

    def _calculate_angles(self, landmarks: dict[str, Landmark]) -> dict[str, float]:
        """
        Calculate joint angles relevant for exercise analysis.

        Returns angles in degrees for:
        - left_knee_angle, right_knee_angle
        - left_hip_angle, right_hip_angle
        - left_elbow_angle, right_elbow_angle
        - torso_angle (relative to vertical)
        - shoulder_hip_alignment
        """
        angles = {}

        # Knee angles
        if all(k in landmarks for k in ["left_hip", "left_knee", "left_ankle"]):
            angles["left_knee_angle"] = self._calculate_angle(
                landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"]
            )

        if all(k in landmarks for k in ["right_hip", "right_knee", "right_ankle"]):
            angles["right_knee_angle"] = self._calculate_angle(
                landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"]
            )

        # Hip angles
        if all(k in landmarks for k in ["left_shoulder", "left_hip", "left_knee"]):
            angles["left_hip_angle"] = self._calculate_angle(
                landmarks["left_shoulder"], landmarks["left_hip"], landmarks["left_knee"]
            )

        if all(k in landmarks for k in ["right_shoulder", "right_hip", "right_knee"]):
            angles["right_hip_angle"] = self._calculate_angle(
                landmarks["right_shoulder"], landmarks["right_hip"], landmarks["right_knee"]
            )

        # Elbow angles
        if all(k in landmarks for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            angles["left_elbow_angle"] = self._calculate_angle(
                landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"]
            )

        if all(k in landmarks for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            angles["right_elbow_angle"] = self._calculate_angle(
                landmarks["right_shoulder"], landmarks["right_elbow"], landmarks["right_wrist"]
            )

        # Torso angle
        if all(k in landmarks for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            angles["torso_angle"] = self._calculate_torso_angle(landmarks)
            angles["shoulder_hip_alignment"] = self._calculate_alignment(landmarks)

        return angles

    @staticmethod
    def _calculate_angle(point1: Landmark, point2: Landmark, point3: Landmark) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3.

        Returns angle in degrees (0-180).
        """
        a = point1.to_array()
        b = point2.to_array()
        c = point3.to_array()

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return float(np.degrees(angle))

    def _calculate_torso_angle(self, landmarks: dict[str, Landmark]) -> float:
        """Calculate torso angle relative to vertical."""
        mid_shoulder = np.array([
            (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2,
            (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2,
        ])
        mid_hip = np.array([
            (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2,
            (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2,
        ])

        torso_vector = mid_shoulder - mid_hip
        vertical = np.array([0, -1])  # Up is negative y in image coordinates

        cosine_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return float(np.degrees(angle))

    def _calculate_alignment(self, landmarks: dict[str, Landmark]) -> float:
        """Calculate shoulder-hip alignment (lateral shift)."""
        shoulder_center_x = (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2
        hip_center_x = (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2
        return abs(shoulder_center_x - hip_center_x)

    def _calculate_confidence(self, landmarks: dict[str, Landmark]) -> float:
        """Calculate overall pose confidence from landmark visibilities."""
        if not landmarks:
            return 0.0
        visibilities = [lm.visibility for lm in landmarks.values()]
        return sum(visibilities) / len(visibilities)

    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
