"""
Form Quality Classifier

Wraps a pre-trained scikit-learn classifier that predicts exercise form quality
from normalized pose features. This is the second AI model in the orchestration
pipeline, operating on structured numerical data (a different domain from the
computer vision model used for pose estimation).

The classifier was trained on synthetic pose data generated from biomechanically-
informed thresholds. See tools/train_form_classifier.py for the training process.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Feature definitions per exercise
SQUAT_FEATURES = [
    "avg_knee_angle",
    "left_knee_angle",
    "right_knee_angle",
    "torso_angle",
    "left_hip_angle",
    "right_hip_angle",
    "knee_angle_diff",
    "knee_ankle_x_offset",
    "shoulder_hip_alignment",
]

PUSHUP_FEATURES = [
    "avg_elbow_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "hip_deviation",
    "torso_angle",
    "left_hip_angle",
    "right_hip_angle",
    "shoulder_width_ratio",
]

QUALITY_LABELS = {0: "good_form", 1: "minor_issues", 2: "major_issues"}

# Binary labels for real-data models (correct vs wrong form)
BINARY_LABELS = {0: "bad_form", 1: "good_form"}

# Unified feature set used by real-data models (raw angles from PoseEstimator)
UNIFIED_FEATURES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "torso_angle",
    "shoulder_hip_alignment",
]

SQUAT_ERROR_TYPES = [
    "insufficient_depth",
    "excessive_forward_lean",
    "knee_cave",
    "asymmetric_depth",
]

PUSHUP_ERROR_TYPES = [
    "insufficient_depth",
    "hip_sag",
    "hip_pike",
    "elbow_flare",
]


@dataclass
class FormClassification:
    """Result from the ML form quality classifier."""
    quality_label: str            # "good_form", "minor_issues", "major_issues"
    quality_score: int            # 0, 1, or 2
    confidence: float             # Classifier confidence (0-1)
    predicted_errors: list[str] = field(default_factory=list)  # Predicted error types
    feature_values: dict = field(default_factory=dict)         # Input features used

    def to_dict(self) -> dict:
        return {
            "quality_label": self.quality_label,
            "quality_score": self.quality_score,
            "confidence": round(self.confidence, 3),
            "predicted_errors": self.predicted_errors,
        }


class FormClassifier:
    """
    Pre-trained ML classifier for exercise form quality.

    Loads a serialized scikit-learn pipeline (StandardScaler + SVM/RandomForest)
    and classifies pose features into form quality categories.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the form classifier.

        Loads real-data models (trained on video data) as primary,
        with synthetic-data models as fallback.
        """
        if model_dir is None:
            model_dir = str(Path(__file__).parent / "pretrained")

        self.model_dir = Path(model_dir)
        self.models: dict = {}           # Primary models (real-data preferred)
        self.model_types: dict = {}      # "realdata" or "synthetic" per exercise
        self.metadata: dict = {}

        self._load_models()

    def _load_models(self):
        """Load all available pre-trained models. Real-data models take priority."""
        exercises = ["squat", "pushup", "lunge", "plank", "deadlift",
                     "bicep_curl", "shoulder_press", "situp"]

        for exercise in exercises:
            # Try real-data model first (trained on actual video frames)
            realdata_path = self.model_dir / f"{exercise}_form_realdata.joblib"
            if realdata_path.exists():
                self.models[exercise] = joblib.load(realdata_path)
                self.model_types[exercise] = "realdata"
                logger.info(f"Loaded {exercise} real-data classifier (primary)")
                continue

            # Fall back to synthetic-data model
            synthetic_path = self.model_dir / f"{exercise}_classifier.joblib"
            if synthetic_path.exists():
                self.models[exercise] = joblib.load(synthetic_path)
                self.model_types[exercise] = "synthetic"
                logger.info(f"Loaded {exercise} synthetic classifier (fallback)")

                metadata_path = self.model_dir / f"{exercise}_classifier_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        self.metadata[exercise] = json.load(f)

    def is_available(self, exercise: str) -> bool:
        """Check if a classifier is available for the given exercise."""
        return exercise in self.models

    def classify(self, exercise: str, pose_angles: dict, pose_landmarks: dict) -> FormClassification:
        """
        Classify form quality from pose features.

        Uses real-data model (binary: good/bad form) if available,
        otherwise falls back to synthetic-data model (3-class: good/minor/major).
        """
        if not self.is_available(exercise):
            return FormClassification(
                quality_label="unknown",
                quality_score=-1,
                confidence=0.0,
            )

        model_type = self.model_types.get(exercise, "synthetic")
        model = self.models[exercise]

        if model_type == "realdata":
            return self._classify_realdata(model, pose_angles)
        else:
            return self._classify_synthetic(model, exercise, pose_angles, pose_landmarks)

    def _classify_realdata(self, model, pose_angles: dict) -> FormClassification:
        """Classify using real-data model (unified feature set, binary output)."""
        feature_vector = np.array([[
            pose_angles.get(f, 170.0 if "angle" in f else 0.0)
            for f in UNIFIED_FEATURES
        ]])

        prediction = model.predict(feature_vector)[0]

        # Get confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            confidence = float(max(probs))
        else:
            confidence = 0.85

        # Map binary prediction to our quality labels
        if prediction == 1:  # correct form
            quality_label = "good_form"
            quality_score = 0
        else:  # wrong form
            quality_label = "major_issues"
            quality_score = 2

        return FormClassification(
            quality_label=quality_label,
            quality_score=quality_score,
            confidence=confidence,
            predicted_errors=[],
            feature_values={f: pose_angles.get(f, 0) for f in UNIFIED_FEATURES},
        )

    def _classify_synthetic(self, model, exercise: str, pose_angles: dict, pose_landmarks: dict) -> FormClassification:
        """Classify using synthetic-data model (exercise-specific features, 3-class)."""
        features = self._extract_features(exercise, pose_angles, pose_landmarks)
        feature_names = SQUAT_FEATURES if exercise == "squat" else PUSHUP_FEATURES
        feature_vector = np.array([[features.get(f, 0.0) for f in feature_names]])

        prediction = model.predict(feature_vector)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_vector)[0]
            confidence = float(probs[prediction])
        elif hasattr(model, "decision_function"):
            decisions = model.decision_function(feature_vector)[0]
            if decisions.ndim == 0:
                confidence = 0.8
            else:
                exp_d = np.exp(decisions - np.max(decisions))
                confidence = float((exp_d / exp_d.sum())[prediction])
        else:
            confidence = 0.8

        predicted_errors = self._predict_errors(exercise, features)

        return FormClassification(
            quality_label=QUALITY_LABELS.get(prediction, "unknown"),
            quality_score=int(prediction),
            confidence=confidence,
            predicted_errors=predicted_errors,
            feature_values=features,
        )

    def _extract_features(self, exercise: str, angles: dict, landmarks: dict) -> dict:
        """
        Extract classifier features from raw pose data.

        Transforms the PoseEstimator output into the feature format
        expected by the trained classifier.
        """
        features = {}

        if exercise == "squat":
            left_knee = angles.get("left_knee_angle", 170.0)
            right_knee = angles.get("right_knee_angle", 170.0)
            features["avg_knee_angle"] = (left_knee + right_knee) / 2
            features["left_knee_angle"] = left_knee
            features["right_knee_angle"] = right_knee
            features["torso_angle"] = angles.get("torso_angle", 0.0)
            features["left_hip_angle"] = angles.get("left_hip_angle", 170.0)
            features["right_hip_angle"] = angles.get("right_hip_angle", 170.0)
            features["knee_angle_diff"] = abs(left_knee - right_knee)

            # Knee-ankle offset (cave indicator)
            knee_offset = 0.0
            if "left_knee" in landmarks and "left_ankle" in landmarks:
                knee_offset = abs(landmarks["left_knee"].x - landmarks["left_ankle"].x)
            features["knee_ankle_x_offset"] = knee_offset

            # Shoulder-hip alignment
            alignment = angles.get("shoulder_hip_alignment", 0.0)
            features["shoulder_hip_alignment"] = alignment

        elif exercise == "pushup":
            left_elbow = angles.get("left_elbow_angle", 170.0)
            right_elbow = angles.get("right_elbow_angle", 170.0)
            features["avg_elbow_angle"] = (left_elbow + right_elbow) / 2
            features["left_elbow_angle"] = left_elbow
            features["right_elbow_angle"] = right_elbow

            # Hip deviation from shoulder-ankle line
            hip_dev = 0.0
            required = ["left_shoulder", "right_shoulder", "left_hip",
                        "right_hip", "left_ankle", "right_ankle"]
            if all(k in landmarks for k in required):
                mid_shoulder_y = (landmarks["left_shoulder"].y + landmarks["right_shoulder"].y) / 2
                mid_hip_y = (landmarks["left_hip"].y + landmarks["right_hip"].y) / 2
                mid_ankle_y = (landmarks["left_ankle"].y + landmarks["right_ankle"].y) / 2
                mid_shoulder_x = (landmarks["left_shoulder"].x + landmarks["right_shoulder"].x) / 2
                mid_hip_x = (landmarks["left_hip"].x + landmarks["right_hip"].x) / 2
                mid_ankle_x = (landmarks["left_ankle"].x + landmarks["right_ankle"].x) / 2

                if abs(mid_ankle_x - mid_shoulder_x) > 0.01:
                    t = (mid_hip_x - mid_shoulder_x) / (mid_ankle_x - mid_shoulder_x)
                    expected_hip_y = mid_shoulder_y + t * (mid_ankle_y - mid_shoulder_y)
                    hip_dev = mid_hip_y - expected_hip_y

            features["hip_deviation"] = hip_dev
            features["torso_angle"] = angles.get("torso_angle", 0.0)
            features["left_hip_angle"] = angles.get("left_hip_angle", 170.0)
            features["right_hip_angle"] = angles.get("right_hip_angle", 170.0)

            # Shoulder width ratio (elbow flare indicator)
            shoulder_ratio = 1.0
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks \
               and "left_elbow" in landmarks and "right_elbow" in landmarks:
                shoulder_w = abs(landmarks["left_shoulder"].x - landmarks["right_shoulder"].x)
                elbow_w = abs(landmarks["left_elbow"].x - landmarks["right_elbow"].x)
                if shoulder_w > 0.01:
                    shoulder_ratio = elbow_w / shoulder_w
            features["shoulder_width_ratio"] = shoulder_ratio

        return features

    def _predict_errors(self, exercise: str, features: dict) -> list[str]:
        """
        Predict specific error types from features.

        Uses lightweight threshold checks on the extracted features to
        identify which specific errors are likely present. This complements
        the overall quality classification with error-type granularity.
        """
        errors = []

        if exercise == "squat":
            if features.get("avg_knee_angle", 0) > 100:
                errors.append("insufficient_depth")
            if features.get("torso_angle", 0) > 45:
                errors.append("excessive_forward_lean")
            if features.get("knee_ankle_x_offset", 0) > 0.03:
                errors.append("knee_cave")
            if features.get("knee_angle_diff", 0) > 15:
                errors.append("asymmetric_depth")

        elif exercise == "pushup":
            if features.get("avg_elbow_angle", 0) > 110:
                errors.append("insufficient_depth")
            if features.get("hip_deviation", 0) > 0.05:
                errors.append("hip_sag")
            if features.get("hip_deviation", 0) < -0.05:
                errors.append("hip_pike")
            if features.get("shoulder_width_ratio", 1.0) > 1.3:
                errors.append("elbow_flare")

        return errors

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        info = {}
        for exercise, metadata in self.metadata.items():
            info[exercise] = {
                "model_type": metadata.get("best_model", "unknown"),
                "features": metadata.get("features", []),
                "training_samples": metadata.get("training_samples", 0),
                "accuracy": metadata.get("cross_validation_results", {})
                    .get(metadata.get("best_model", ""), {})
                    .get("mean_accuracy", 0.0),
            }
        return info
