"""
Pre-trained model wrappers for the AI Workout Coach.

This package contains wrapper classes for the pre-trained ML models used
in the orchestration pipeline.
"""
from .form_classifier import FormClassifier, FormClassification

__all__ = ["FormClassifier", "FormClassification"]
