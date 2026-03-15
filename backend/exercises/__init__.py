"""
Exercise-specific form analysis modules
"""
from .squat import SquatAnalyzer
from .pushup import PushupAnalyzer
from .lunge import LungeAnalyzer
from .plank import PlankAnalyzer
from .deadlift import DeadliftAnalyzer
from .bicep_curl import BicepCurlAnalyzer
from .shoulder_press import ShoulderPressAnalyzer
from .situp import SitupAnalyzer
from .base import ExerciseAnalyzer, FormError, ExercisePhase

__all__ = [
    "SquatAnalyzer",
    "PushupAnalyzer",
    "LungeAnalyzer",
    "PlankAnalyzer",
    "DeadliftAnalyzer",
    "BicepCurlAnalyzer",
    "ShoulderPressAnalyzer",
    "SitupAnalyzer",
    "ExerciseAnalyzer",
    "FormError",
    "ExercisePhase"
]


