"""
Base classes for exercise analysis
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from ..pose_estimator import PoseResult


class ExercisePhase(Enum):
    """Phases of an exercise repetition"""
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"
    # Push-up specific
    PLANK = "plank"
    LOWERING = "lowering"
    LOWEST = "lowest"
    PUSHING = "pushing"


class ErrorSeverity(Enum):
    """Severity levels for form errors (ordered by increasing severity)"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __lt__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        order = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other):
        return self == other or self.__lt__(other)

    def __gt__(self, other):
        if not isinstance(other, ErrorSeverity):
            return NotImplemented
        return not self.__le__(other)

    def __ge__(self, other):
        return self == other or self.__gt__(other)


@dataclass
class FormError:
    """Represents a detected form error"""
    error_type: str
    message: str
    severity: ErrorSeverity
    body_part: str
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "body_part": self.body_part,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "confidence": self.confidence
        }


@dataclass
class RepCounter:
    """Tracks repetition counting for exercises"""
    count: int = 0
    phase: ExercisePhase = ExercisePhase.STANDING
    phase_history: list = field(default_factory=list)
    last_bottom_time: float = 0.0
    
    def update_phase(self, new_phase: ExercisePhase, timestamp: float = 0.0):
        """Update phase and count reps on complete cycles"""
        if new_phase != self.phase:
            self.phase_history.append(self.phase)

            # Count rep when returning to standing from ascending or descending
            # (ascending→standing for squat/lunge/deadlift; descending→standing for bicep curl)
            if (self.phase in (ExercisePhase.ASCENDING, ExercisePhase.DESCENDING)
                    and new_phase == ExercisePhase.STANDING):
                # Only count if we went through BOTTOM at some point
                if ExercisePhase.BOTTOM in self.phase_history[-5:]:
                    self.count += 1

            # For push-ups: count when returning to plank from pushing
            if (self.phase == ExercisePhase.PUSHING and
                new_phase == ExercisePhase.PLANK):
                self.count += 1

            self.phase = new_phase


@dataclass
class AnalysisResult:
    """Result from exercise form analysis"""
    exercise_type: str
    phase: ExercisePhase
    errors: list[FormError]
    rep_count: int
    is_good_form: bool
    angles: dict[str, float]
    feedback_priority: list[str]  # Ordered list of error types to address
    
    def to_dict(self) -> dict:
        return {
            "exercise_type": self.exercise_type,
            "phase": self.phase.value,
            "errors": [e.to_dict() for e in self.errors],
            "rep_count": self.rep_count,
            "is_good_form": self.is_good_form,
            "angles": self.angles,
            "feedback_priority": self.feedback_priority
        }


class ExerciseAnalyzer(ABC):
    """
    Abstract base class for exercise-specific form analyzers.
    
    Each exercise type implements its own analyzer with specific
    form rules, phase detection, and error checking.
    """
    
    def __init__(self):
        self.rep_counter = RepCounter()
        self.error_history: list[list[FormError]] = []
    
    @property
    @abstractmethod
    def exercise_name(self) -> str:
        """Name of the exercise"""
        pass
    
    @abstractmethod
    def analyze(self, pose_result: PoseResult) -> AnalysisResult:
        """
        Analyze pose for form errors.
        
        Args:
            pose_result: Result from pose estimation
            
        Returns:
            AnalysisResult with detected errors and rep count
        """
        pass
    
    @abstractmethod
    def detect_phase(self, pose_result: PoseResult) -> ExercisePhase:
        """
        Detect current phase of the exercise.
        
        Args:
            pose_result: Result from pose estimation
            
        Returns:
            Current ExercisePhase
        """
        pass
    
    def reset(self):
        """Reset the analyzer state"""
        self.rep_counter = RepCounter()
        self.error_history = []
    
    def _prioritize_errors(self, errors: list[FormError]) -> list[str]:
        """
        Prioritize errors for feedback.
        
        Critical errors first, then by confidence.
        """
        severity_order = {
            ErrorSeverity.CRITICAL: 0,
            ErrorSeverity.ERROR: 1,
            ErrorSeverity.WARNING: 2,
            ErrorSeverity.INFO: 3
        }
        
        sorted_errors = sorted(
            errors,
            key=lambda e: (severity_order[e.severity], -e.confidence)
        )
        
        return [e.error_type for e in sorted_errors]


