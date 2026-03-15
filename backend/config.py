"""
Configuration management for AI Workout Coach
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Pose Estimation Settings (higher = fewer false detections but may miss poses)
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6

    # Form Analysis Thresholds
    squat_depth_threshold: float = 90.0  # degrees
    pushup_elbow_threshold: float = 90.0  # degrees

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


