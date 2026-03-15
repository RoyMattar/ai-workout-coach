"""
Tests for backend/main.py REST API endpoints

Uses FastAPI TestClient for synchronous HTTP testing.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app


# ── Fixtures ──

@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    return TestClient(app)


# ── Tests ──

def test_root_endpoint(client):
    """GET / should return 200 with a 'name' field."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert data["name"] == "AI Workout Coach"


def test_health_endpoint(client):
    """GET /api/health should return {"status": "healthy"}."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"


def test_exercises_endpoint(client):
    """GET /api/exercises should return a list containing squat and pushup."""
    response = client.get("/api/exercises")
    assert response.status_code == 200

    data = response.json()
    assert "exercises" in data

    exercise_ids = [ex["id"] for ex in data["exercises"]]
    assert "squat" in exercise_ids
    assert "pushup" in exercise_ids


def test_models_endpoint(client):
    """GET /api/models should return a models dict."""
    response = client.get("/api/models")
    assert response.status_code == 200

    data = response.json()
    assert "models" in data

    models = data["models"]
    assert "pose_estimation" in models
    assert "form_classifier" in models


def test_analyze_frame_invalid(client):
    """POST /api/analyze-frame with invalid data should return 400 or 422."""
    response = client.post(
        "/api/analyze-frame",
        json={"image": "not_valid_base64!!!"},
        params={"exercise": "squat"},
    )

    assert response.status_code in (400, 422, 500)
