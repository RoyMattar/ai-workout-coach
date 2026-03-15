"""
AI Workout Coach - FastAPI Backend Server

Provides REST API and WebSocket endpoints for real-time workout coaching.
Orchestrates four AI models: MediaPipe (pose), scikit-learn (classification),
GPT-4o-mini (NLP feedback), and gTTS (text-to-speech).
"""
import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from .orchestrator import WorkoutOrchestrator, ExerciseType
from .tts_engine import TTSEngine
from .coach_personas import list_personas, get_persona
from .database import Database, SessionRecord
from .workout_planner import WorkoutPlanner
from .auth import AuthManager
from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
orchestrators: dict[str, WorkoutOrchestrator] = {}
db = Database()
planner = WorkoutPlanner(db=db)
auth = AuthManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("AI Workout Coach starting up...")
    yield
    # Cleanup on shutdown
    for orchestrator in orchestrators.values():
        orchestrator.close()
    orchestrators.clear()
    logger.info("AI Workout Coach shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Workout Coach",
    description="Real-time exercise form analysis and coaching powered by orchestrated AI models",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ExerciseConfig(BaseModel):
    """Configuration for exercise session"""
    exercise_type: str = "squat"


class FrameData(BaseModel):
    """Single frame data for processing"""
    image: str  # Base64 encoded image


# ── REST API Endpoints ──

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "AI Workout Coach",
        "version": "2.0.0",
        "description": "Orchestrated AI pipeline for exercise form coaching",
        "models": [
            "MediaPipe BlazePose (pose estimation)",
            "scikit-learn SVM (form classification)",
            "GPT-4o-mini (NLP feedback)",
            "gTTS (text-to-speech)",
        ],
        "endpoints": {
            "websocket": "/ws/workout",
            "exercises": "/api/exercises",
            "health": "/api/health",
            "models": "/api/models",
            "tts": "/api/tts",
        },
    }


# ── Auth Endpoints ──

class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
async def register(req: AuthRequest):
    """Register a new user account."""
    try:
        user = auth.register(req.username, req.password)
        token = auth.login(req.username, req.password)
        return {"user": user.to_dict(), "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login(req: AuthRequest):
    """Login and receive a JWT token."""
    try:
        token = auth.login(req.username, req.password)
        user = auth.verify_token(token)
        return {"user": user.to_dict(), "token": token}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/api/auth/me")
async def get_current_user(request: Request):
    """Get current user profile from JWT token."""
    user = auth.get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user.to_dict()}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-workout-coach", "version": "2.0.0"}


@app.get("/api/exercises")
async def list_exercises():
    """List supported exercises"""
    return {
        "exercises": [
            {"id": "squat", "name": "Squat", "icon": "🦵",
             "description": "Bodyweight or barbell squat",
             "tracked_metrics": ["depth", "knee_tracking", "torso_angle", "symmetry"]},
            {"id": "pushup", "name": "Push-up", "icon": "💪",
             "description": "Standard push-up",
             "tracked_metrics": ["depth", "hip_alignment", "elbow_position"]},
            {"id": "lunge", "name": "Lunge", "icon": "🏃",
             "description": "Forward lunge",
             "tracked_metrics": ["depth", "knee_tracking", "torso_lean"]},
            {"id": "plank", "name": "Plank", "icon": "🧘",
             "description": "Isometric plank hold",
             "tracked_metrics": ["hip_alignment", "hold_duration"]},
            {"id": "deadlift", "name": "Deadlift", "icon": "🏋️",
             "description": "Hip hinge deadlift",
             "tracked_metrics": ["back_angle", "knee_bend", "lockout"]},
            {"id": "bicep_curl", "name": "Bicep Curl", "icon": "💪",
             "description": "Standing bicep curl",
             "tracked_metrics": ["range_of_motion", "elbow_stability", "swinging"]},
            {"id": "shoulder_press", "name": "Shoulder Press", "icon": "🙌",
             "description": "Overhead shoulder press",
             "tracked_metrics": ["lockout", "symmetry", "back_arch"]},
            {"id": "situp", "name": "Sit-up", "icon": "🔄",
             "description": "Standard sit-up / crunch",
             "tracked_metrics": ["range_of_motion", "neck_position"]},
        ]
    }


@app.get("/api/coaches")
async def list_coaches():
    """List all available coach personas."""
    return {"coaches": list_personas()}


@app.get("/api/models")
async def list_models():
    """List all AI models in the orchestration pipeline."""
    orchestrator = WorkoutOrchestrator()
    info = orchestrator.get_model_info()
    orchestrator.close()
    return {"models": info}


@app.get("/api/tts")
async def text_to_speech(text: str = Query(..., min_length=1, max_length=500)):
    """
    Text-to-Speech endpoint.

    Converts coaching text to spoken audio using gTTS (Google's WaveNet TTS model).
    Returns MP3 audio bytes.
    """
    tts = TTSEngine()
    audio = tts.synthesize(text)

    if audio is None:
        raise HTTPException(status_code=503, detail="TTS engine unavailable")

    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/analyze-frame")
async def analyze_frame(frame_data: FrameData, exercise: str = Query(default="squat")):
    """
    Analyze a single frame (REST API alternative to WebSocket).
    """
    try:
        image_bytes = base64.b64decode(frame_data.image.split(",")[-1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        exercise_type = ExerciseType(exercise)
        orchestrator = WorkoutOrchestrator(exercise_type=exercise_type)
        result = orchestrator.process_frame(frame)
        orchestrator.close()

        return result.to_dict()

    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid exercise type: {exercise}")
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session & Plan Endpoints ──

class SaveSessionRequest(BaseModel):
    exercise_type: str
    coach_persona: str = "coach_pro"
    start_time: float
    end_time: float
    total_reps: int = 0
    good_form_reps: int = 0
    form_score: float = 0.0
    errors: dict = {}
    pipeline_avg_latency_ms: float = 0.0


@app.post("/api/session/save")
async def save_session(req: SaveSessionRequest):
    """Save a completed workout session and check for achievements."""
    import json as _json
    session = SessionRecord(
        exercise_type=req.exercise_type,
        coach_persona=req.coach_persona,
        start_time=req.start_time,
        end_time=req.end_time,
        total_reps=req.total_reps,
        good_form_reps=req.good_form_reps,
        form_score=req.form_score,
        errors_json=_json.dumps(req.errors),
        pipeline_avg_latency_ms=req.pipeline_avg_latency_ms,
    )
    session_id = db.save_session(session)
    new_achievements = db.check_and_unlock_achievements(session)

    return {
        "session_id": session_id,
        "achievements_unlocked": new_achievements,
        "achievement_details": [
            db.ACHIEVEMENT_DEFS.get(a, {}) for a in new_achievements
        ],
    }


@app.get("/api/sessions")
async def get_sessions(limit: int = Query(default=20, le=100), offset: int = Query(default=0)):
    """Get workout session history."""
    return {
        "sessions": db.get_sessions(limit=limit, offset=offset),
        "total": db.get_session_count(),
    }


@app.get("/api/sessions/stats")
async def get_session_stats():
    """Get aggregate stats and trends."""
    return db.get_session_stats()


@app.get("/api/achievements")
async def get_achievements():
    """Get all unlocked achievements."""
    return {"achievements": db.get_achievements()}


@app.post("/api/workout-plan/generate")
async def generate_workout_plan(difficulty: str = Query(default="intermediate")):
    """Generate an AI-powered adaptive workout plan."""
    plan = await planner.generate_plan(difficulty=difficulty)
    if plan is None:
        raise HTTPException(status_code=500, detail="Failed to generate plan")
    return plan


@app.get("/api/workout-plan/current")
async def get_current_plan():
    """Get the current active workout plan."""
    plan = db.get_current_plan()
    if plan is None:
        return {"plan": None, "message": "No active plan. Generate one first."}
    return plan


# ── WebSocket endpoint for real-time processing ──

@app.websocket("/ws/workout")
async def websocket_workout(websocket: WebSocket):
    """
    WebSocket endpoint for real-time workout coaching.

    Protocol messages:
    Client → Server:
      {"type": "config", "exercise": "squat"}     — configure exercise
      {"type": "frame", "image": "base64..."}      — send frame for analysis
      {"type": "reset"}                            — reset session
      {"type": "summary"}                          — get session summary
      {"type": "ping"}                             — keep-alive

    Server → Client:
      {"type": "config_ack", ...}                  — configuration acknowledged
      {"type": "analysis", ...}                    — frame analysis result
      {"type": "llm_feedback", ...}                — async LLM-generated feedback
      {"type": "tts_audio", "audio": "base64..."}  — TTS audio for spoken feedback
      {"type": "reset_ack", ...}                   — reset confirmed
      {"type": "summary", ...}                     — session summary
      {"type": "pong"}                             — ping response
      {"type": "error", ...}                       — error notification
    """
    await websocket.accept()

    session_id = str(id(websocket))
    orchestrator: Optional[WorkoutOrchestrator] = None

    logger.info(f"New WebSocket connection: {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "")

            if msg_type == "config":
                exercise = message.get("exercise", "squat")
                persona_id = message.get("coach", "coach_pro")
                try:
                    exercise_type = ExerciseType(exercise)
                except ValueError:
                    exercise_type = ExerciseType.SQUAT

                if orchestrator:
                    orchestrator.close()

                orchestrator = WorkoutOrchestrator(
                    exercise_type=exercise_type,
                    persona_id=persona_id,
                )
                orchestrators[session_id] = orchestrator

                await websocket.send_json({
                    "type": "config_ack",
                    "exercise": exercise_type.value,
                    "coach": orchestrator.persona.to_dict(),
                    "status": "ready",
                    "models": orchestrator.get_model_info(),
                })

            elif msg_type == "frame":
                if not orchestrator:
                    orchestrator = WorkoutOrchestrator(exercise_type=ExerciseType.SQUAT)
                    orchestrators[session_id] = orchestrator

                try:
                    image_data = message.get("image", "")
                    if "," in image_data:
                        image_data = image_data.split(",")[1]

                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Run synchronous pipeline (Stages 1-4a)
                        result = orchestrator.process_frame(frame)

                        # Send immediate analysis result
                        await websocket.send_json({
                            "type": "analysis",
                            **result.to_dict(),
                        })

                        # Trigger async LLM feedback (Stage 4b) in background
                        if (result.analysis_result.errors
                                and orchestrator.feedback_generator.client):
                            asyncio.create_task(
                                _send_llm_feedback(websocket, orchestrator, result)
                            )

                        # Trigger async TTS audio (Stage 4c) in background
                        spoken = result.feedback.spoken_feedback
                        if spoken and orchestrator.tts_engine.is_available:
                            asyncio.create_task(
                                _send_tts_audio(websocket, orchestrator, spoken)
                            )

                        # Trigger async Vision analysis (GPT-4o) in background
                        if (orchestrator.feedback_generator.client
                                and result.pose_result.is_valid):
                            asyncio.create_task(
                                _send_vision_analysis(websocket, orchestrator, frame)
                            )
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Failed to decode image",
                        })

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Frame processing error: {str(e)}",
                    })

            elif msg_type == "reset":
                if orchestrator:
                    orchestrator.reset_session()
                await websocket.send_json({
                    "type": "reset_ack",
                    "status": "session_reset",
                })

            elif msg_type == "summary":
                if orchestrator:
                    summary = orchestrator.get_session_summary()
                    await websocket.send_json({"type": "summary", **summary})
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No active session",
                    })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if orchestrator:
            orchestrator.close()
        orchestrators.pop(session_id, None)


async def _send_llm_feedback(
    websocket: WebSocket,
    orchestrator: WorkoutOrchestrator,
    result,
):
    """Send LLM-generated feedback as a follow-up WebSocket message."""
    try:
        llm_feedback = await orchestrator.feedback_generator.generate_personalized_feedback(
            result.analysis_result
        )
        if not llm_feedback.is_cached:
            await websocket.send_json({
                "type": "llm_feedback",
                "spoken": llm_feedback.spoken_feedback,
                "detailed": llm_feedback.detailed_feedback,
                "tip": llm_feedback.tip,
                "encouragement": llm_feedback.encouragement,
            })
    except Exception as e:
        logger.debug(f"LLM feedback task error: {e}")


async def _send_tts_audio(
    websocket: WebSocket,
    orchestrator: WorkoutOrchestrator,
    text: str,
):
    """Send TTS audio as a follow-up WebSocket message."""
    try:
        audio_b64 = await orchestrator.generate_tts_audio(text)
        if audio_b64:
            await websocket.send_json({
                "type": "tts_audio",
                "audio": audio_b64,
                "text": text,
            })
    except Exception as e:
        logger.debug(f"TTS audio task error: {e}")


async def _send_vision_analysis(
    websocket: WebSocket,
    orchestrator: WorkoutOrchestrator,
    frame,
):
    """Send GPT-4o Vision form analysis as a follow-up WebSocket message."""
    try:
        vision_result = await orchestrator.analyze_frame_with_vision(frame)
        if vision_result:
            await websocket.send_json({
                "type": "vision_analysis",
                **vision_result,
            })
    except Exception as e:
        logger.debug(f"Vision analysis task error: {e}")


# Serve frontend static files
import os
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/app", response_class=HTMLResponse)
    async def serve_app():
        """Serve the frontend application"""
        with open(os.path.join(frontend_path, "index.html")) as f:
            return f.read()


def start_server():
    """Start the server"""
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start_server()
