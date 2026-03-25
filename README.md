# AI Workout Coach

**Real-time exercise form analysis and coaching powered by six orchestrated AI models**

## Project Overview

**Module**: CM3020 Artificial Intelligence (University of London)
**Template**: Orchestrating AI Models to Achieve a Goal (Project Idea 1, Section 4.1)

This system orchestrates **six AI models** across four data domains to analyze exercise form in real-time from a webcam and provide personalized coaching feedback with voice guidance.

## Hybrid Architecture

The system uses a **hybrid client-server architecture**: MediaPipe runs in the browser via WebAssembly/WebGL for instant skeleton rendering (60+ FPS), while ML classification, form analysis, and AI feedback generation run on the server.

```
Browser (Client)                          Server (Python/FastAPI)
┌──────────────────────┐                  ┌─────────────────────────────────────┐
│ Webcam Capture        │                  │                                     │
│       │               │                  │  [Stage 2a: KNN Classifier]         │
│       v               │   angles+        │      form quality (97.4% accuracy)  │
│ [MediaPipe BlazePose] │───landmarks──>   │                                     │
│   WASM/WebGL (9-17ms) │   (~200 bytes)   │  [Stage 2b: Rule-Based Analysis]    │
│       │               │                  │      error detection + phase + reps │
│       v               │                  │           │                         │
│ Skeleton Overlay      │                  │           v                         │
│ (zero-latency draw)   │   <──results──   │  [Stage 3: Result Fusion]           │
│                       │                  │       │                             │
│ Rep Counter           │                  │       ├─> [GPT-4o-mini] NLP text    │
│ Phase Indicator       │                  │       ├─> [GPT-4o Vision] analysis  │
│ Form Score            │                  │       └─> [OpenAI TTS] voice        │
└──────────────────────┘                  └─────────────────────────────────────┘
```

### Six AI Models

| # | Model | Domain | Role |
|---|-------|--------|------|
| 1 | MediaPipe BlazePose | Computer Vision | Pose estimation (33 landmarks), runs client-side via WASM |
| 2 | KNN Classifier | ML Classification | Form quality classification, trained on real video data (97.4% accuracy) |
| 3 | GPT-4o-mini | NLP / Text Generation | Personalized coaching feedback |
| 4 | GPT-4o Vision | Visual Understanding | Independent form assessment from video frames |
| 5 | OpenAI TTS | Audio Synthesis | Voice coaching with persona-specific voices |
| 6 | Exercise Recognizer | ML Classification | Automatic exercise type detection (99.6% accuracy) |

### Seven-Approach Model Comparison

The form classification model was selected through systematic evaluation of seven approaches on 1,483 real push-up video frames:

| Approach | Accuracy | Type |
|----------|----------|------|
| KNN (k=5) | **97.4%** | Classical ML |
| SVM (RBF) | 97.0% | Classical ML |
| Random Forest | 95.9% | Classical ML |
| MLP (128-64-32) | 95.8% | Neural Network |
| MLP (64-32) | 91.2% | Neural Network |
| GPT-4o Vision | 60.0% | LLM Vision |
| Rule-based thresholds | 51.7% | Heuristic |

## Features

- **8 exercises**: Squat, Push-up, Lunge, Plank, Deadlift, Bicep Curl, Shoulder Press, Sit-up
- **5 coach personas**: Coach Pro, Drill Sergeant, Zen Master, Hype Beast, Pop Diva — each with distinct voice (OpenAI TTS), personality, and catchphrases
- **AI workout planner**: GPT-4o-mini generates adaptive plans based on session history
- **Real-time skeleton overlay**: MediaPipe runs in-browser at 60+ FPS with smooth interpolation
- **Session tracking**: SQLite database stores workouts, form scores, and achievements
- **User authentication**: JWT-based login/registration
- **Transparent AI pipeline**: Live panel shows which models are active, their predictions, and agreement status

## Quick Start

### Prerequisites
- Python 3.10+
- Webcam
- OpenAI API key (for LLM feedback, TTS voices, and vision analysis)

### Installation

```bash
cd ai-workout-coach
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
JWT_SECRET_KEY=your_secret_key_here
```

### Running

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/app` in your browser.

### Running Tests

```bash
python -m pytest tests/ -v
```

123 tests covering all modules (pose estimation, form classification, exercise analyzers, feedback generation, TTS, orchestration, API, auth, database, coach personas).

### Docker Deployment

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8000/app`. Configure your `.env` file before building.

### Evaluation Datasets

The evaluation scripts expect video datasets in `data/evaluation/`. These are not included in the repo due to file size. Download from Kaggle:

1. [LSTM Push-Up Videos](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup) → `data/evaluation/LSTM Exercise Classification - Push Up Videos/`
2. [Workout/Exercises Video](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video) → `data/evaluation/Workout:Exercises Video/`
3. [Multi-Class Exercise Poses](https://www.kaggle.com/datasets/dp5995/exercise-poses) → `data/kaggle/dataset_all_points.csv`

Then run:
```bash
python -m tools.evaluate_videos      # Evaluate pipeline on real video data
python -m tools.compare_form_models  # Compare 7 model architectures
python -m tools.train_all_realdata   # Train KNN classifiers on real video features
```

## Project Structure

```
ai-workout-coach/
├── backend/
│   ├── main.py                # FastAPI server, REST + WebSocket endpoints
│   ├── config.py              # Configuration management
│   ├── orchestrator.py        # AI model orchestration (core component)
│   ├── pose_estimator.py      # MediaPipe PoseLandmarker wrapper
│   ├── feedback_generator.py  # GPT-4o-mini + GPT-4o Vision feedback
│   ├── tts_engine.py          # OpenAI TTS / gTTS voice synthesis
│   ├── coach_personas.py      # 5 coaching personalities with voice mapping
│   ├── workout_planner.py     # AI-adaptive workout plan generator
│   ├── database.py            # SQLite session/plan/achievement storage
│   ├── auth.py                # JWT authentication
│   ├── models/
│   │   ├── form_classifier.py     # KNN/SVM classifier wrapper
│   │   └── pretrained/            # Serialized model artifacts (.joblib)
│   └── exercises/
│       ├── base.py            # Base classes and dataclasses
│       ├── squat.py           # Squat form analyzer
│       ├── pushup.py          # Push-up form analyzer
│       ├── bicep_curl.py      # Bicep curl form analyzer
│       ├── shoulder_press.py  # Shoulder press form analyzer
│       ├── deadlift.py        # Deadlift form analyzer
│       ├── lunge.py           # Lunge form analyzer
│       ├── plank.py           # Plank form analyzer
│       └── situp.py           # Sit-up form analyzer
├── frontend/
│   ├── index.html             # Main UI
│   ├── app.js                 # Client-side MediaPipe, WebSocket, skeleton rendering
│   └── styles.css             # Dark athletic theme
├── tests/                     # 123 unit + integration tests
├── tools/                     # Training, evaluation, and comparison scripts
├── data/
│   ├── kaggle/                # Kaggle landmark dataset
│   └── evaluation/            # Video datasets (gitignored, download from Kaggle)
├── docs/
│   ├── model_exploration.md   # Model evaluation and selection rationale
│   └── evaluation_results/    # JSON evaluation reports
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── README.md
```

## Supported Exercises

| Exercise | Key Metrics | Visibility Requirement |
|----------|------------|----------------------|
| Squat | Depth, forward lean, knee tracking, symmetry | Full body |
| Push-up | Depth, hip alignment, elbow position | Side view preferred |
| Bicep Curl | Range of motion, elbow drift, body swing | Upper body |
| Shoulder Press | Lockout, back arch, symmetry | Upper body |
| Deadlift | Back rounding, lockout, knee position | Full body |
| Lunge | Depth, knee tracking, torso lean | Full body |
| Plank | Hip alignment, head position | Side view preferred |
| Sit-up | Range of motion, neck strain | Side view preferred |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and model list |
| GET | `/api/health` | Health check |
| GET | `/api/exercises` | Supported exercises |
| GET | `/api/models` | Model pipeline status |
| POST | `/api/analyze-frame` | Single frame analysis |
| POST | `/api/auth/register` | User registration |
| POST | `/api/auth/login` | User login |
| GET | `/api/sessions` | Session history |
| GET | `/api/achievements` | User achievements |
| GET | `/api/workout-plan` | Current workout plan |
| POST | `/api/workout-plan/generate` | Generate AI workout plan |
| WS | `/ws/workout` | Real-time workout session |

## Model Exploration

See [docs/model_exploration.md](docs/model_exploration.md) for the detailed evaluation of all models considered, including comparison metrics and selection rationale.

## License

This project is created for educational purposes as part of the University of London Computer Science degree program.
