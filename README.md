# AI Workout Coach

**Real-time exercise form analysis and coaching powered by orchestrated AI models**

## Project Overview

**Module**: CM3020 Artificial Intelligence (University of London)
**Template**: Orchestrating AI Models to Achieve a Goal (Project Idea 1, Section 4.1)

This system orchestrates **four pre-trained AI models** across three data domains to analyze exercise form from webcam video and provide real-time coaching feedback.

## AI Orchestration Pipeline

```
Camera Frame
     |
     v
[Stage 1: MediaPipe BlazePose] --- Computer Vision domain
     |    33 body landmarks + joint angles
     |
     +---> [Stage 2a: SVM Classifier] --- ML Classification domain
     |         form quality: good / minor / major issues
     |
     +---> [Stage 2b: Rule-Based Analysis]
     |         error detection + phase + rep counting
     |
     v
[Stage 3: Result Fusion] --- weighted consensus of ML + rules
     |
     +---> [Stage 4a: Template Feedback] --- immediate (<50ms)
     +---> [Stage 4b: GPT-4o-mini] --- NLP domain (async, 500-2000ms)
     +---> [Stage 4c: gTTS Voice] --- Audio Synthesis domain (async)
```

### The Four Models

| # | Model | Domain | Role |
|---|-------|--------|------|
| 1 | MediaPipe BlazePose | Computer Vision | Pose estimation (33 landmarks) |
| 2 | scikit-learn SVM | ML Classification | Form quality classification (90%+ accuracy) |
| 3 | GPT-4o-mini | NLP / Text Generation | Personalized coaching feedback |
| 4 | gTTS (Google WaveNet) | Audio Synthesis | Voice coaching (text-to-speech) |

## Quick Start

### Prerequisites
- Python 3.10+
- Webcam
- OpenAI API key (optional, for LLM feedback)

### Installation

```bash
cd ai-workout-coach
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration (optional)
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
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

46 tests covering all modules (pose estimation, form classification, exercise analyzers, feedback generation, TTS, orchestration, API).

### Training the Form Classifier

```bash
python -m tools.generate_test_data   # Generate synthetic training data
python -m tools.train_form_classifier # Train and evaluate classifiers
```

## Project Structure

```
ai-workout-coach/
├── backend/
│   ├── main.py              # FastAPI server, REST + WebSocket endpoints
│   ├── config.py            # Configuration management
│   ├── orchestrator.py      # AI model orchestration (core component)
│   ├── pose_estimator.py    # MediaPipe PoseLandmarker (Model 1)
│   ├── feedback_generator.py # GPT-4o-mini feedback (Model 3)
│   ├── tts_engine.py        # gTTS voice synthesis (Model 4)
│   ├── models/
│   │   ├── form_classifier.py    # scikit-learn SVM wrapper (Model 2)
│   │   └── pretrained/           # Serialized model artifacts (.joblib)
│   └── exercises/
│       ├── base.py          # Base classes and dataclasses
│       ├── squat.py         # Squat form analyzer
│       └── pushup.py        # Push-up form analyzer
├── frontend/
│   ├── index.html           # Main UI with skeleton overlay
│   ├── app.js               # Camera, WebSocket, TTS audio playback
│   └── styles.css           # Dark athletic theme
├── tests/                   # 46 unit + integration tests
├── tools/                   # Training scripts
├── docs/
│   └── model_exploration.md # Model evaluation and selection rationale
├── requirements.txt
├── pytest.ini
└── README.md
```

## Supported Exercises

### Squat
- Depth (knee angle), forward lean (torso angle), knee tracking (cave), symmetry

### Push-up
- Depth (elbow angle), hip alignment (sag/pike), elbow position (flare)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and model list |
| GET | `/api/health` | Health check |
| GET | `/api/exercises` | Supported exercises |
| GET | `/api/models` | Model pipeline status |
| GET | `/api/tts?text=...` | Text-to-speech audio |
| POST | `/api/analyze-frame` | Single frame analysis |
| WS | `/ws/workout` | Real-time workout session |

## Model Exploration

See [docs/model_exploration.md](docs/model_exploration.md) for the detailed evaluation of all models considered, including comparison metrics and selection rationale.

## License

This project is created for educational purposes as part of the University of London Computer Science degree program.
