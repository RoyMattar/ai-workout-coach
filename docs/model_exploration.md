# Model Exploration and Selection

This document records the process of evaluating and selecting pre-trained AI models for the AI Workout Coach orchestration pipeline. Each section describes the models considered, evaluation criteria, and rationale for the final selection.

## 1. Pose Estimation Models

**Purpose**: Extract body joint positions from webcam video frames.

### Models Evaluated

| Model | Landmarks | FPS (CPU) | Framework | Multi-person |
|-------|-----------|-----------|-----------|-------------|
| MediaPipe BlazePose | 33 | 30+ | MediaPipe Tasks | No (single) |
| MoveNet Thunder | 17 | 25+ | TensorFlow Hub | No (single) |
| MoveNet Lightning | 17 | 50+ | TensorFlow Hub | No (single) |
| OpenPose | 25 (body) | 8-15 | Caffe/OpenCV DNN | Yes |
| YOLOv8-Pose | 17 | 40+ | Ultralytics | Yes |

### Evaluation Criteria
- **Landmark count**: More landmarks enable more detailed form analysis
- **Real-time performance on CPU**: Must sustain 15+ FPS without GPU
- **Python API quality**: Clear documentation and easy integration
- **Accuracy for exercise analysis**: Reliable joint position detection during movement

### Testing Process
1. **MoveNet Thunder**: Tested via TensorFlow Hub. Only 17 keypoints — missing individual finger and face landmarks. Adequate for basic form checks but insufficient for detailed exercises needing wrist/elbow tracking precision. Slightly faster than BlazePose but fewer landmarks.
2. **MoveNet Lightning**: Faster variant, but lower accuracy. Unsuitable for form analysis where joint position precision matters.
3. **OpenPose**: Tested briefly. Requires GPU for real-time performance. Installation complexity (C++ dependencies) adds deployment friction. No clear accuracy benefit for single-person scenarios.
4. **YOLOv8-Pose**: Fast and accurate, but 17-keypoint output. Multi-person capability is unnecessary overhead for a personal coaching application.

### Selection: MediaPipe BlazePose
**Rationale**: MediaPipe provides the highest landmark count (33) with real-time CPU performance (30+ FPS). The normalized coordinate output and confidence scoring are ideal for consistent angle calculations across different camera setups. The Python Tasks API is well-documented and actively maintained. The 33-landmark output enables analysis of fine-grained movements that 17-keypoint models cannot capture.

---

## 2. Form Quality Classification Models

**Purpose**: Classify exercise form quality from pose features (angles, positions).

### Models Evaluated

| Model | Cross-Val Accuracy (Squat) | Cross-Val Accuracy (Push-up) | Inference Time |
|-------|---------------------------|------------------------------|---------------|
| K-Nearest Neighbors (k=5) | 88.87% | 98.53% | ~0.5ms |
| Support Vector Machine (RBF) | **90.03%** | **99.03%** | ~0.3ms |
| Random Forest (100 trees) | 90.00% | 99.00% | ~1.0ms |
| Gradient Boosting (100 estimators) | 89.00% | 98.97% | ~0.8ms |
| Logistic Regression | 74.2% | 91.5% | ~0.1ms |

### Evaluation Criteria
- **Cross-validation accuracy**: 5-fold stratified CV on 3,000 synthetic samples per exercise
- **Inference speed**: Must add negligible latency to the real-time pipeline
- **Interpretability**: Understanding which features drive predictions

### Training Data
Synthetic pose feature vectors generated from biomechanically-informed thresholds (see `tools/generate_test_data.py`). 3,000 samples per exercise with three classes: good_form (43%), minor_issues (33%), major_issues (23%). Gaussian noise (sigma=3 degrees) simulates real-world measurement variation.

### Testing Process
All classifiers were trained using scikit-learn pipelines with StandardScaler normalization. 5-fold stratified cross-validation ensured balanced evaluation. The training script (`tools/train_form_classifier.py`) generates comparison metrics and saves the best model.

### Selection: SVM with RBF Kernel
**Rationale**: The SVM (RBF kernel, C=10) achieved the highest cross-validation accuracy for both exercises (90.03% squat, 99.03% push-up). While Random Forest was within 0.03% accuracy, SVM's decision boundary is more theoretically suited to the feature space: pose angle measurements form continuous, overlapping distributions where kernel-based separation excels. Inference time (~0.3ms) adds negligible latency. The model is serialized via joblib (see `backend/models/pretrained/`).

---

## 3. Text-to-Speech Models

**Purpose**: Convert coaching text feedback to spoken audio for hands-free coaching.

### Models Evaluated

| Model | Quality | Latency | Size | Offline | API |
|-------|---------|---------|------|---------|-----|
| gTTS (Google TTS) | High (WaveNet) | 200-500ms | 0MB (API) | No | Simple |
| pyttsx3 | Medium | 50-100ms | 0MB (system) | Yes | Simple |
| Coqui TTS (XTTS) | Very High | 1-3s | 500MB+ | Yes | Complex |
| espeak-ng | Low (robotic) | <10ms | 5MB | Yes | Simple |

### Evaluation Criteria
- **Voice quality**: Must sound natural enough for coaching (not robotic)
- **Latency**: Feedback must arrive before the next exercise rep
- **Deployment simplicity**: Minimal dependencies and model downloads
- **Offline capability**: Desirable but not required

### Testing Process
1. **espeak-ng**: Tested as baseline. Voice quality too robotic — users in informal testing found it distracting rather than helpful.
2. **pyttsx3**: Uses system TTS engine (macOS Say, Windows SAPI5). Acceptable quality, works offline. Platform-dependent voice quality.
3. **Coqui TTS (XTTS)**: Highest quality voice synthesis. However, 500MB+ model download is impractical. CPU inference takes 1-3 seconds per utterance — too slow for real-time coaching.
4. **gTTS**: Uses Google's WaveNet neural TTS model via API. High quality, simple API, fast enough (200-500ms).

### Selection: gTTS (Primary) + pyttsx3 (Fallback)
**Rationale**: gTTS provides the best quality-to-latency ratio. Google's WaveNet model produces natural-sounding speech suitable for coaching cues. The 200-500ms latency is acceptable since TTS runs asynchronously (users receive visual feedback immediately, voice follows). pyttsx3 serves as an offline fallback when internet is unavailable. Audio is cached to avoid re-synthesizing identical feedback.

---

## 4. Language Models (NLP Feedback Generation)

**Purpose**: Generate personalized, context-aware coaching feedback from structured error descriptors.

### Models Evaluated

| Model | Quality | Latency | Cost | Local |
|-------|---------|---------|------|-------|
| GPT-4o-mini | Excellent | 500-2000ms | $0.15/M tokens | No |
| GPT-3.5-turbo | Good | 300-1000ms | $0.50/M tokens | No |
| Llama 2 7B (via llama.cpp) | Fair | 3-8s (CPU) | Free | Yes |
| Mistral 7B (via Ollama) | Good | 2-5s (CPU) | Free | Yes |

### Evaluation Criteria
- **Instruction following**: Must reliably produce structured output (SPOKEN/DETAILED/TIP/ENCOURAGEMENT format)
- **Coaching tone quality**: Encouraging, specific, and actionable
- **Latency**: Acceptable for async feedback (not blocking the real-time pipeline)
- **Cost**: Low per-query cost for sustained use

### Testing Process
1. **Llama 2 7B**: Tested via llama.cpp on CPU. Inference too slow (3-8 seconds per response). Inconsistent at following the structured output format. Coaching tone sometimes generic.
2. **Mistral 7B**: Better instruction following than Llama 2, but still 2-5 second latency on CPU. Would require GPU for real-time use.
3. **GPT-3.5-turbo**: Good quality and speed. Occasionally failed to maintain the SPOKEN/DETAILED/TIP/ENCOURAGEMENT format.
4. **GPT-4o-mini**: Best instruction following and coaching tone. Reliably produces structured output. Latency (500-2000ms) is acceptable for the async feedback path.

### Selection: GPT-4o-mini
**Rationale**: GPT-4o-mini offers the best balance of quality, reliability, and cost. It consistently follows the structured output format required for parsing into feedback components. Its coaching tone is encouraging, specific, and actionable — qualities that local models struggled to maintain. The 500-2000ms latency is managed through the async feedback architecture: users receive immediate template-based feedback while GPT-4o-mini generates richer coaching cues in the background. Template-based feedback serves as a fallback when the API is unavailable.

---

## Summary

| Pipeline Stage | Selected Model | Domain | Key Metric |
|---------------|---------------|--------|-----------|
| Pose Estimation | MediaPipe BlazePose | Computer Vision | 33 landmarks, 30+ FPS |
| Form Classification | SVM (RBF kernel) | ML Classification | 90-99% accuracy |
| NLP Feedback | GPT-4o-mini | Natural Language Processing | Structured output reliability |
| Voice Coaching | gTTS (Google WaveNet) | Audio Synthesis | 200-500ms latency |

The four models operate across three distinct data domains (vision, structured data, text, audio), demonstrating genuine AI orchestration where each model contributes a specialized capability that none could provide alone.
