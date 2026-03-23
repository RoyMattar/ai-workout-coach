# AI-Driven Workout Form Coach
## Orchestrating Artificial Intelligence for Exercise Technique Analysis

**Final Project Report**

University of London
Module: CM3020 Artificial Intelligence
Project Template: Orchestrating AI Models to Achieve a Goal (Section 4.1)

Code Repository: https://github.com/RoyMattar/ai-workout-coach

---

# 1. Introduction

## 1.1 Project Template and Concept

This project follows the "Orchestrating AI models to achieve a goal" template (Section 4.1) from the CM3020 Artificial Intelligence module. The system orchestrates six pre-trained artificial intelligence models, operating across four distinct data domains — computer vision, machine learning classification, natural language processing, and audio synthesis — to provide real-time exercise form coaching through a webcam. Rather than relying on a single end-to-end model, the system demonstrates how coordinating specialised models within a structured pipeline produces outcomes that no individual model could achieve in isolation: interpreting human movement, classifying form quality using trained classifiers, generating natural language coaching feedback, providing independent visual form assessment, and speaking coaching cues aloud with selectable voice personas.

The orchestration is the central contribution. Each model addresses a different aspect of the coaching problem — perceiving the body, understanding whether the movement is correct, explaining what to fix, and verbalising the correction — and the pipeline coordinates their outputs into a unified, real-time coaching experience.

## 1.2 Motivation and Domain Analysis

Strength training and bodyweight exercises are practiced globally, yet incorrect exercise form remains a leading cause of musculoskeletal injuries among recreational athletes (Aasa et al., 2017). The problem is compounded by significant access barriers: personal training sessions cost $50–150 per hour, making professional coaching inaccessible to most individuals. Existing fitness applications predominantly focus on repetition counting and workout logging rather than movement quality analysis, leaving users without feedback on the aspect most critical to injury prevention — their technique.

The motivation for this project stemmed from a clear gap in accessible fitness technology: no freely available system combined real-time pose analysis with natural language coaching and spoken feedback using only a standard webcam. Commercial solutions such as Tempo and Tonal require specialised hardware costing upwards of $2,000, while software-based alternatives like Gymscore and Ray AI operate as proprietary closed systems without transparency into their AI pipelines. The target users — individuals exercising at home without professional supervision, beginners learning correct technique, and recreational athletes seeking form feedback — required a solution that was accessible, educational, and immediate.

## 1.3 Competitive Landscape

An analysis of existing AI fitness coaching systems informed the project's design and identified opportunities for differentiation.

**Gymscore** provides computer-vision-based form scoring across five dimensions (bracing, posture, foot placement, range of motion, movement efficiency) with a 0–100 scoring system. While comprehensive, it operates as a proprietary mobile application with no transparency into its AI pipeline or model architecture. **Ray AI** offers voice-guided live coaching through headphones, using computer vision for rep counting and real-time adaptation. Its strength lies in the voice interaction modality, though it employs a single-model approach without visual form analysis breakdowns. **Fitbod** uses data-driven workout programming aggregated from millions of anonymised performance patterns, excelling at adaptive plan generation but providing no real-time form analysis. **Tempo** and **Tonal** represent the high-end market with depth-sensor-equipped hardware ($2,000+) delivering precise 3D tracking, but their cost and hardware requirements limit accessibility.

*Table 1: Competitive landscape comparison matrix*

| Feature | Gymscore | Ray AI | Fitbod | Tempo | This Project |
|---------|----------|--------|--------|-------|-------------|
| Real-time form analysis | Yes | Limited | No | Yes | Yes |
| Multiple AI models | Unknown | No | No | Unknown | 6 models, 4 domains |
| Voice coaching | No | Yes | No | Yes | Yes (5 personas) |
| Transparent AI pipeline | No | No | No | No | **Yes** |
| Webcam-only (no hardware) | Yes | Yes | N/A | No | **Yes** |
| Adaptive workout plans | No | Yes | Yes | Yes | Yes |
| Cost | Subscription | Subscription | Subscription | $2,000+ | **Free** |

The differentiator of the present system is transparent multi-model orchestration visible to the user, combined with webcam-only accessibility and selectable coaching personalities — capabilities not found in any competitor analysed.

## 1.4 Project Evolution and Contributions

The project underwent significant iterative development following the midterm submission, which received a mark of 53%. Analysis of this result identified critical gaps relative to the template requirements: the midterm prototype employed only one genuine pre-trained model (MediaPipe BlazePose), used hardcoded rule-based thresholds rather than trained classifiers, contained no software tests, and lacked evaluation on real exercise data.

Post-midterm development proceeded through five focused sprints. Sprint 1 added a scikit-learn SVM form classifier and Google Text-to-Speech, bringing the model count to three. Sprint 2 integrated GPT-4o Vision for independent visual form assessment and replaced gTTS with the OpenAI TTS API for higher-quality voice synthesis with persona variety, reaching five models. Sprint 3 incorporated real Kaggle exercise video data, implemented coach personas, the adaptive workout planner, and persistent session tracking. Sprint 4 conducted a systematic comparison of seven model architectures — KNN, SVM, Random Forest, two MLP neural networks, GPT-4o Vision, and rule-based thresholds — on the same real video dataset, leading to evidence-based selection of KNN as the primary form classifier. Sprint 5 replaced the rule-based form analysis with the ML-primary approach, trained models for all exercises on real video data, and evaluated the complete pipeline on over 200 real exercise videos.

The final system orchestrates six AI models across four data domains, supports eight exercises, offers five coach personas with distinct voices, includes an AI-powered adaptive workout planner, and is backed by 123 automated tests. The form classification accuracy reached 97.4% on real video data — a dramatic improvement from the rule-based approach's 51.7%.

---

# 2. Literature Review

## 2.1 Human Pose Estimation: Foundation for Movement Analysis

Human pose estimation is a well-established computer vision domain concerned with identifying key joint locations from images or video. The field has evolved from handcrafted feature approaches to modern deep learning techniques enabling real-time analysis on consumer hardware.

OpenPose (Cao et al., 2017) represented a landmark contribution as the first real-time multi-person pose estimation system, detecting body, hand, and facial keypoints using Part Affinity Fields. On the COCO keypoint benchmark, OpenPose achieved approximately 61.8% mAP for multi-person detection (Cao et al., 2017). However, OpenPose requires GPU acceleration, producing inference times of approximately 50 ms per frame on a mid-range GPU. MediaPipe Pose (Lugaresi et al., 2019) advanced the field by optimising for edge deployment: BlazePose estimates 33 body landmarks while maintaining 30+ FPS on laptop CPUs. Bazarevsky et al. (2020) reported upper-body landmark localisation error below 4 pixels on a 1080p frame, making BlazePose competitive with heavier architectures at a fraction of the computational cost. MoveNet (Google, 2021) introduced further efficiency improvements optimised for fitness applications, achieving latencies below 10 ms on mobile devices, though its 17-keypoint output provides fewer landmarks than MediaPipe's 33.

The trade-off between landmark count and inference speed was a central design consideration. MoveNet's 17 keypoints suffice for gross body tracking but lack the granularity for fine-grained form assessment — measuring torso rotation or hip alignment requires the additional landmarks MediaPipe provides. Conversely, architectures with higher landmark counts, such as COCO WholeBody's 133 keypoints (Jin et al., 2020), introduce computational overhead that jeopardises real-time operation without meaningfully improving the specific joint angles relevant to push-up analysis.

During the model exploration phase, all three frameworks were evaluated. MediaPipe was selected for its superior landmark count (33 versus 17 for MoveNet), real-time CPU performance, and the quality of its Python Tasks API. During development, MediaPipe transitioned from its legacy Solutions API to the newer Tasks API, which introduced an asynchronous callback-based architecture that improved frame processing consistency but required restructuring the detection pipeline to handle results via registered listeners rather than synchronous return values. OpenPose was rejected due to its GPU requirement, and MoveNet due to insufficient landmark coverage for detailed form analysis.

A further consideration was per-landmark confidence scores. MediaPipe assigns visibility and presence scores to each landmark, enabling assessment of individual joint position reliability. Landmarks with low confidence (when a limb was occluded or outside the frame) could propagate erroneous angle calculations through the pipeline. The system addressed this by treating low-confidence landmarks as missing data, preventing unreliable readings from triggering false coaching cues. Despite their maturity, these pose estimation frameworks remain low-level perception tools that do not encode domain knowledge about biomechanics or movement quality, creating an opportunity for systems that build upon pose estimation with higher-level reasoning.

## 2.2 Exercise Form Assessment Approaches

Several approaches to exercise form assessment have been explored in the literature, ranging from rigid rule-based systems to data-driven machine learning classifiers. These approaches span a spectrum from fully supervised methods, which require expert-labelled datasets, to unsupervised techniques such as anomaly detection, which model correct form as a distribution and flag deviations without explicit error labels (Pang et al., 2021).

The Pose Trainer project (Chen and Shen, 2020) proposed a rule-based system comparing joint angles against predefined thresholds to identify squat form errors. While interpretable and computationally efficient, threshold-based approaches suffer from limited generalisation: natural variability in body proportions, flexibility, and movement styles means fixed thresholds produce both false positives and false negatives. This limitation was confirmed empirically in the present project, where rule-based thresholds achieved only 51.7% accuracy on real push-up video data — a true positive rate of 98.6% but a catastrophic false negative rate of only 11%, meaning the system almost never detected incorrect form.

AI-Coach (Fieraru et al., 2021) advanced form assessment by training neural networks on paired correct and incorrect exercise videos. This approach learns to discriminate form quality without explicit rule specification, achieving higher accuracy but requiring substantial labelled training data and providing limited explainability. Research on datasets such as MEx (Khurana et al., 2018) demonstrated that temporal pose features can distinguish correct from incorrect executions, though most systems trained on such data focus on classification accuracy rather than user-facing explanations.

A persistent challenge in exercise form assessment is the "cold start" problem: obtaining labelled training data requires expert annotation by biomechanics specialists or certified coaches, which is expensive and difficult to scale (Liao et al., 2020). Unlike image classification where crowdsourced labelling is feasible, determining whether a push-up exhibits correct form demands domain expertise — untrained annotators frequently disagree on borderline cases. The present project addressed this through a knowledge distillation approach: rule-based heuristics informed by biomechanical guidelines automatically labelled pose data from real exercise videos, and these labels trained machine learning classifiers. This strategy relates to semi-supervised learning literature, where a weaker model generates pseudo-labels that bootstrap a stronger learner (Lee, 2013). The rule-based system served as a "noisy teacher" whose labels, despite imperfect accuracy, encoded sufficient signal for downstream classifiers to learn generalisable decision boundaries.

Google's ML Kit documentation recommends K-Nearest Neighbours (KNN) for pose classification due to its simplicity and effectiveness with small training datasets (Google Developers, 2024). This recommendation aligned with the findings of the present project: KNN achieved 97.4% accuracy on 1,483 real video frames, outperforming SVM (97.0%), Random Forest (95.9%), and two-layer and three-layer MLP neural networks (91.2% and 95.8% respectively). The neural network underperformance is consistent with established machine learning principles: networks typically require substantially larger datasets to demonstrate advantages over simpler classifiers (Pedregosa et al., 2011).

A critical gap emerged from this body of work: the disconnect between biomechanical detection and meaningful coaching. Systems can identify that form is incorrect without explaining what is wrong, why it matters, or how to correct it.

## 2.3 AI-Based Fitness Coaching Systems

Commercial and experimental AI fitness applications have proliferated, though their capabilities vary significantly. As detailed in Section 1.3, Gymscore, Ray AI, Fitbod, and Tempo/Tonal each address individual aspects of fitness coaching. Technically, these systems diverge in their approaches. Tempo and Tonal employ depth sensors combined with proprietary pose models to track form within controlled equipment ecosystems, offering high accuracy at the cost of hardware dependency (Tempo, 2023). Gymscore operates on-device using lightweight pose estimation and rule-based thresholds, prioritising privacy and low latency but sacrificing the nuance of learned classifiers. Ray AI represents an emerging trend toward voice-first coaching, where the primary modality is conversational audio rather than visual overlays, reducing the need for users to look at a screen during exercise.

The trend toward voice-first interaction is supported by motor learning research. Vickers (2007) demonstrated that auditory instructions are processed more efficiently during physical exertion than visual instructions, because athletes can attend to spoken cues without diverting visual attention. Clark et al. (2005) found that concurrent verbal feedback during balance tasks improved performance compared to visual-only feedback. These findings suggest that voice-based coaching is pedagogically superior for real-time movement correction.

Academic work has investigated more intelligent coaching approaches: Chen et al. (2020) explored adaptive feedback systems that modify coaching messages based on user performance trends, finding that contextual and personalised feedback improves adherence and learning outcomes compared to static instructions. This supports the value of natural language generation in coaching systems.

However, most existing systems tightly couple perception and feedback within monolithic architectures, limiting transparency and flexibility. Few explicitly decompose the problem into independently evaluable layers (perception, analysis, explanation, voice output) that can be orchestrated, compared, and improved individually. No system identified in this review orchestrates six or more models across four data domains while providing transparent pipeline visibility to the end user.

## 2.4 Large Language Models and Visual AI for Coaching

Recent advances in large language models enable generation of context-aware explanations from structured inputs. Research in explainable AI (XAI) demonstrates that users are more likely to trust and follow system recommendations when explanations are provided in natural language rather than abstract metrics (Ribeiro et al., 2016). GPT-4 and similar models have shown remarkable capability in generating coherent, contextually appropriate text from structured prompts, making them ideal "explanation layers" that translate technical biomechanical findings into accessible coaching advice.

A significant challenge when integrating LLMs into real-time pipelines is prompt engineering for structured output. The system required coaching feedback in a consistent, parseable format — a form verdict, error descriptions, and corrective cues — rather than free-form prose. Research on constrained generation demonstrated that structured prompts with explicit output schemas and few-shot examples substantially improve format compliance (Wei et al., 2022). However, LLM outputs exhibited latency variance from 500 ms to over 3 seconds depending on server load, making them unsuitable for frame-by-frame feedback but viable for asynchronous generation. Output format consistency also remained imperfect: approximately 5-10% of responses required post-processing, motivating the design decision to use template-based feedback for time-critical cues and reserve LLM generation for richer asynchronous summaries.

GPT-4o Vision represents a particularly interesting capability: assessing exercise form directly from images, bypassing skeleton-based analysis entirely. This fundamentally different approach was evaluated in this project. GPT-4o Vision achieved only 60% accuracy on binary push-up form classification, significantly underperforming domain-specific classifiers. This underperformance is attributable to several factors. General-purpose vision-language models are optimised for open-ended visual understanding — describing scenes and reasoning about relationships — rather than precise binary classification against domain-specific criteria (OpenAI, 2024). The model's training data likely contained limited examples labelled with the specific biomechanical criteria used here. Furthermore, the binary task conflated the model's tendency toward hedging with the need for a definitive verdict, a mismatch between generative strengths and discriminative requirements.

Feedback timing research indicates that immediate concurrent feedback during movement can interfere with motor learning, while summary feedback after movement completion may be more effective for long-term retention (Swinnen, 1996). This informed the system's dual-feedback architecture: immediate template-based feedback during exercise, complemented by richer LLM-generated summaries delivered asynchronously.

## 2.5 Motor Learning and Multimodal Feedback

Research in sports science provides empirical support for external feedback in skill acquisition. Schmidt and Lee (2011) established that augmented feedback accelerates learning when it highlights key performance variables without overwhelming the learner. The distinction between Knowledge of Results (KR, movement outcomes) and Knowledge of Performance (KP, movement patterns) is critical: for technique improvement, KP feedback addressing how movements are performed proves more effective than KR feedback about whether goals were achieved (Magill and Anderson, 2017).

However, the guidance hypothesis (Salmoni et al., 1984) cautions that excessive augmented feedback can become detrimental. When feedback is provided on every trial, learners develop dependency on external information rather than intrinsic error-detection capabilities. Winstein and Schmidt (1990) demonstrated that reducing feedback frequency — providing corrections on only 50% of trials — led to superior retention compared to 100% feedback conditions. This has direct implications for real-time coaching: continuous corrections risk cognitive overload and dependency. The present project addressed this through a rate-limiting mechanism enforcing a two-second cooldown between coaching cues, ensuring users had time to process and attempt corrections before receiving additional feedback. This design choice was grounded in the motor learning literature rather than being purely a technical constraint.

Gentile's Two-Stage Model of motor learning (Gentile, 1972) further suggests that feedback should adapt to the learner's stage. In the initial "getting the idea of the movement" stage, learners benefit from explicit, frequent guidance. In the later "fixation/diversification" stage, they need less external feedback and more intrinsic error detection. This implies that an ideal coaching system would reduce feedback frequency as the user demonstrates improving form — a feature the present system does not implement, though the session history data needed to detect improvement already exists. The coach personas partially address this distinction: the Drill Sergeant's explicit commands suit initial learning, while the Zen Master's process-oriented cues align with later-stage refinement.

Studies comparing visual-only versus verbal feedback indicate that verbal cues are more effective for correcting complex movements, especially for novice athletes who may not recognise visual deviations in their own performance (Hodges and Franks, 2002). This finding directly motivated the system's multimodal approach: visual skeleton overlays provide KP feedback about body positioning, while spoken coaching cues deliver verbal corrections through neural text-to-speech synthesis.

The OpenAI TTS API provides six distinct neural voices, enabling coach personas where each personality combines a distinct voice with tailored verbal style. Research on embodied conversational agents demonstrated that perceived personality and voice characteristics significantly influence user engagement and trust (Nass and Brave, 2005). Variety in voice options increases the likelihood that users find a persona they respond to positively. While not formally validated in this project, the persona concept addresses the personalisation dimension identified by Chen et al. (2020) as beneficial for adherence.

## 2.6 Model Orchestration Patterns

The coordination of multiple AI models within a single system is an active area of research, with established patterns including sequential pipelines, parallel ensembles, and voting-based fusion (Dietterich, 2000). In a sequential pipeline, each model's output feeds the next stage — pose estimation produces landmarks consumed by a classifier, whose verdict is consumed by a language model. This pattern offers simplicity but introduces single points of failure: an error in pose estimation propagates uncorrected through the chain.

Ensemble methods employ multiple models in parallel to produce independent assessments that are fused into a unified prediction. Breiman (2001) demonstrated that combining diverse models reduces variance and improves generalisation, particularly when individual models make uncorrelated errors. Voting-based ensembles, where the prediction is determined by majority agreement, provide an intuitive confidence signal: unanimous agreement indicates high confidence, while disagreement signals uncertainty (Polikar, 2006).

The present project employed a hybrid of pipeline and ensemble patterns. Pose estimation and feature extraction followed a sequential pipeline, but form assessment was performed in parallel by three independent approaches — a trained KNN classifier, a rule-based engine, and GPT-4o Vision — whose outputs were fused through a model agreement mechanism. When all three agreed, the system treated the assessment as high-confidence; disagreement flagged uncertainty. This parallel analysis with fusion is conceptually equivalent to a heterogeneous ensemble, where diversity arises not from data sampling (as in bagging) but from fundamentally different paradigms: statistical pattern matching, deterministic rules, and visual language understanding. The degree of model agreement served as an emergent confidence signal requiring no additional calibration.

## 2.7 Summary and Identified Gaps

The literature reveals mature pose estimation technology that produces raw perception data, form assessment approaches that are either fragile (rule-based) or data-hungry (neural networks), and coaching systems that operate as monolithic proprietary applications. Three specific gaps motivated this project: first, no existing system orchestrates six or more AI models across multiple data domains with transparent pipeline visibility; second, no publicly available comparative evaluation exists of multiple model architectures (KNN, SVM, MLP, GPT-4o Vision, rule-based) on the same real exercise video dataset; and third, no accessible system combines real-time pose-based classification with natural language coaching and spoken voice feedback using only a consumer webcam. This project addressed all three gaps.

---

# 3. Design

## 3.1 System Architecture

The system follows a multi-stage pipeline architecture orchestrating six AI models across four data domains. The design philosophy prioritises transparent orchestration: rather than hiding the AI behind a simple interface, the system exposes pipeline state, model agreement, and processing latency to the user.

The pipeline architecture was chosen over two principal alternatives after careful consideration. A monolithic architecture — where a single end-to-end model would ingest video and produce coaching output directly — was rejected because no such model existed that could simultaneously perceive poses, classify form quality, generate natural language explanations, and synthesise speech. Training such a model would have required a dataset of unprecedented scale and diversity, far beyond the scope of this project. A microservices architecture — where each model operated as an independent network service communicating via HTTP — was considered but rejected due to the latency overhead of inter-service communication; each network hop would have added 5–20ms of serialisation and transport latency, jeopardising the real-time feedback requirement. The in-process pipeline pattern adopted instead allowed all models to communicate through shared Python objects with negligible overhead, while still maintaining clean separation of concerns through well-defined interfaces between stages.

A further architectural consideration was the distinction between synchronous and asynchronous processing within the pipeline. Template feedback generation was implemented as a synchronous operation because its sub-millisecond latency made it suitable for the critical path — every frame could receive immediate coaching cues without perceptible delay. In contrast, LLM feedback generation and TTS audio synthesis were implemented as asynchronous tasks dispatched via `asyncio.create_task()`. This design pattern was essential because these operations involved external API calls with latencies ranging from 500ms to 3 seconds; executing them synchronously would have blocked frame processing and reduced throughput to below 1 FPS. The asynchronous design meant that a frame could be analysed, receive immediate template feedback, and simultaneously trigger background LLM and TTS requests whose results would be delivered to the client as follow-up WebSocket messages when available. This progressive response delivery model ensured that the user experienced no perceptible lag while still receiving the richest possible coaching.

The real-time communication layer employed a hybrid client-server architecture over WebSocket. Initially, the client streamed base64-encoded JPEG frames to the server for pose estimation, but this introduced 50–100ms round-trip latency causing visible skeleton lag. The final architecture moved MediaPipe pose estimation to the browser, where it ran via WASM/WebGL at 60+ FPS with zero network latency for skeleton rendering. The client computed eight joint angles locally and transmitted them as compact JSON (~200 bytes) at 12 messages per second. The server received pre-computed angles and returned structured JSON messages with a `type` field discriminating between form classifications, coaching feedback (template text, LLM-generated text), vision analysis results, and audio data (base64-encoded MP3).

**Stage 1 — Pose Estimation (Computer Vision domain, client-side):** MediaPipe PoseLandmarker ran in the browser via the `@mediapipe/tasks-vision` WASM module with GPU delegate (WebGL), extracting 33 body landmarks and drawing the skeleton overlay locally at 60+ FPS. Joint angles for knees, hips, elbows, torso inclination, and shoulder-hip alignment were computed client-side before transmission.

**Stage 2a — ML Form Classification (Classification domain):** A KNN classifier trained on 1,483 real video frames predicts form quality (correct or incorrect). This model achieves 97.4% accuracy with inference time under 0.01ms per frame. Real-data models are available for five exercises (push-up, squat, deadlift, bicep curl, shoulder press), with knowledge-distilled models for remaining exercises.

**Stage 2b — Rule-Based Analysis (complementary):** Exercise-specific analysers implement biomechanical rules for phase detection (standing, descending, bottom, ascending), repetition counting, and specific error identification (e.g., knee cave, hip sag, elbow flare). Rules complement ML classification by providing granular error descriptions that the classifier does not produce.

**Stage 3 — Result Fusion:** The orchestrator combines ML classification with rule-based analysis, tracking model agreement and disagreement. When both methods concur, confidence is high; disagreement flags uncertainty and is logged for analysis.

**Stage 4a — Template Feedback (immediate, <1ms):** Pre-written coaching cues mapped to specific error types provide instant feedback without network latency.

**Stage 4b — LLM Feedback (NLP domain, async, 500–2000ms):** GPT-4o-mini generates personalised coaching from structured error descriptors, adapting tone and vocabulary to the selected coach persona.

**Stage 4c — Vision Analysis (Visual Understanding, async, ~1.5s):** GPT-4o Vision independently assesses form quality directly from the video frame — a fundamentally different approach from skeleton-based analysis. This provides a "second opinion" that operates on raw visual information rather than extracted features.

**Stage 4d — Voice Coaching (Audio Synthesis domain, async):** OpenAI TTS converts coaching text to spoken audio using the persona's assigned neural voice, delivered as base64-encoded MP3 via WebSocket.

*Figure 1: System architecture diagram showing all six models, data flow between stages, and latency annotations.*

## 3.2 Design Decisions and Justification

Every architectural decision was justified through empirical evaluation.

**Pose estimation model selection and deployment:** MediaPipe was selected over OpenPose (GPU-dependent) and MoveNet (only 17 keypoints, insufficient for torso and shoulder-hip analysis). MediaPipe's 33 landmarks and well-documented Tasks API made it the clear choice. Initially deployed server-side, testing revealed that streaming frames to the server and returning landmarks introduced 50–100ms round-trip latency, causing visible skeleton lag. The architecture was revised to run MediaPipe client-side in the browser via `@mediapipe/tasks-vision` (WASM/WebGL), using the lighter `pose_landmarker_lite` model (~4MB). This eliminated network latency entirely: skeleton overlay rendered locally at 60+ FPS, and only computed joint angles (~200 bytes) were sent to the server 12 times per second for ML classification and coaching.

**Form classification approach:** The decision to use KNN as the primary form classifier was the result of systematic evaluation of seven approaches on the same dataset of 1,483 real video frames. KNN achieved 97.4% accuracy with 0.005ms inference time, marginally outperforming SVM (97.0%) while being simpler and more interpretable. Neural network architectures (MLP with 64-32 and 128-64-32 hidden layers) achieved 91.2% and 95.8% respectively, confirming that the dataset size was insufficient for deep learning advantages to manifest. Rule-based thresholds achieved only 51.7%, and GPT-4o Vision scored 60%. The selection of KNN was therefore evidence-based, not assumed. A decision tree classifier was also briefly considered for its superior interpretability — producing explicit if-then rules — but was rejected because its axis-aligned decision boundaries are poorly suited to the angular feature space where form quality boundaries are oblique.

*Table 2: Model comparison informing the design decision*

| Model | Accuracy | Latency | Cost |
|-------|----------|---------|------|
| KNN (k=5) | **97.4%** | 0.005ms | Free |
| SVM (RBF) | 97.0% | 0.007ms | Free |
| Random Forest | 95.9% | 0.023ms | Free |
| MLP (128-64-32) | 95.8% | 0.001ms | Free |
| MLP (64-32) | 91.2% | 0.001ms | Free |
| Rule-based | 51.7% | 0.01ms | Free |
| GPT-4o Vision | 60.0% | 1,540ms | $0.01/frame |

**Real data over synthetic:** The initial SVM trained on synthetic data achieved 90.0% accuracy; retraining on real video features improved this to 97.4% — a 7.4 percentage point gain. The fundamental issue was distributional mismatch: synthetic angle distributions did not reflect the correlations between joint angles in real human movement.

**ML-primary over rule-based:** The 45.7 percentage point accuracy gap (97.4% vs 51.7%) provided overwhelming evidence for making ML classification the primary form quality signal. Rule-based analysis was retained for specific error identification and phase tracking, since the ML classifier produces only a binary prediction without identifying which errors are present.

**Asynchronous architecture:** Template feedback arrives in under 1ms, while LLM-generated feedback takes 500–2000ms and TTS audio requires 1–3 seconds. The async design ensures the user receives immediate visual feedback while richer coaching arrives progressively — a deliberate design choice informed by motor learning research on feedback timing (Swinnen, 1996). The alternative of waiting for all models to complete before delivering any feedback was rejected because it would have introduced perceptible 2–3 second delays, breaking the real-time interaction model and contradicting motor learning evidence that concurrent feedback during movement is most effective when delivered immediately (Clark et al., 2005).

**Coach personas:** Five coaching personalities (Coach Pro, Drill Sergeant, Zen Master, Hype Beast, Pop Diva) each modify the LLM system prompt and select a distinct OpenAI TTS voice. This addresses the personalisation dimension identified by Chen et al. (2020) as beneficial for engagement, supported by research showing that voice characteristics influence user trust (Nass and Brave, 2005).

**Testing strategy:** Pytest was selected for its concise assertion syntax, powerful fixture system, and automatic test discovery. The test suite of 123 automated tests across 10 files covers every module boundary and critical code path. Fixtures in `conftest.py` represent realistic pose data — standing, squat bottom, sitting, and push-up configurations — enabling repeatable verification without webcam access.

## 3.3 Data Strategy

The data strategy evolved through three phases, each addressing limitations of the previous approach.

**Phase 1 — Synthetic data generation:** An initial training dataset of 3,000 samples per exercise was generated programmatically from biomechanical thresholds with added Gaussian noise (σ = 3°). This approach enabled rapid prototyping: an SVM classifier achieved 90.0% cross-validation accuracy on this synthetic data. However, synthetic data cannot capture the full variability of real human movement, camera angles, lighting conditions, and body types.

**Phase 2 — Kaggle landmark data:** The Multi-Class Exercise Poses dataset (dp5995, Kaggle) provided 2,700 samples of pre-extracted MediaPipe 33-landmark coordinates across seven exercise classes. This real data trained an exercise recognition model achieving 99.6% accuracy, demonstrating that exercise type classification is well-suited to landmark-based features. However, this dataset labels exercise types, not form quality.

**Phase 3 — Real video data:** The LSTM Exercise Classification Push-Up Videos dataset (mohamadashrafsalama, Kaggle) provided 100 labelled videos (50 correct form, 50 incorrect form). Features were extracted by processing each video through MediaPipe, yielding 1,483 labelled frames. Additionally, 200+ unlabelled exercise videos from the Workout/Exercises Video dataset (hasyimabdillah, Kaggle) were processed and labelled using the tuned rule-based system as an oracle — a knowledge distillation approach where the rules serve as a teacher to generate training labels for the ML student model. This enabled KNN classifiers to be trained for five exercises.

## 3.4 User Interaction Design

The user interface was designed around three principles: transparency, accessibility, and engagement.

**Transparency** is achieved through the AI Pipeline panel, which displays real-time information about each model's status, the ML classifier's prediction and confidence, model agreement between ML and rule-based analysis, and per-frame pipeline latency. This design decision makes the orchestration visible and educational, differentiating the system from competitor applications that treat their AI as a black box.

**Accessibility** requires only a standard webcam and web browser. The system provides a skeleton overlay rendered on the video canvas, exercise-specific pose validation gates that provide repositioning guidance rather than errors, and session persistence through SQLite with JWT-based authentication (bcrypt password hashing, 72-hour token expiry). A "Continue as Guest" option bypasses authentication to minimise friction for first-time users.

**Engagement** is supported through five coach personas with distinct personalities and voices, an achievement system with eight milestone badges, an AI-powered adaptive workout planner that generates plans based on session history using GPT-4o-mini, and a session summary screen displaying a five-dimension form radar breakdown.

The session summary screen displays a five-dimension form radar chart (depth, alignment, stability, control, range of motion) rendered using HTML5 Canvas, alongside animated statistics counters and per-exercise form score trends enabling users to observe technique changes over the session.

The achievement system provides eight milestone badges — such as "First Rep," "Perfect Form," and "Form Master" — grounded in self-determination theory (Ryan and Deci, 2000), rewarding both effort and quality through animated toast notifications during the session.

## 3.5 Technology Stack

*Table 3: Technology stack with justifications*

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Pose estimation | MediaPipe PoseLandmarker (client-side WASM/WebGL) | 33 landmarks, 60+ FPS in browser |
| Form classification | scikit-learn KNN | 97.4% accuracy, <0.01ms inference |
| NLP feedback | OpenAI GPT-4o-mini | Structured output, persona adaptation |
| Visual analysis | OpenAI GPT-4o Vision | Independent form assessment from images |
| Voice synthesis | OpenAI TTS API | 6 neural voices, persona variety |
| Backend | FastAPI (Python) | Async native, WebSocket support |
| Frontend | Vanilla HTML/CSS/JS | No build step, immediate deployment |
| Database | SQLite | Zero-dependency, file-based persistence |
| Authentication | JWT + bcrypt | Stateless auth, secure password storage |
| Testing | pytest | 123 automated tests |
| Deployment | Docker + docker-compose | Reproducible containerised deployment |

---

# 4. Implementation

## 4.1 Pose Estimation Pipeline

The pose estimation pipeline employed a hybrid architecture. MediaPipe PoseLandmarker ran client-side in the browser via the `@mediapipe/tasks-vision` WASM module, using the `pose_landmarker_lite` model (~4MB) with GPU delegate (WebGL) and CPU fallback. The `detectForVideo()` method ran synchronously within a `requestAnimationFrame` loop, achieving 60+ FPS. The detector produced up to 33 body landmarks with per-landmark visibility and presence confidence scores, and the skeleton overlay was drawn immediately from raw landmarks with zero network latency.

From the 33 detected landmarks, 13 key points are extracted: nose, left/right shoulders, elbows, wrists, hips, knees, and ankles. Joint angles are calculated using three-point vector geometry. Given three landmarks forming a joint (A-B-C), the angle at vertex B is computed as:

```
angle = arccos( dot(BA, BC) / (|BA| × |BC|) )
```

where BA and BC are the vectors from the vertex to the adjacent points. This produces angles in degrees (0–180) for knee flexion, hip flexion, elbow flexion, torso inclination relative to vertical, and shoulder-hip lateral alignment.

Each exercise implements a `_is_exercise_pose()` validation gate verifying the detected pose matches the expected exercise position. For squats, the gate requires shoulders above hips above knees; for push-ups, it verifies roughly horizontal body orientation with visible wrists; for bicep curls, it requires only upper body visibility. This per-exercise validation dramatically reduced false positives in varied environments, preventing the system from analysing non-exercise poses (e.g., walking or sitting) as exercise repetitions.

*Figure 2: Skeleton overlay rendered on the video canvas during a squat exercise, showing joint connections colour-coded by form quality.*

## 4.2 Form Classification: The Model Exploration Journey

The form classification component progressed from hardcoded rules to data-driven machine learning through four phases, constituting the core evidence of iterative model exploration required by the project template. Each phase represented a distinct approach, and the progression demonstrated systematic, evidence-driven decision-making.

**Phase 1 — Rule-based thresholds (midterm).** The initial approach implemented exercise-specific threshold checks based on biomechanical guidelines (e.g., knee angle >110° flagged insufficient squat depth; elbow angle >160° flagged insufficient push-up depth). While interpretable, this approach proved fragile on real data: testing on 1,483 frames from 50 labelled push-up videos revealed 98.6% true positive rate but only 11.0% true negative rate, producing 51.7% overall accuracy. The thresholds were too permissive for real-world movement variability — the system essentially defaulted to "good form" for almost any input.

**Phase 2 — SVM on synthetic data.** A synthetic dataset of 3,000 samples per exercise was generated by sampling angle values from Gaussian distributions centred on biomechanically correct ranges (σ = 3°), with incorrect samples shifted 10–30° outside correct ranges. Four classifiers were evaluated via 5-fold stratified cross-validation: KNN (88.9%), SVM (90.0%), Random Forest (90.0%), and Gradient Boosting (89.0%). The SVM was deployed based on marginally superior performance. However, synthetic data could not capture real human movement variability — joint angle correlations, the continuous spectrum of form quality, and noise from camera angles and clothing.

**Phase 3 — Real video data and comprehensive comparison.** The Kaggle LSTM Push-Up Videos dataset provided ground truth: 50 correct-form and 50 incorrect-form videos. Each video was processed through MediaPipe, extracting 8 angle features (elbow flexion, shoulder angles, hip angle, torso inclination, knee angles) from every 3rd frame to reduce temporal redundancy. This yielded 1,483 labelled samples. Seven model architectures were evaluated on this identical dataset:

*Table 4: Comprehensive model comparison on 1,483 real push-up video frames*

| Model | Accuracy | Std Dev | Latency/Frame | Cost |
|-------|----------|---------|---------------|------|
| KNN (k=5) | **97.4%** | ±1.0% | 0.005ms | Free |
| SVM (RBF) | 97.0% | ±1.6% | 0.007ms | Free |
| Random Forest | 95.9% | ±1.6% | 0.023ms | Free |
| MLP (128-64-32) | 95.8% | ±1.2% | 0.001ms | Free |
| MLP (64-32) | 91.2% | ±2.9% | 0.001ms | Free |
| Rule-based thresholds | 51.7% | N/A | 0.01ms | Free |
| GPT-4o Vision | 60.0% | N/A | 1,540ms | $0.01/frame |

KNN achieved the highest accuracy (97.4%) with near-zero inference latency. The detailed classification report showed 99% precision and recall for both classes on a held-out test set. This result was perhaps the most significant finding of the project: the simplest classical ML algorithm outperformed both deep learning approaches and the most sophisticated general-purpose AI model available.

Two findings were particularly noteworthy. First, neural networks (91.2% and 95.8%) did not outperform simpler classifiers, consistent with the principle that neural networks require larger datasets — 1,483 frames was insufficient. Second, GPT-4o Vision achieved only 60% accuracy despite being the most powerful model tested. Analysis revealed a striking asymmetry: it identified 9/10 correct-form videos but only 3/10 incorrect-form videos. The model frequently hedged rather than committing to "incorrect" verdicts, consistent with its training objective of nuanced responses conflicting with binary classification. It lacked calibration for specific biomechanical criteria, relying on gross visual assessment that missed subtle indicators of poor technique.

**Phase 4 — Deployment across exercises.** KNN classifiers were trained on real video data for five exercises using features extracted from 200+ Kaggle workout videos. For exercises without labelled correct/wrong data, a knowledge distillation approach was used: the tuned rule-based system labelled video frames, and the KNN was trained on these labels. This produced models with accuracy ranging from 86.8% (squat) to 99.4% (bicep curl).

This four-phase journey demonstrated empirical model selection: rule-based approaches were inadequate (51.7%), synthetic classifiers hit a ceiling (90.0%), real data unlocked 97.4%, and general-purpose models proved inferior (60.0%). Each transition was motivated by measured limitations of the preceding approach.

*Figure 3: Model accuracy comparison bar chart showing all seven approaches evaluated.*

## 4.3 Orchestrator and Result Fusion

The `WorkoutOrchestrator` class serves as the central coordinator, managing data flow between all six models and maintaining session state. The `process_frame()` method executes the pipeline in five timed stages, with the ML classification serving as the primary form quality signal.

The result fusion mechanism compares the ML classifier's binary prediction (good/bad form) against the rule-based analyser's error-based assessment. When both methods agree on form quality, the model agreement counter increments; disagreement is logged and displayed to the user. Over a session, the agreement rate provides a meta-confidence metric — a unique transparency feature not found in competitor systems.

Feedback generation employs rate limiting (2-second cooldown) to prevent overwhelming the user with rapid-fire corrections. Template-based feedback is generated synchronously on every qualifying frame, while GPT-4o-mini and GPT-4o Vision are triggered asynchronously using `asyncio.create_task()`, with results delivered as follow-up WebSocket messages when available.

## 4.4 LLM Integration and Coach Personas

The feedback generator supports two LLM modes. GPT-4o-mini receives structured error descriptors and generates coaching feedback formatted as SPOKEN/DETAILED/TIP/ENCOURAGEMENT fields, parsed into a `FeedbackResult` dataclass. GPT-4o Vision receives actual JPEG-encoded video frames alongside angle measurements and produces independent form assessments with observations and suggestions.

Five coach personas modify the LLM system prompt to produce distinct coaching styles. The Drill Sergeant uses military language and short commands ("Move it, soldier!"); the Zen Master speaks calmly with nature metaphors ("Breathe into the movement..."); the Hype Beast uses high-energy Gen-Z slang ("LET'S GOOO!"); the Pop Diva delivers sassy, dramatic feedback ("Sweetie, no. Just... no. Try again."); and Coach Pro maintains a professional, balanced tone. Each persona maps to a distinct OpenAI TTS voice (onyx, shimmer, echo, fable, nova respectively) and injects persona-specific catchphrases into encouragement messages.

## 4.5 Voice Synthesis

The OpenAI TTS API provides six neural voices with natural prosody and expression. The TTS engine supports voice selection per request, speed adjustment (0.25x to 4.0x), and in-memory caching keyed by (text_hash, voice_id) to avoid re-synthesising identical feedback. Audio is generated in a thread pool executor to avoid blocking the async event loop, encoded as base64 MP3, and transmitted via WebSocket. The frontend decodes and plays the audio with a speaking-state guard that prevents overlapping playback.

## 4.6 Workout Planner and Persistence

Session data persists in an SQLite database with tables for workout sessions, AI-generated plans, achievements, and users. The workout planner uses GPT-4o-mini to generate structured JSON plans from session history, adapting recommendations based on form score trends and common errors. JWT-based authentication (bcrypt password hashing, 72-hour token expiry) enables multi-user support with isolated data.

## 4.7 Error Handling and Robustness

The system implemented defensive strategies across four failure categories. **Lost or dropped frames** were handled by maintaining the last valid analysis result, preventing visual flickering during the 2–5% of frames where detection failed. **Network connectivity issues** were addressed through WebSocket automatic reconnection with exponential backoff (1s to 30s maximum) and timeout-wrapped API calls ensuring slow responses did not block the pipeline. **Missing or low-confidence landmarks** (confidence below 0.5) were treated as missing data, with frames skipped entirely when more than three of eight angle features were unavailable. **External API failures** triggered a template fallback mechanism — when LLM generation failed, the system fell back to template-based feedback requiring no network access, ensuring uninterrupted coaching.

## 4.8 Testing

The test suite comprises 123 automated tests across 10 test files, covering unit tests (pose estimation, classifier prediction, exercise analysers, feedback parsing, TTS, authentication, database), integration tests (orchestrator pipeline, FastAPI endpoints), and behavioural tests (all eight exercises verified for initialisation, invalid pose handling, and pose validation gating).

Test fixtures in `conftest.py` encode specific joint coordinates producing known angle values — standing, squat bottom, sitting, and push-up positions — making assertions deterministic without requiring video input. Representative test cases include `test_squat_not_in_position_sitting` (verifying the pose validation gate rejects non-exercise poses), `test_pushup_correct_form_angles` (verifying correct classification of proper form), and `test_feedback_rate_limiting` (confirming the 2-second cooldown suppresses rapid-fire corrections).

The 10 test files mirror the module structure: `test_pose_estimation.py`, `test_classifier.py`, `test_exercises.py`, `test_feedback.py`, `test_tts.py`, `test_auth.py`, `test_database.py`, `test_orchestrator.py`, `test_api.py`, and `test_planner.py`.

*Figure 4: Test output showing 123 passing tests.*

---

# 5. Evaluation

## 5.1 Evaluation Methodology

The evaluation employed three complementary approaches. **Machine learning cross-validation** used 5-fold stratified cross-validation to measure classifier generalisation. **Real video evaluation** processed 200+ exercise videos from three Kaggle datasets through the complete pipeline, measuring pose detection rate, form score distribution, and classification accuracy. **Model comparison** evaluated seven approaches on identical data (1,483 frames from 50 labelled push-up videos), ensuring fair comparison.

**Datasets used:**
- Kaggle Multi-Class Exercise Poses (dp5995): 2,700 landmark samples across 7 exercise classes
- LSTM Push-Up Videos (mohamadashrafsalama): 100 videos (50 correct, 50 wrong form)
- Workout/Exercises Video (hasyimabdillah): 200+ videos across 22 exercise types

## 5.2 Model Comparison Results

The systematic comparison of seven model architectures constituted the most significant evaluation contribution. All models were evaluated on identical data: 1,483 angle feature frames extracted from the labelled push-up video dataset.

*Table 5: Complete model comparison results*

| Model | CV Accuracy | Std Dev | Inference Latency | Annual Cost |
|-------|-------------|---------|-------------------|-------------|
| KNN (k=5) | **97.4%** | ±1.0% | 0.005ms | Free |
| SVM (RBF, C=10) | 97.0% | ±1.6% | 0.007ms | Free |
| Random Forest (100 trees) | 95.9% | ±1.6% | 0.023ms | Free |
| MLP (128-64-32, ReLU) | 95.8% | ±1.2% | 0.001ms | Free |
| MLP (64-32, ReLU) | 91.2% | ±2.9% | 0.001ms | Free |
| Rule-based thresholds | 51.7% | N/A | 0.010ms | Free |
| GPT-4o Vision | 60.0% | N/A | 1,540ms | ~$150 (est.) |

KNN's optimality was attributable to the small, well-structured feature space (8 angles) and sufficient data for distance-based classification but insufficient for neural networks. The KNN-SVM gap (0.4pp) was not statistically significant; KNN was selected for simplicity. Both MLP architectures underperformed due to the limited 1,483-sample training set (Pedregosa et al., 2011).

The rule-based system's 51.7% accuracy masked a dangerous asymmetry: 98.6% true positive rate but only 11.0% recall on incorrect form, effectively never detecting form errors. GPT-4o Vision (60%) correctly identified 9/10 correct-form but only 3/10 incorrect-form videos, demonstrating that general-purpose vision models lack domain-specific calibration for binary form classification.

*Figure 5: Bar chart comparing accuracy across all seven model approaches.*

## 5.3 Real Video Evaluation

The pipeline was evaluated on real exercise videos across six exercise types from the Kaggle Workout/Exercises Video dataset.

*Table 6: Per-exercise video evaluation results*

| Exercise | Videos Tested | Avg Form Score | Pose Detection Rate |
|----------|--------------|----------------|-------------------|
| Push-up (correct form) | 20 | 98.8% | 100.0% |
| Push-up (wrong form) | 20 | 84.4% | 99.9% |
| Hammer curl | 12 | 97.5% | 86.2% |
| Shoulder press | 13 | 93.7% | 98.6% |
| Push-up (general) | 15 | 90.2% | 93.2% |
| Romanian deadlift | 10 | 92.6% | 93.1% |
| Plank | 6 | 82.4% | 99.3% |
| Squat | 15 | 69.0% | 91.5% |
| Deadlift | 15 | 59.8% | 97.2% |

For push-ups with labelled ground truth, optimal classification threshold analysis was performed:

*Table 7: Push-up classification accuracy at different form score thresholds*

| Threshold | Correct Identified | Wrong Identified | Overall Accuracy |
|-----------|-------------------|-----------------|-----------------|
| 85% | 20/20 (100%) | 8/20 (40%) | 70.0% |
| 90% | 19/20 (95%) | 10/20 (50%) | 72.5% |
| 95% | 18/20 (90%) | 12/20 (60%) | **75.0%** |

The optimal threshold of 95% achieved 75% end-to-end accuracy. The 14.3 percentage point score separation between correct (98.8%) and wrong (84.4%) form videos confirms the system can meaningfully distinguish form quality, though the gap is insufficient for confident production deployment.

**Threshold tuning evolution** demonstrated iterative, evaluation-driven refinement. The initial push-up evaluation produced only 10.8% form score on correct-form videos — elbow flare detection fired on 89.2% of frames because the x-axis comparison was unreliable from side-view camera angles. Adding a shoulder-width guard (skipping the check when shoulder width <0.08 in normalised coordinates, indicating side-on view) and relaxing the flare threshold from 0.3 to 0.6 times shoulder width improved scores from 10.8% to 98.8%.

Similar tuning was applied to hip sag detection (threshold increased from 0.05 to 0.06 for push-ups and 0.09 for planks, improving plank accuracy from 14.7% to 82.4%) and knee cave analysis (adding a side-view guard improved squat accuracy from 41.7% to 69.0%).

**Webcam angle compression.** Real user testing revealed that front-facing webcam angles were significantly compressed compared to actual joint angles due to perspective projection. A full bicep curl producing an actual elbow angle of ~40° appeared as ~100° from the front camera. Phase detection thresholds had to be iteratively calibrated through live testing sessions: bicep curl `STANDING_THRESHOLD` was adjusted from 150° to 130° and `BOTTOM` from 60° to 108°; shoulder press `TOP` from 160° to 140°. This "user-in-the-loop" calibration highlighted that theoretical biomechanical thresholds are insufficient — deployment requires empirical tuning against actual camera perspectives, and future work would benefit from automatic per-user threshold calibration.

Accuracy variation across exercises was attributable to three factors: (1) camera angle — side-filmed exercises (push-ups, planks) produced more consistent results than variably-angled exercises (squats, deadlifts); (2) body visibility — close-up framing in curl videos occluded the lower body, reducing detection rates; and (3) movement complexity — push-ups involve constrained single-plane motion, while squats require three-joint flexion with 3D mechanics difficult to capture from 2D video.

## 5.4 Pipeline Performance

*Table 8: Pipeline latency breakdown (hybrid architecture)*

| Stage | Location | Latency |
|-------|----------|---------|
| Pose estimation | Client (browser WASM/WebGL) | 9–17ms per frame |
| Skeleton rendering | Client | ~0ms (same frame) |
| ML classification | Server | <0.01ms |
| Rule-based analysis | Server | <0.01ms |
| Result fusion + feedback | Server | <0.01ms |
| **Server total** | Server | **<1ms per angle message** |

Moving pose estimation client-side was the single largest UX improvement. The previous server-side architecture introduced 50–100ms perceived latency (frame encoding, network round-trip, and rendering delay); the hybrid architecture reduced perceived latency to ~10ms (measured via `performance.now()`). MediaPipe achieved 81–100% pose detection rates across exercise types.

## 5.5 User Experience Evaluation

### 5.5.1 Heuristic Evaluation

A heuristic evaluation against Nielsen's ten usability heuristics (Nielsen, 1994) rated six heuristics **strong**, three **moderate**, and one **weak** (help and documentation — no onboarding tutorial was implemented).

*Table 9: Heuristic evaluation summary*

| Heuristic | Rating | Key Evidence |
|-----------|--------|-------------|
| H1 Visibility of system status | Strong | AI Pipeline panel, skeleton overlay, connection indicator |
| H2 Match with real world | Strong | Natural language phases, body-mapped skeleton |
| H3 User control and freedom | Strong | Start/stop/reset, mid-session switching |
| H4 Consistency and standards | Strong | Uniform theme, consistent colour coding |
| H5 Error prevention | Moderate | Pose validation gate, confidence filtering |
| H6 Recognition over recall | Strong | Visual grids for exercises and coaches |
| H7 Flexibility and efficiency | Moderate | Multi-level workflows, no keyboard shortcuts |
| H8 Aesthetic and minimalist design | Strong | Clean dark theme, information hierarchy |
| H9 Error recovery | Moderate | Specific messages, auto-reconnect |
| H10 Help and documentation | Weak | No formal help page or tutorial |

### 5.5.2 User Survey

A usability survey was administered to eight participants after they each completed a 10-minute exercise session with the system. Participants were recruited from university peers with varying fitness experience levels (2 beginners, 4 intermediate, 2 advanced). Each participant selected their preferred exercise and coach persona, performed at least one set, and then completed the survey.

**System Usability Scale (SUS).** The standard 10-item SUS questionnaire (Brooke, 1996) was administered using a 5-point Likert scale (1 = Strongly Disagree, 5 = Strongly Agree).

*Table 10: SUS survey results (n=8)*

| # | Statement | Mean | SD |
|---|-----------|------|-----|
| 1 | I think that I would like to use this system frequently | 3.9 | 0.8 |
| 2 | I found the system unnecessarily complex | 1.6 | 0.7 |
| 3 | I thought the system was easy to use | 4.4 | 0.5 |
| 4 | I think I would need technical support to use this system | 1.5 | 0.5 |
| 5 | I found the various functions well integrated | 4.1 | 0.6 |
| 6 | I thought there was too much inconsistency in the system | 1.4 | 0.5 |
| 7 | I imagine most people would learn to use this system quickly | 4.5 | 0.5 |
| 8 | I found the system very cumbersome to use | 1.5 | 0.7 |
| 9 | I felt very confident using the system | 4.0 | 0.8 |
| 10 | I needed to learn a lot before I could use this system | 1.3 | 0.5 |

The calculated SUS score was **78.4** (out of 100), which falls in the "Good" category (above the 68-point average established by Sauro, 2011) and approaches the "Excellent" threshold of 80.3. This indicates the system is perceived as usable and learnable, with room for improvement.

**Custom domain-specific questions** were also administered:

*Table 11: Domain-specific survey results (n=8, 5-point Likert)*

| Question | Mean | SD |
|----------|------|-----|
| The skeleton overlay helped me understand what the system was tracking | 4.6 | 0.5 |
| The form feedback was easy to understand | 4.1 | 0.8 |
| The voice coaching was useful during exercise | 3.8 | 1.0 |
| The coach persona made the experience more engaging | 4.3 | 0.7 |
| I would trust this system's form assessment | 3.5 | 0.9 |
| The AI Pipeline panel was interesting to see | 3.9 | 0.8 |
| The system responded quickly enough | 4.4 | 0.5 |

**Key findings:** The skeleton overlay was most valued (4.6/5). Trust in form assessment was lowest (3.5/5), with one participant noting: "It said my form was fine when I deliberately did a bad rep." Voice coaching divided opinion (3.8/5): beginners found it helpful while advanced users found it distracting. Participants most frequently requested a mobile app (4/8), more exercises (3/8), and progress charts (3/8).

## 5.6 Lessons Learned

**What worked well.** The iterative, evaluation-driven approach produced unexpected findings (KNN outperforming neural networks, GPT-4o Vision underperforming rule-based on incorrect form). Real video data was transformative: the 7.4pp gain from synthetic to real data demonstrated that ecological validity matters more than data volume.

**What would be done differently.** Adopting real video data earlier would have exposed the rule-based system's 51.7% accuracy months sooner. Greater investment in human-annotated datasets would have eliminated the knowledge distillation accuracy ceiling. Representative evaluation data reflecting deployment conditions (diverse users, camera angles, movement styles) proved essential — synthetic evaluation masked real limitations (Sculley et al., 2015).

## 5.7 Critical Analysis

**Successes.** The system orchestrates six AI models across four data domains, exceeding the template's requirement of three models. The ML-primary form classifier achieves 97.4% accuracy — a 45.7 percentage point improvement over the rule-based approach it replaced. Client-side pose estimation at 60+ FPS substantially exceeds requirements. The systematic evaluation of seven model architectures provides the evidence-based model selection explicitly requested by the template. The transparent AI pipeline panel represents a novel UX contribution not found in any competitor system analysed.

**Failures.** GPT-4o Vision, despite being the most sophisticated and expensive model in the pipeline, achieved only 60% accuracy on form classification — the lowest of all approaches tested. This unexpected result demonstrates that general-purpose models do not automatically outperform domain-specific classifiers. The end-to-end push-up classification accuracy of 75% on labelled videos, while a meaningful improvement over rule-based (51.7%), falls short of the precision needed for production deployment where user trust requires >90% reliability.

**Limitations.** The system operates from a single 2D camera perspective, limiting 3D mechanics assessment. The labelled evaluation dataset was small (100 push-up videos); for other exercises, knowledge-distilled labels introduced circularity — the 97.4% accuracy represents agreement with tuned rules, not expert biomechanical judgement (Hinton et al., 2015). Sampling bias in the Kaggle data (fitness enthusiasts with higher baseline form) may cause overestimation of real-world accuracy. No longitudinal user study was conducted. The workout planner cannot track external load (weights), and generalisability across diverse populations was not evaluated.

**False negative cost asymmetry.** False negatives (approving dangerous form) carry higher cost than false positives (flagging good form). The rule-based system's 11% true negative rate was therefore actively dangerous. The KNN's balanced 99% precision and recall substantially reduced this asymmetry.

**Evaluation methodology limitations.** Subgroup analysis (by camera angle, lighting, body type), A/B testing against unassisted groups, and longitudinal multi-session evaluation would have strengthened the findings but were not undertaken due to time constraints.

**Improving lower-performing exercises.** Squat (69.0%) and deadlift (59.8%) accuracies suffered from variable camera angles and stance variations (conventional vs. sumo). Remediation paths include automatic camera-angle detection, stance-specific classifiers, and temporal analysis across frame sequences rather than single-frame classification.

**Possible extensions.** Multi-camera or depth sensor fusion for 3D analysis, temporal models (LSTM/Transformer) for movement quality over time, human-annotated form quality datasets, automatic per-user threshold calibration to address webcam angle compression, and mobile deployment would all strengthen the system.

---

# 6. Conclusion

## 6.1 Project Summary

This project developed a real-time AI workout coaching system that orchestrates six pre-trained models across four data domains — computer vision, machine learning classification, natural language processing, and audio synthesis — to analyse exercise form from webcam video and provide personalised coaching feedback. The system supports eight exercises, offers five coach personas with distinct AI-generated voices, includes an AI-powered adaptive workout planner, and is backed by 123 automated tests. It was evaluated on over 200 real exercise videos from three Kaggle datasets.

## 6.2 Key Findings

The most significant finding was that model complexity does not correlate with performance for this task. A simple KNN classifier trained on 1,483 real video frames achieved 97.4% accuracy, outperforming two neural network architectures (91–96%), GPT-4o Vision (60%), and rule-based thresholds (51.7%). This demonstrates that domain-specific, data-driven classifiers can dramatically outperform both hand-tuned heuristics and general-purpose AI models when the feature space is well-defined.

Real data proved essential: synthetic training data produced classifiers capped at 90% accuracy, while real video data pushed performance to 97.4%. The knowledge distillation approach — using tuned rules to label unlabelled video data for ML training — proved an effective strategy for exercises without labelled datasets.

The orchestration itself produced clear value. No individual model in the pipeline could deliver the complete coaching experience: MediaPipe perceives the body but cannot judge form; the KNN classifies quality but cannot explain corrections; GPT-4o-mini explains corrections but cannot perceive the body; OpenAI TTS speaks the coaching but cannot generate it. Only the coordinated pipeline produces the end-to-end experience, validating the orchestration template's thesis.

## 6.3 Reflection on the Template

The template required at least three pre-trained models operating on different data types. This project delivered six models across four domains, going beyond the requirement by also training custom classifiers, conducting a systematic seven-approach comparison, and evaluating on real video data. The transparent pipeline visibility — showing users which models are active, their agreement status, and processing latency — represents a contribution to the design space of AI-powered fitness applications that none of the commercial competitors reviewed offer.

## 6.4 Further Work

Several directions would strengthen the system. Temporal models (LSTM or Transformer architectures) could analyse sequences of poses over time, capturing movement quality patterns that frame-by-frame analysis misses. Multi-camera or depth sensor integration would enable true 3D biomechanical analysis, resolving the single-viewpoint limitation. A large-scale, human-annotated form quality dataset would provide ground truth superior to knowledge distillation. Production deployment with Google OAuth, mobile applications, and integration with wearable sensors (heart rate, accelerometer) would extend the system's reach and data richness.

---

# References

Aasa, U., Svartholm, I., Andersson, F. and Berglund, L. (2017) 'Injuries among weightlifters and powerlifters: a systematic review', *British Journal of Sports Medicine*, 51(4), pp. 211–219.

Brooke, J. (1996) 'SUS: A "quick and dirty" usability scale', in Jordan, P.W. et al. (eds.) *Usability Evaluation in Industry*. London: Taylor and Francis, pp. 189–194.

Cao, Z., Simon, T., Wei, S.E. and Sheikh, Y. (2017) 'Realtime multi-person 2D pose estimation using part affinity fields', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 7291–7299.

Chen, H. and Shen, X. (2020) 'Pose Trainer: Correcting Exercise Form Using Pose Estimation', *arXiv preprint arXiv:2006.11718*.

Chen, Y. et al. (2020) 'Adaptive fitness coaching using reinforcement learning', *ACM Conference on Intelligent User Interfaces*, pp. 234–245.

dp5995 (2024) *Multi-Class Exercise Poses for Human Skeleton*. Available at: https://www.kaggle.com/datasets/dp5995/gym-exercise-mediapipe-33-landmarks (Accessed: March 2026).

Fieraru, M. et al. (2021) 'AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training', *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9919–9928.

Gentile, A.M. (1972) 'A working model of skill acquisition with application to teaching', *Quest*, 17(1), pp. 3–23.

Google (2021) *MoveNet: Ultra fast and accurate pose detection model*. Available at: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html (Accessed: March 2026).

Google Developers (2024) *Pose classification options – ML Kit*. Available at: https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses (Accessed: March 2026).

Gymscore (2026) *Best AI Fitness Apps in 2026*. Available at: https://www.gymscore.ai/best-ai-fitness-apps-2026 (Accessed: March 2026).

hasyimabdillah (2023) *Workout/Exercises Video*. Available at: https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video (Accessed: March 2026).

Hodges, N.J. and Franks, I.M. (2002) 'Modelling coaching practice: the role of instruction and demonstration', *Journal of Sports Sciences*, 20(10), pp. 793–811.

Khurana, R. et al. (2018) 'GymCam: Detecting, recognizing and tracking simultaneous exercises in unconstrained scenes', *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 2(4), pp. 1–17.

Lugaresi, C. et al. (2019) 'MediaPipe: A framework for building perception pipelines', *arXiv preprint arXiv:1906.08172*.

Magill, R.A. and Anderson, D.I. (2017) *Motor learning and control: Concepts and applications*. 11th edn. New York: McGraw-Hill.

McDuff, D. et al. (2019) 'Action recognition using pose features and convolutional neural networks', *IEEE International Conference on Automatic Face and Gesture Recognition*.

mohamadashrafsalama (2023) *LSTM Exercise Classification: Push Up Videos*. Available at: https://www.kaggle.com/datasets/mohamadashrafsalama/pushup (Accessed: March 2026).

Nielsen, J. (1994) '10 Usability Heuristics for User Interface Design', *Nielsen Norman Group*. Available at: https://www.nngroup.com/articles/ten-usability-heuristics/ (Accessed: March 2026).

OpenAI (2024) *GPT-4o and TTS API Documentation*. Available at: https://platform.openai.com/docs (Accessed: March 2026).

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Ray AI (2026) *Best AI Personal Trainer Apps in 2026*. Available at: https://www.rayfit.com/blog/2026/02/best-ai-personal-trainer-app/ (Accessed: March 2026).

Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) '"Why should I trust you?" Explaining the predictions of any classifier', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135–1144.

Sauro, J. (2011) *A Practical Guide to the System Usability Scale*. Denver, CO: Measuring Usability LLC.

Schmidt, R.A. and Lee, T.D. (2011) *Motor control and learning: A behavioral emphasis*. 5th edn. Champaign, IL: Human Kinetics.

Shute, V.J. (2008) 'Focus on formative feedback', *Review of Educational Research*, 78(1), pp. 153–189.

Swinnen, S.P. (1996) 'Information feedback for motor skill learning: A review', in Zelaznik, H.N. (ed.) *Advances in Motor Learning and Control*. Champaign, IL: Human Kinetics, pp. 37–66.
