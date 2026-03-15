/**
 * AI Workout Coach - Frontend Application
 *
 * Handles camera capture, WebSocket communication, skeleton overlay rendering,
 * TTS audio playback, and UI updates for real-time workout form analysis.
 *
 * Communicates with a backend that orchestrates four AI models:
 * 1. MediaPipe BlazePose (pose estimation)
 * 2. scikit-learn SVM (form classification)
 * 3. GPT-4o-mini (NLP feedback)
 * 4. gTTS (text-to-speech)
 */

class WorkoutCoach {
    constructor() {
        this.config = {
            wsUrl: `ws://${window.location.host}/ws/workout`,
            frameRate: 15,
            videoWidth: 640,
            videoHeight: 480,
        };

        // State
        this.isRunning = false;
        this.currentExercise = 'squat';
        this.currentCoach = 'coach_pro';
        this.sessionStartTime = null;
        this.ws = null;
        this.frameInterval = null;
        this.durationInterval = null;
        this.ttsEnabled = true;
        this.ttsQueue = [];
        this.isSpeaking = false;

        // DOM Elements
        this.elements = {
            video: document.getElementById('videoElement'),
            canvas: document.getElementById('overlayCanvas'),
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            resetBtn: document.getElementById('resetBtn'),
            connectionStatus: document.getElementById('connectionStatus'),
            feedbackOverlay: document.getElementById('feedbackOverlay'),
            feedbackText: document.getElementById('feedbackText'),
            repCounter: document.querySelector('.rep-number'),
            phaseIndicator: document.querySelector('.phase-text'),
            totalReps: document.getElementById('totalReps'),
            goodReps: document.getElementById('goodReps'),
            formScore: document.getElementById('formScore'),
            duration: document.getElementById('duration'),
            leftKnee: document.getElementById('leftKnee'),
            rightKnee: document.getElementById('rightKnee'),
            torsoAngle: document.getElementById('torsoAngle'),
            tipText: document.getElementById('tipText'),
            errorLog: document.getElementById('errorLog'),
            exerciseBtns: document.querySelectorAll('.exercise-btn'),
            videoContainer: document.querySelector('.video-container'),
            ttsToggle: document.getElementById('ttsToggle'),
            modelAgreement: document.getElementById('modelAgreement'),
            mlClassification: document.getElementById('mlClassification'),
            pipelineTiming: document.getElementById('pipelineTiming'),
        };

        // Canvas context for skeleton overlay
        this.ctx = this.elements.canvas.getContext('2d');

        // Skeleton connections for drawing
        this.skeletonConnections = [
            ['left_shoulder', 'right_shoulder'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist'],
            ['left_shoulder', 'left_hip'],
            ['right_shoulder', 'right_hip'],
            ['left_hip', 'right_hip'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
        ];

        this.init();
    }

    async init() {
        this.authToken = localStorage.getItem('authToken');
        this.username = localStorage.getItem('username');

        // Show auth modal if not logged in
        if (!this.authToken) {
            this.showAuthModal();
        } else {
            this.hideAuthModal();
        }

        this.bindEvents();
        await this.setupCamera();
        this.loadSessionHistory();
        this.loadAchievements();
        this.loadCurrentPlan();
    }

    showAuthModal() {
        document.getElementById('authModal')?.classList.remove('hidden');
    }

    hideAuthModal() {
        document.getElementById('authModal')?.classList.add('hidden');
        // Show username and logout in header
        const userEl = document.getElementById('headerUser');
        const logoutBtn = document.getElementById('logoutBtn');
        if (userEl && this.username) {
            userEl.textContent = this.username;
            userEl.style.color = 'var(--accent-light)';
            userEl.style.fontSize = '0.8rem';
            userEl.style.fontWeight = '600';
        }
        if (logoutBtn && this.authToken) {
            logoutBtn.style.display = 'inline-flex';
        }
    }

    logout() {
        localStorage.removeItem('authToken');
        localStorage.removeItem('username');
        this.authToken = null;
        this.username = null;
        // Stop any active workout
        if (this.isRunning) this.stop();
        // Show login modal
        document.getElementById('headerUser').textContent = '';
        document.getElementById('logoutBtn').style.display = 'none';
        this.showAuthModal();
    }

    async handleAuth(action) {
        const username = document.getElementById('authUsername')?.value;
        const password = document.getElementById('authPassword')?.value;
        const errorEl = document.getElementById('authError');

        if (!username || !password) {
            if (errorEl) errorEl.textContent = 'Please enter username and password';
            return;
        }

        try {
            const resp = await fetch(`/api/auth/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await resp.json();

            if (!resp.ok) {
                if (errorEl) errorEl.textContent = data.detail || 'Authentication failed';
                return;
            }

            this.authToken = data.token;
            this.username = data.user.username;
            localStorage.setItem('authToken', data.token);
            localStorage.setItem('username', data.user.username);
            this.hideAuthModal();
        } catch (e) {
            if (errorEl) errorEl.textContent = 'Connection error';
        }
    }

    bindEvents() {
        this.elements.startBtn.addEventListener('click', () => this.start());
        this.elements.stopBtn.addEventListener('click', () => this.stop());
        this.elements.resetBtn.addEventListener('click', () => this.reset());

        this.elements.exerciseBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const exercise = e.currentTarget.dataset.exercise;
                this.selectExercise(exercise);
            });
        });

        // TTS toggle
        if (this.elements.ttsToggle) {
            this.elements.ttsToggle.addEventListener('click', () => this.toggleTTS());
        }

        // Coach selection
        document.querySelectorAll('.coach-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const coachId = e.currentTarget.dataset.coach;
                this.selectCoach(coachId);
            });
        });

        // Generate plan button
        const planBtn = document.getElementById('generatePlanBtn');
        if (planBtn) {
            planBtn.addEventListener('click', () => this.generatePlan());
        }

        // Auth buttons
        document.getElementById('loginBtn')?.addEventListener('click', () => this.handleAuth('login'));
        document.getElementById('registerBtn')?.addEventListener('click', () => this.handleAuth('register'));
        document.getElementById('skipAuthBtn')?.addEventListener('click', () => {
            this.authToken = null;
            this.username = 'Guest';
            this.hideAuthModal();
        });

        // Enter key in auth form
        document.getElementById('authPassword')?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.handleAuth('login');
        });

        // Logout
        document.getElementById('logoutBtn')?.addEventListener('click', () => this.logout());

        // Session summary close
        document.getElementById('closeSummaryBtn')?.addEventListener('click', () => {
            document.getElementById('summaryModal')?.classList.add('hidden');
        });
    }

    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: this.config.videoWidth },
                    height: { ideal: this.config.videoHeight },
                    facingMode: 'user',
                },
            });

            this.elements.video.srcObject = stream;

            this.elements.video.onloadedmetadata = () => {
                this.elements.canvas.width = this.elements.video.videoWidth;
                this.elements.canvas.height = this.elements.video.videoHeight;
            };

            this.updateFeedback('Camera ready! Click "Start Workout" to begin.', 'info');
        } catch (err) {
            console.error('Camera access error:', err);
            this.updateFeedback('Camera access denied. Please allow camera access and reload.', 'error');
            // Show a more helpful message
            this.elements.video.style.display = 'none';
            const container = this.elements.video.parentElement;
            const msg = document.createElement('div');
            msg.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#94a3b8;font-size:1.1rem;padding:20px;';
            msg.innerHTML = 'Camera access required.<br><small style="color:#64748b">Please allow camera permission and reload the page.</small>';
            container.appendChild(msg);
        }
    }

    // ── WebSocket ──

    connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                this.ws.send(JSON.stringify({
                    type: 'config',
                    exercise: this.currentExercise,
                    coach: this.currentCoach,
                }));
                resolve();
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleServerMessage(data);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
                reject(error);
            };

            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.updateConnectionStatus(false);
                if (this.isRunning) {
                    setTimeout(() => this.connectWebSocket(), 2000);
                }
            };
        });
    }

    handleServerMessage(data) {
        switch (data.type) {
            case 'config_ack':
                console.log('Exercise configured:', data.exercise);
                if (data.models) {
                    this.updateModelInfo(data.models);
                }
                break;

            case 'analysis':
                this.updateUI(data);
                this.drawSkeleton(data.pose?.landmarks, data.analysis?.is_good_form);
                break;

            case 'llm_feedback':
                // Rich LLM-generated feedback (arrives async after analysis)
                this.updateFeedbackFromLLM(data);
                break;

            case 'tts_audio':
                // TTS audio from server
                if (this.ttsEnabled && data.audio) {
                    this.playAudioBase64(data.audio);
                }
                break;

            case 'vision_analysis':
                // GPT-4o Vision form analysis (arrives async)
                this.updateVisionAnalysis(data);
                break;

            case 'error':
                console.error('Server error:', data.message);
                this.updateFeedback(data.message, 'error');
                break;

            case 'reset_ack':
                this.resetUI();
                break;

            case 'summary':
                console.log('Session summary:', data);
                break;

            case 'pong':
                break;
        }
    }

    // ── Skeleton Overlay Drawing ──

    drawSkeleton(landmarks, isGoodForm) {
        const canvas = this.elements.canvas;
        const ctx = this.ctx;
        const w = canvas.width;
        const h = canvas.height;

        // Clear previous overlay
        ctx.clearRect(0, 0, w, h);

        if (!landmarks || Object.keys(landmarks).length === 0) return;

        const color = isGoodForm ? '#00ff88' : '#ff4444';
        const jointColor = isGoodForm ? '#00ff88' : '#ffaa00';

        // Draw connections
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.8;

        for (const [from, to] of this.skeletonConnections) {
            if (landmarks[from] && landmarks[to]) {
                const fx = (1 - landmarks[from].x) * w; // Mirror for selfie view
                const fy = landmarks[from].y * h;
                const tx = (1 - landmarks[to].x) * w;
                const ty = landmarks[to].y * h;

                ctx.beginPath();
                ctx.moveTo(fx, fy);
                ctx.lineTo(tx, ty);
                ctx.stroke();
            }
        }

        // Draw joints
        ctx.fillStyle = jointColor;
        ctx.globalAlpha = 0.9;

        for (const [name, lm] of Object.entries(landmarks)) {
            if (lm.visibility > 0.5) {
                const x = (1 - lm.x) * w;
                const y = lm.y * h;

                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        ctx.globalAlpha = 1.0;
    }

    // ── TTS Audio ──

    toggleTTS() {
        this.ttsEnabled = !this.ttsEnabled;
        if (this.elements.ttsToggle) {
            this.elements.ttsToggle.textContent = this.ttsEnabled ? 'Mute Voice' : 'Unmute Voice';
            this.elements.ttsToggle.classList.toggle('muted', !this.ttsEnabled);
        }
    }

    playAudioBase64(base64Audio) {
        if (!this.ttsEnabled || this.isSpeaking) return;

        try {
            this.isSpeaking = true;
            const audioData = atob(base64Audio);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }

            const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);

            audio.onended = () => {
                this.isSpeaking = false;
                URL.revokeObjectURL(url);
            };

            audio.onerror = () => {
                this.isSpeaking = false;
                URL.revokeObjectURL(url);
            };

            audio.play().catch(() => {
                this.isSpeaking = false;
            });
        } catch (e) {
            console.error('Audio playback error:', e);
            this.isSpeaking = false;
        }
    }

    // ── UI Updates ──

    updateUI(data) {
        // Feedback — don't overwrite valid feedback with "cannot detect" / "not in position"
        if (data.feedback) {
            const isInfoOnly = data.analysis?.errors?.every(e =>
                e.error_type === 'no_pose_detected' || e.error_type === 'not_in_position'
            );

            // Only update feedback if it's actual form feedback (not transient detection loss)
            if (!isInfoOnly || !this._lastValidFeedback) {
                const severity = data.analysis?.is_good_form ? 'info' :
                    (data.analysis?.errors?.[0]?.severity || 'warning');
                this.updateFeedback(data.feedback.spoken, severity);
                if (!isInfoOnly) {
                    this._lastValidFeedback = data.feedback.spoken;
                }
            }

            if (data.feedback.tip && this.elements.tipText) {
                this.elements.tipText.textContent = data.feedback.tip;
            }
        }

        // Rep counter and phase
        if (data.analysis) {
            this.elements.repCounter.textContent = data.analysis.rep_count;
            this.elements.phaseIndicator.textContent = this.formatPhase(data.analysis.phase);

            const feedbackContent = this.elements.feedbackOverlay?.querySelector('.feedback-content');
            if (feedbackContent) {
                feedbackContent.classList.remove('error', 'warning');
                if (!data.analysis.is_good_form) {
                    feedbackContent.classList.add('warning');
                }
            }
        }

        // Session stats
        if (data.session) {
            this.elements.totalReps.textContent = data.session.total_reps;
            this.elements.goodReps.textContent = data.session.good_form_reps;
            this.elements.formScore.textContent = `${Math.round(data.session.form_score)}%`;

            // Store form dimensions for summary
            if (data.session.form_dimensions) {
                this._lastDimensions = data.session.form_dimensions;
            }
        }

        // Angles
        if (data.pose?.angles) {
            const angles = data.pose.angles;
            if (this.currentExercise === 'pushup') {
                this.elements.leftKnee.textContent = angles.left_elbow_angle
                    ? `${Math.round(angles.left_elbow_angle)}` : '--';
                this.elements.rightKnee.textContent = angles.right_elbow_angle
                    ? `${Math.round(angles.right_elbow_angle)}` : '--';
            } else {
                this.elements.leftKnee.textContent = angles.left_knee_angle
                    ? `${Math.round(angles.left_knee_angle)}` : '--';
                this.elements.rightKnee.textContent = angles.right_knee_angle
                    ? `${Math.round(angles.right_knee_angle)}` : '--';
            }
            this.elements.torsoAngle.textContent = angles.torso_angle
                ? `${Math.round(angles.torso_angle)}` : '--';
        }

        // ML Classification info
        if (data.ml_classification && this.elements.mlClassification) {
            const cls = data.ml_classification;
            this.elements.mlClassification.textContent =
                `${cls.quality_label} (${Math.round(cls.confidence * 100)}%)`;
            this.elements.mlClassification.className = `ml-value ${cls.quality_label}`;
        }

        // Model agreement
        if (this.elements.modelAgreement) {
            this.elements.modelAgreement.textContent = data.models_agree ? 'Agree' : 'Disagree';
            this.elements.modelAgreement.className = `agreement-value ${data.models_agree ? 'agree' : 'disagree'}`;
        }

        // Pipeline timing
        if (data.pipeline_timing && this.elements.pipelineTiming) {
            this.elements.pipelineTiming.textContent = `${Math.round(data.pipeline_timing.total_ms)}ms`;
        }

        // Error log — only show actual form errors, not "no_pose_detected" or "not_in_position"
        if (data.analysis?.errors && data.analysis.errors.length > 0) {
            const formErrors = data.analysis.errors.filter(
                e => e.error_type !== 'no_pose_detected' && e.error_type !== 'not_in_position'
            );
            if (formErrors.length > 0) {
                this.updateErrorLog(formErrors);
            }
        }
    }

    updateFeedbackFromLLM(data) {
        // Update with richer LLM feedback when it arrives
        if (data.spoken) {
            this.updateFeedback(data.spoken, 'info');
        }
        if (data.detailed) {
            const detailedEl = document.getElementById('detailedFeedback');
            if (detailedEl) detailedEl.textContent = data.detailed;
        }
        if (data.tip && this.elements.tipText) {
            this.elements.tipText.textContent = data.tip;
        }
    }

    updateVisionAnalysis(data) {
        const el = document.getElementById('visionAnalysis');
        if (!el) return;

        const assessClass = data.form_assessment === 'good' ? 'good_form' :
            data.form_assessment === 'major_issues' ? 'major_issues' : 'minor_issues';

        let html = `<div class="vision-header">
            <span class="vision-label">GPT-4o Vision</span>
            <span class="ml-value ${assessClass}">${data.form_assessment} (${data.confidence})</span>
        </div>`;

        if (data.observations && data.observations.length > 0) {
            html += `<div class="vision-observations">${data.observations.join('; ')}</div>`;
        }
        if (data.suggestions && data.suggestions.length > 0) {
            html += `<div class="vision-suggestions">${data.suggestions.join('; ')}</div>`;
        }

        el.innerHTML = html;
    }

    updateModelInfo(models) {
        const infoEl = document.getElementById('modelPipelineInfo');
        if (!infoEl) return;

        const entries = Object.entries(models).map(([key, m]) => {
            const status = m.status === 'active' ? 'active' : 'inactive';
            return `<div class="model-entry ${status}">
                <span class="model-name">${m.name}</span>
                <span class="model-domain">${m.domain}</span>
            </div>`;
        });

        infoEl.innerHTML = entries.join('');
    }

    updateFeedback(text, severity = 'info') {
        this.elements.feedbackText.textContent = text;
        const content = this.elements.feedbackOverlay?.querySelector('.feedback-content');
        if (content) {
            content.classList.remove('error', 'warning', 'info');
            content.classList.add(severity);
        }
    }

    updateErrorLog(errors) {
        const log = this.elements.errorLog;
        if (!log) return;

        const uniqueErrors = [...new Map(errors.map(e => [e.error_type, e])).values()];

        log.innerHTML = uniqueErrors.map(error => `
            <div class="error-item ${error.severity}">
                <span>${error.message}</span>
            </div>
        `).join('');
    }

    updateConnectionStatus(connected) {
        const statusEl = this.elements.connectionStatus;
        if (!statusEl) return;

        if (connected) {
            statusEl.classList.add('connected');
            const textEl = statusEl.querySelector('.status-text');
            if (textEl) textEl.textContent = 'Session Active';
        } else {
            statusEl.classList.remove('connected');
            const textEl = statusEl.querySelector('.status-text');
            if (textEl) textEl.textContent = 'Ready';
        }
    }

    formatPhase(phase) {
        const phaseNames = {
            'standing': 'Standing',
            'descending': 'Going Down',
            'bottom': 'Bottom',
            'ascending': 'Coming Up',
            'plank': 'Plank',
            'lowering': 'Lowering',
            'lowest': 'Bottom',
            'pushing': 'Pushing Up',
        };
        return phaseNames[phase] || phase;
    }

    selectExercise(exercise) {
        this.currentExercise = exercise;

        this.elements.exerciseBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.exercise === exercise);
        });

        // Update angle labels
        if (exercise === 'pushup') {
            const items = document.querySelectorAll('.angle-item .angle-name');
            if (items[0]) items[0].textContent = 'Left Elbow';
            if (items[1]) items[1].textContent = 'Right Elbow';
            if (items[2]) items[2].textContent = 'Hip Angle';
        } else {
            const items = document.querySelectorAll('.angle-item .angle-name');
            if (items[0]) items[0].textContent = 'Left Knee';
            if (items[1]) items[1].textContent = 'Right Knee';
            if (items[2]) items[2].textContent = 'Torso';
        }

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'config',
                exercise: exercise,
                coach: this.currentCoach,
            }));
        }
    }

    selectCoach(coachId) {
        this.currentCoach = coachId;

        document.querySelectorAll('.coach-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.coach === coachId);
        });

        // Send config update to server if connected
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'config',
                exercise: this.currentExercise,
                coach: coachId,
            }));
        }
    }

    // ── Session Control ──

    async start() {
        try {
            await this.connectWebSocket();

            this.isRunning = true;
            this.sessionStartTime = Date.now();

            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.videoContainer.classList.add('active');

            this.startFrameCapture();
            this.startDurationTimer();

            this.updateFeedback('Workout started! Get in position.', 'info');
        } catch (err) {
            console.error('Failed to start:', err);
            this.updateFeedback('Failed to connect. Is the server running?', 'error');
        }
    }

    stop() {
        this.isRunning = false;

        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.elements.videoContainer.classList.remove('active');

        // Clear skeleton overlay
        this.ctx.clearRect(0, 0, this.elements.canvas.width, this.elements.canvas.height);

        // Save session and show summary
        this.saveSession().then(() => this.showSessionSummary());

        this.updateFeedback('Workout stopped. Great job!', 'info');
    }

    reset() {
        this.elements.totalReps.textContent = '0';
        this.elements.goodReps.textContent = '0';
        this.elements.formScore.textContent = '100%';
        this.elements.repCounter.textContent = '0';
        this.elements.duration.textContent = '0:00';
        if (this.elements.errorLog) {
            this.elements.errorLog.innerHTML = '<div class="error-empty">No errors detected yet</div>';
        }

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }

        this.sessionStartTime = Date.now();
        this.updateFeedback('Session reset. Ready to start!', 'info');
    }

    resetUI() {
        this.elements.totalReps.textContent = '0';
        this.elements.goodReps.textContent = '0';
        this.elements.formScore.textContent = '100%';
        this.elements.repCounter.textContent = '0';
    }

    showSessionSummary() {
        const modal = document.getElementById('summaryModal');
        if (!modal) return;

        const totalReps = parseInt(this.elements.totalReps?.textContent || '0');
        const goodReps = parseInt(this.elements.goodReps?.textContent || '0');
        const formScore = parseFloat(this.elements.formScore?.textContent || '0');
        const duration = this.elements.duration?.textContent || '0:00';

        // Stats
        const statsEl = document.getElementById('summaryStats');
        if (statsEl) {
            statsEl.innerHTML = `
                <div class="summary-stat">
                    <span class="summary-stat-value">${totalReps}</span>
                    <span class="summary-stat-label">Total Reps</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">${goodReps}</span>
                    <span class="summary-stat-label">Good Form</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">${Math.round(formScore)}%</span>
                    <span class="summary-stat-label">Form Score</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">${duration}</span>
                    <span class="summary-stat-label">Duration</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">${this.currentExercise}</span>
                    <span class="summary-stat-label">Exercise</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">${this.currentCoach.replace('_', ' ')}</span>
                    <span class="summary-stat-label">Coach</span>
                </div>
            `;
        }

        // Dimension radar bars (from last known session data)
        const radarEl = document.getElementById('summaryRadar');
        if (radarEl && this._lastDimensions) {
            radarEl.innerHTML = Object.entries(this._lastDimensions).map(([dim, score]) => {
                const cls = score >= 80 ? '' : score >= 50 ? 'ok' : 'bad';
                return `<div class="radar-dim">
                    <span class="radar-label">${dim}</span>
                    <div class="radar-bar-bg">
                        <div class="radar-bar-fill ${cls}" style="width:${score}%"></div>
                    </div>
                    <span class="radar-value">${Math.round(score)}%</span>
                </div>`;
            }).join('');
        } else if (radarEl) {
            // Default: all 100%
            radarEl.innerHTML = ['Depth', 'Alignment', 'Symmetry', 'Tempo', 'Consistency'].map(dim =>
                `<div class="radar-dim">
                    <span class="radar-label">${dim}</span>
                    <div class="radar-bar-bg">
                        <div class="radar-bar-fill" style="width:100%"></div>
                    </div>
                    <span class="radar-value">100%</span>
                </div>`
            ).join('');
        }

        // Coach note
        const noteEl = document.getElementById('summaryCoachNote');
        if (noteEl) {
            const notes = {
                coach_pro: "Solid session! Keep up the consistent work.",
                drill_sergeant: "Not bad, soldier. But we go HARDER next time!",
                zen_master: "A beautiful journey of movement. Carry this peace with you.",
                hype_beast: "YO THAT SESSION WAS ABSOLUTELY FIRE! 🔥",
                pop_diva: "Darling, you were FABULOUS today! Standing ovation! 👏",
            };
            noteEl.textContent = notes[this.currentCoach] || notes.coach_pro;
        }

        modal.classList.remove('hidden');
    }

    startFrameCapture() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.config.videoWidth;
        canvas.height = this.config.videoHeight;

        const captureFrame = () => {
            if (!this.isRunning || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

            ctx.drawImage(this.elements.video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.7);

            this.ws.send(JSON.stringify({
                type: 'frame',
                image: imageData,
            }));
        };

        this.frameInterval = setInterval(captureFrame, 1000 / this.config.frameRate);
    }

    startDurationTimer() {
        const updateDuration = () => {
            if (!this.sessionStartTime) return;
            const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            this.elements.duration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        };

        this.durationInterval = setInterval(updateDuration, 1000);
    }

    // ── API Methods (Session, Plan, Achievements) ──

    async saveSession() {
        const totalReps = parseInt(this.elements.totalReps?.textContent || '0');
        if (totalReps === 0 && !this.sessionStartTime) return;

        const goodReps = parseInt(this.elements.goodReps?.textContent || '0');
        const formScore = parseFloat(this.elements.formScore?.textContent || '0');

        try {
            const resp = await fetch('/api/session/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    exercise_type: this.currentExercise,
                    coach_persona: this.currentCoach,
                    start_time: (this.sessionStartTime || Date.now()) / 1000,
                    end_time: Date.now() / 1000,
                    total_reps: totalReps,
                    good_form_reps: goodReps,
                    form_score: formScore,
                    errors: {},
                    pipeline_avg_latency_ms: 0,
                }),
            });
            const data = await resp.json();

            // Show newly unlocked achievements
            if (data.achievements_unlocked?.length > 0) {
                const names = data.achievement_details.map(a => `${a.icon} ${a.name}`).join(', ');
                this.updateFeedback(`Achievement unlocked: ${names}`, 'info');
            }

            // Refresh UI
            this.loadSessionHistory();
            this.loadAchievements();
        } catch (e) {
            console.error('Failed to save session:', e);
        }
    }

    async loadSessionHistory() {
        try {
            const resp = await fetch('/api/sessions?limit=5');
            const data = await resp.json();
            const el = document.getElementById('sessionHistory');
            if (!el) return;

            if (!data.sessions || data.sessions.length === 0) {
                el.innerHTML = '<p class="plan-empty">No sessions yet</p>';
                return;
            }

            el.innerHTML = data.sessions.map(s => {
                const scoreClass = s.form_score >= 80 ? 'good' : s.form_score >= 50 ? 'ok' : 'bad';
                const date = new Date(s.start_time * 1000).toLocaleDateString();
                return `<div class="session-item">
                    <span class="session-exercise">${s.exercise_type}</span>
                    <span class="session-reps">${s.total_reps} reps</span>
                    <span class="session-score ${scoreClass}">${Math.round(s.form_score)}%</span>
                    <span class="session-reps">${date}</span>
                </div>`;
            }).join('');
        } catch (e) {
            console.error('Failed to load sessions:', e);
        }
    }

    async loadAchievements() {
        try {
            const resp = await fetch('/api/achievements');
            const data = await resp.json();
            const el = document.getElementById('achievements');
            if (!el) return;

            if (!data.achievements || data.achievements.length === 0) {
                el.innerHTML = '<p class="plan-empty">Complete workouts to earn badges!</p>';
                return;
            }

            el.innerHTML = data.achievements.map(a =>
                `<div class="achievement-badge">
                    <span class="achievement-icon">${a.icon || '⭐'}</span>
                    <span class="achievement-name">${a.name || a.achievement_type}</span>
                </div>`
            ).join('');
        } catch (e) {
            console.error('Failed to load achievements:', e);
        }
    }

    async loadCurrentPlan() {
        try {
            const resp = await fetch('/api/workout-plan/current');
            const data = await resp.json();
            this.displayPlan(data?.plan || null);
        } catch (e) {
            console.error('Failed to load plan:', e);
        }
    }

    async generatePlan() {
        const btn = document.getElementById('generatePlanBtn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Generating...';
        }

        try {
            const resp = await fetch('/api/workout-plan/generate', { method: 'POST' });
            const plan = await resp.json();
            this.displayPlan(plan);
        } catch (e) {
            console.error('Failed to generate plan:', e);
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'Generate AI Plan';
            }
        }
    }

    displayPlan(plan) {
        const el = document.getElementById('workoutPlan');
        if (!el) return;

        if (!plan || !plan.exercises) {
            el.innerHTML = '<p class="plan-empty">No plan yet</p>';
            return;
        }

        let html = plan.exercises.map(ex =>
            `<div class="plan-exercise">
                <div class="plan-exercise-name">${ex.exercise} — ${ex.sets}x${ex.target_reps}</div>
                <div class="plan-exercise-detail">Rest ${ex.rest_seconds}s between sets</div>
                ${ex.focus ? `<div class="plan-exercise-focus">${ex.focus}</div>` : ''}
            </div>`
        ).join('');

        if (plan.ai_notes) {
            html += `<div class="plan-notes">${plan.ai_notes}</div>`;
        }

        el.innerHTML = html;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.workoutCoach = new WorkoutCoach();
});
