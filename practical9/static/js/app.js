/* ==========================================================================
   AegisVision - Public Safety Mask Compliance App Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Mode Buttons & Containers
    const btnModeCam = document.getElementById('btnModeCam');
    const btnModeSim = document.getElementById('btnModeSim');
    const btnModeUpload = document.getElementById('btnModeUpload');
    
    const webcamControls = document.getElementById('webcamControls');
    const uploadControls = document.getElementById('uploadControls');
    const simControls = document.getElementById('simControls');
    
    const btnStartWebcam = document.getElementById('btnStartWebcam');
    const btnStopWebcam = document.getElementById('btnStopWebcam');
    const btnRefreshSim = document.getElementById('btnRefreshSim');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    
    // Viewport Elements
    const videoElement = document.getElementById('videoElement');
    const outputCanvas = document.getElementById('outputCanvas');
    const ctx = outputCanvas.getContext('2d');
    const placeholderOverlay = document.getElementById('placeholderOverlay');
    const streamTag = document.getElementById('streamTag');
    const camTitle = document.getElementById('camTitle');
    const fpsDisplay = document.getElementById('fpsDisplay');
    
    // Settings & Metrics
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confVal = document.getElementById('confVal');
    const toggleAudioAlert = document.getElementById('toggleAudioAlert');
    
    const statTotal = document.getElementById('statTotal');
    const statCompliant = document.getElementById('statCompliant');
    const statViolations = document.getElementById('statViolations');
    const statRate = document.getElementById('statRate');
    
    const globalAlertBanner = document.getElementById('globalAlertBanner');
    const globalAlertIcon = document.getElementById('globalAlertIcon');
    const globalStatusTitle = document.getElementById('globalStatusTitle');
    const globalStatusDesc = document.getElementById('globalStatusDesc');

    // State Variables
    let activeMode = 'webcam'; // 'webcam', 'sim', 'upload'
    let isWebcamRunning = false;
    let animationFrameId = null;
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let fps = 0;

    // Web Audio Synthesizer for Safety Warning Beep
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function playWarningBeep() {
        if (!toggleAudioAlert.checked) return;
        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.3);
        } catch (e) {
            console.log('Audio alert blocked:', e);
        }
    }

    // --- Tab Switching Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });

    // --- Mode Switching Logic ---
    function setMode(mode) {
        activeMode = mode;
        stopWebcam();
        
        btnModeCam.classList.toggle('active', mode === 'webcam');
        btnModeSim.classList.toggle('active', mode === 'sim');
        btnModeUpload.classList.toggle('active', mode === 'upload');
        
        webcamControls.classList.toggle('hidden', mode !== 'webcam');
        simControls.classList.toggle('hidden', mode !== 'sim');
        uploadControls.classList.toggle('hidden', mode !== 'upload');

        placeholderOverlay.classList.remove('hidden');
        
        if (mode === 'webcam') {
            streamTag.innerHTML = '<i class="fa-solid fa-circle"></i> CAMERA STREAM';
            camTitle.textContent = 'CAM_01 • MAIN ENTRANCE HALLWAY';
        } else if (mode === 'sim') {
            streamTag.innerHTML = '<i class="fa-solid fa-tower-broadcast"></i> SURVEILLANCE SIMULATOR';
            camTitle.textContent = 'CAM_04 • PUBLIC PLAZA SIMULATOR';
            loadSurveillanceSimulation();
        } else if (mode === 'upload') {
            streamTag.innerHTML = '<i class="fa-solid fa-image"></i> PHOTO ANALYSIS';
            camTitle.textContent = 'FILE_UPLOAD • STATIC ANALYSIS';
        }
    }

    btnModeCam.addEventListener('click', () => setMode('webcam'));
    btnModeSim.addEventListener('click', () => setMode('sim'));
    btnModeUpload.addEventListener('click', () => setMode('upload'));

    // --- Settings Slider ---
    confidenceSlider.addEventListener('input', (e) => {
        confVal.textContent = e.target.value + '%';
    });

    // --- Webcam Controls ---
    async function startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 420 }
            });
            videoElement.srcObject = stream;
            videoElement.play();
            isWebcamRunning = true;
            
            btnStartWebcam.classList.add('hidden');
            btnStopWebcam.classList.remove('hidden');
            placeholderOverlay.classList.add('hidden');
            
            requestAnimationFrame(processWebcamFrame);
        } catch (err) {
            alert('Camera access denied or device unavailable. Switching to Simulation Mode.');
            setMode('sim');
        }
    }

    function stopWebcam() {
        if (isWebcamRunning) {
            isWebcamRunning = false;
            if (videoElement.srcObject) {
                videoElement.srcObject.getTracks().forEach(track => track.stop());
            }
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            btnStartWebcam.classList.remove('hidden');
            btnStopWebcam.classList.add('hidden');
        }
    }

    btnStartWebcam.addEventListener('click', startWebcam);
    btnStopWebcam.addEventListener('click', stopWebcam);

    // --- Frame Processing Loop for Live Camera ---
    async function processWebcamFrame() {
        if (!isWebcamRunning) return;
        
        ctx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);
        
        // Capture frame base64
        const dataUrl = outputCanvas.toDataURL('image/jpeg', 0.8);
        
        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: dataUrl,
                    threshold: parseFloat(confidenceSlider.value) / 100.0
                })
            });
            const data = await res.json();
            
            if (data.success) {
                renderDetections(data);
            }
        } catch (e) {
            console.error('Frame detect error:', e);
        }

        // FPS Calculation
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            fpsDisplay.textContent = fps + ' FPS';
            frameCount = 0;
            lastFrameTime = now;
        }

        if (isWebcamRunning) {
            animationFrameId = requestAnimationFrame(processWebcamFrame);
        }
    }

    // --- Simulation Mode Handler ---
    async function loadSurveillanceSimulation() {
        placeholderOverlay.classList.remove('hidden');
        try {
            const res = await fetch('/api/sample');
            const data = await res.json();
            if (data.success) {
                const img = new Image();
                img.onload = () => {
                    outputCanvas.width = img.width;
                    outputCanvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    placeholderOverlay.classList.add('hidden');
                    renderDetections(data);
                };
                img.src = data.image_url + '?t=' + new Date().getTime();
            }
        } catch (e) {
            console.error('Sim error:', e);
        }
    }

    btnRefreshSim.addEventListener('click', loadSurveillanceSimulation);

    // --- File Upload Handler ---
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = async () => {
                outputCanvas.width = img.width;
                outputCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                placeholderOverlay.classList.add('hidden');
                
                const dataUrl = outputCanvas.toDataURL('image/jpeg', 0.85);
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: dataUrl,
                        threshold: parseFloat(confidenceSlider.value) / 100.0
                    })
                });
                const data = await res.json();
                if (data.success) {
                    renderDetections(data);
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // --- Bounding Box & Statistics Render Function ---
    function renderDetections(data) {
        const detections = data.detections || [];
        const total = data.total_faces || detections.length;
        const compliant = data.compliant_count || 0;
        const violations = data.violations_count || 0;
        const rate = total > 0 ? ((compliant / total) * 100).toFixed(1) : '100.0';

        // Update Stat Cards
        statTotal.textContent = total;
        statCompliant.textContent = compliant;
        statViolations.textContent = violations;
        statRate.textContent = rate + '%';

        // Global Alert Banner State
        const isSafe = parseFloat(rate) >= 70.0;
        if (isSafe) {
            globalAlertBanner.className = 'alert-status-banner safe';
            globalAlertIcon.className = 'fa-solid fa-circle-check';
            globalStatusTitle.textContent = 'HIGH COMPLIANCE ZONE';
            globalStatusDesc.textContent = `Compliance rate is ${rate}%, meeting the ≥70% safety requirement.`;
        } else {
            globalAlertBanner.className = 'alert-status-banner warning';
            globalAlertIcon.className = 'fa-solid fa-triangle-exclamation';
            globalStatusTitle.textContent = 'SAFETY ALERT: LOW COMPLIANCE';
            globalStatusDesc.textContent = `Compliance rate dropped to ${rate}%! Immediate enforcement required.`;
            playWarningBeep();
        }

        // Draw Bounding Boxes on Canvas if provided
        detections.forEach(det => {
            const [x, y, w, h] = det.box;
            const cls = det.class;
            const conf = (det.confidence * 100).toFixed(1);

            let strokeColor = '#10b981'; // Green
            let labelText = `Mask (${conf}%)`;

            if (cls === 'without_mask') {
                strokeColor = '#ef4444'; // Red
                labelText = `NO MASK (${conf}%)`;
            } else if (cls === 'mask_incorrect') {
                strokeColor = '#f59e0b'; // Yellow
                labelText = `INCORRECT (${conf}%)`;
            }

            // Draw bounding box
            ctx.lineWidth = 3;
            ctx.strokeStyle = strokeColor;
            ctx.strokeRect(x, y, w, h);

            // Draw tag header
            ctx.fillStyle = strokeColor;
            ctx.fillRect(x, y - 24, w, 24);

            // Draw label text
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 12px Outfit, sans-serif';
            ctx.fillText(labelText, x + 6, y - 7);
        });
    }
});
