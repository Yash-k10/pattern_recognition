/* ============================================
   GestureSense — Frontend Application Logic
   ============================================ */

// ---- State ----
let cameraActive = false;
let videoStream = null;
let recognitionLoop = null;
let facingMode = "user";
let frameCount = 0;
let detectionCount = 0;
let fpsCounter = 0;
let lastFpsTime = Date.now();
let lastGesture = null;

// ---- DOM Elements ----
const webcam = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const startBtn = document.getElementById("startBtn");
const flipBtn = document.getElementById("flipBtn");
const gestureBadge = document.getElementById("gestureBadge");
const gestureEmoji = document.getElementById("gestureEmoji");
const gestureLabel = document.getElementById("gestureLabel");
const gestureIconLarge = document.getElementById("gestureIconLarge");
const gestureName = document.getElementById("gestureName");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statFrames = document.getElementById("statFrames");
const statDetections = document.getElementById("statDetections");
const statFps = document.getElementById("statFps");
const collectStatus = document.getElementById("collectStatus");
const trainingResult = document.getElementById("trainingResult");
const toastContainer = document.getElementById("toastContainer");

// ---- Gesture Emoji Map ----
const gestureEmojiMap = {
    "Open Palm": "🖐️",
    "Fist": "✊",
    "Thumbs Up": "👍",
    "Peace": "✌️",
    "OK": "👌"
};

// ---- MediaPipe Hand Connections for drawing ----
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],      // Thumb
    [0,5],[5,6],[6,7],[7,8],      // Index
    [0,9],[9,10],[10,11],[11,12], // Middle  -- fixed: was [5,9]
    [0,13],[13,14],[14,15],[15,16], // Ring   -- fixed: was [9,13]
    [0,17],[17,18],[18,19],[19,20], // Pinky  -- fixed: was [13,17]
    [5,9],[9,13],[13,17]           // Palm
];

// ---- Initialize ----
document.addEventListener("DOMContentLoaded", () => {
    checkServerStatus();
});

// ---- Server Status Check ----
async function checkServerStatus() {
    try {
        const res = await fetch("/api/stats");
        const data = await res.json();
        if (data.status === "online") {
            statusDot.classList.add("online");
            statusText.textContent = "System Online";
            showToast("GestureSense is ready! Start your camera.", "success");
        }
    } catch (e) {
        statusText.textContent = "Connection Error";
        showToast("Unable to connect to server.", "error");
    }
}

// ---- Camera Control ----
async function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcam.srcObject = videoStream;

        webcam.onloadedmetadata = () => {
            // Set canvas size to match video
            overlay.width = webcam.videoWidth;
            overlay.height = webcam.videoHeight;
        };

        cameraActive = true;
        cameraPlaceholder.style.display = "none";
        startBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
        startBtn.classList.add("btn-danger");
        startBtn.classList.remove("btn-primary");
        flipBtn.style.display = "inline-flex";

        // Start recognition loop
        startRecognition();

        showToast("Camera started successfully!", "success");

    } catch (err) {
        console.error("Camera error:", err);
        showToast("Could not access camera. Please allow camera permissions.", "error");
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    webcam.srcObject = null;
    cameraActive = false;

    if (recognitionLoop) {
        clearInterval(recognitionLoop);
        recognitionLoop = null;
    }

    cameraPlaceholder.style.display = "flex";
    startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
    startBtn.classList.remove("btn-danger");
    startBtn.classList.add("btn-primary");
    flipBtn.style.display = "none";
    gestureBadge.style.display = "none";

    // Clear overlay
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Reset display
    gestureIconLarge.textContent = "❓";
    gestureName.textContent = "No Hand Detected";
    confidenceValue.textContent = "0%";
    confidenceFill.style.width = "0%";
    clearActiveGesture();
}

async function flipCamera() {
    facingMode = facingMode === "user" ? "environment" : "user";
    if (cameraActive) {
        stopCamera();
        await startCamera();
    }
}

// ---- Recognition Loop ----
function startRecognition() {
    // Send frames at ~5 FPS to avoid overloading the server
    recognitionLoop = setInterval(() => {
        if (cameraActive && webcam.readyState >= 2) {
            captureAndRecognize();
        }
    }, 200);
}

async function captureAndRecognize() {
    try {
        // Capture frame
        const canvas = document.createElement("canvas");
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(webcam, 0, 0);
        const imageData = canvas.toDataURL("image/jpeg", 0.7);

        frameCount++;
        fpsCounter++;

        // Calculate FPS
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
            statFps.textContent = fpsCounter;
            fpsCounter = 0;
            lastFpsTime = now;
        }

        statFrames.textContent = frameCount;

        // Send to server
        const res = await fetch("/api/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData })
        });

        const data = await res.json();

        if (data.success) {
            updateGestureDisplay(data);
            if (data.landmarks) {
                drawLandmarks(data.landmarks);
            } else {
                const octx = overlay.getContext("2d");
                octx.clearRect(0, 0, overlay.width, overlay.height);
            }
        }

    } catch (err) {
        console.error("Recognition error:", err);
    }
}

// ---- Update UI ----
function updateGestureDisplay(data) {
    if (data.hand_detected && data.gesture) {
        const emoji = gestureEmojiMap[data.gesture] || "❓";
        const confidence = Math.round(data.confidence * 100);

        // Update badge overlay
        gestureBadge.style.display = "flex";
        gestureEmoji.textContent = emoji;
        gestureLabel.textContent = data.gesture;

        // Update main display
        if (data.gesture !== lastGesture) {
            gestureIconLarge.textContent = emoji;
            gestureIconLarge.classList.remove("detected");
            void gestureIconLarge.offsetWidth; // trigger reflow
            gestureIconLarge.classList.add("detected");
            lastGesture = data.gesture;
        }

        gestureName.textContent = data.gesture;
        confidenceValue.textContent = confidence + "%";
        confidenceFill.style.width = confidence + "%";

        // Color confidence bar based on level
        if (confidence >= 75) {
            confidenceFill.style.background = "linear-gradient(90deg, #10b981, #34d399)";
        } else if (confidence >= 50) {
            confidenceFill.style.background = "linear-gradient(90deg, #f59e0b, #fbbf24)";
        } else {
            confidenceFill.style.background = "linear-gradient(90deg, #ef4444, #f87171)";
        }

        // Highlight active gesture in reference grid
        setActiveGesture(data.gesture);

        detectionCount++;
        statDetections.textContent = detectionCount;

    } else {
        gestureBadge.style.display = "none";
        gestureIconLarge.textContent = "🔍";
        gestureName.textContent = "No Hand Detected";
        confidenceValue.textContent = "0%";
        confidenceFill.style.width = "0%";
        lastGesture = null;
        clearActiveGesture();
    }
}

function setActiveGesture(gestureName) {
    document.querySelectorAll(".gesture-item").forEach(item => {
        if (item.dataset.gesture === gestureName) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
}

function clearActiveGesture() {
    document.querySelectorAll(".gesture-item").forEach(item => {
        item.classList.remove("active");
    });
}

// ---- Draw Hand Landmarks ----
function drawLandmarks(landmarks) {
    const ctx = overlay.getContext("2d");
    const w = overlay.width;
    const h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    if (!landmarks || landmarks.length === 0) return;

    // Draw connections
    ctx.strokeStyle = "rgba(249, 115, 22, 0.7)";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";

    for (const [i, j] of HAND_CONNECTIONS) {
        if (i < landmarks.length && j < landmarks.length) {
            const p1 = landmarks[i];
            const p2 = landmarks[j];
            // Mirror x-coordinate since video is mirrored
            const x1 = (1 - p1.x) * w;
            const y1 = p1.y * h;
            const x2 = (1 - p2.x) * w;
            const y2 = p2.y * h;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    }

    // Draw landmark points
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        const x = (1 - lm.x) * w;
        const y = lm.y * h;

        // Fingertips get special treatment
        const isTip = [4, 8, 12, 16, 20].includes(i);
        const radius = isTip ? 6 : 4;

        // Glow
        if (isTip) {
            ctx.beginPath();
            ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255, 107, 0, 0.25)";
            ctx.fill();
        }

        // Point
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, isTip ? "#fb923c" : "#f97316");
        gradient.addColorStop(1, isTip ? "#f97316" : "#ea580c");
        ctx.fillStyle = gradient;
        ctx.fill();

        // Border
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// ---- Data Collection ----
async function collectSample() {
    if (!cameraActive) {
        showToast("Please start the camera first.", "error");
        return;
    }

    const gesture = document.getElementById("gestureSelect").value;

    // Capture current frame
    const canvas = document.createElement("canvas");
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(webcam, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.8);

    collectStatus.textContent = "Collecting...";

    try {
        const res = await fetch("/api/collect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData, gesture: gesture })
        });

        const data = await res.json();

        if (data.success) {
            collectStatus.textContent = `✅ Collected "${data.gesture}" sample (Total: ${data.total_samples})`;
            showToast(`Sample collected for "${gesture}"!`, "success");
        } else {
            collectStatus.textContent = `❌ ${data.error}`;
            showToast(data.error, "error");
        }
    } catch (err) {
        collectStatus.textContent = "❌ Error collecting sample.";
        showToast("Error collecting sample.", "error");
    }
}

// ---- Model Retraining ----
async function retrainModel() {
    showToast("Retraining model... This may take a moment.", "info");
    trainingResult.style.display = "none";

    try {
        const res = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        const data = await res.json();

        if (data.success) {
            trainingResult.style.display = "block";
            trainingResult.innerHTML = `
                <strong>✅ Model Retrained Successfully!</strong><br>
                Accuracy: <strong>${data.accuracy}%</strong> | 
                Total Samples: <strong>${data.total_samples}</strong> | 
                Collected: <strong>${data.collected_samples}</strong>
            `;
            showToast(`Model retrained! Accuracy: ${data.accuracy}%`, "success");
        } else {
            showToast("Error retraining model: " + data.error, "error");
        }
    } catch (err) {
        showToast("Error retraining model.", "error");
    }
}

// ---- Toast Notifications ----
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = "toastSlideOut 0.35s ease forwards";
        setTimeout(() => toast.remove(), 350);
    }, 3500);
}
