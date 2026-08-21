import os
import io
import json
import socket
import base64
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pickle
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize FastAPI App
app = FastAPI(
    title="GestureSense - Hand Gesture Recognition System",
    description="Real-Time Hand Gesture Detection & Classification using MediaPipe + ML",
    version="1.0.0"
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# MediaPipe Hand Landmarker Initialization (Tasks API v1.0+)
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

hand_landmarker = None
if os.path.exists(HAND_MODEL_PATH):
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
    print(f"[GestureSense] Loaded HandLandmarker model from '{HAND_MODEL_PATH}'")
else:
    print(f"[GestureSense] ERROR: Hand landmarker model not found at '{HAND_MODEL_PATH}'")
    print("[GestureSense] Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

# Gesture Classes
GESTURE_CLASSES = ["Open Palm", "Fist", "Thumbs Up", "Peace", "OK"]

# ML Model path
ML_MODEL_PATH = os.path.join(MODEL_DIR, "gesture_classifier.pkl")

# Load or initialize ML model
gesture_model = None
if os.path.exists(ML_MODEL_PATH):
    with open(ML_MODEL_PATH, "rb") as f:
        gesture_model = pickle.load(f)
    print(f"[GestureSense] Loaded gesture classifier from '{ML_MODEL_PATH}'")
else:
    print("[GestureSense] No trained classifier found. Will train with built-in dataset...")


def extract_landmarks(image_rgb):
    """Extract 21 hand landmarks (63 features: x, y, z for each) from an image using MediaPipe Tasks API."""
    if hand_landmarker is None:
        return None, None

    # Convert numpy array to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Detect hand landmarks
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        hand_lms = result.hand_landmarks[0]  # First hand
        landmarks = []
        # Get wrist position for normalization
        wrist = hand_lms[0]
        for lm in hand_lms:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return landmarks, hand_lms
    return None, None


def compute_finger_states(landmarks_flat):
    """
    Compute finger extension states from normalized landmark coordinates.
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky].
    """
    def get_point(idx):
        return np.array([landmarks_flat[idx*3], landmarks_flat[idx*3+1], landmarks_flat[idx*3+2]])

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    finger_mcps = [2, 5, 9, 13, 17]

    states = []
    for i, (tip, pip_j, mcp) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
        tip_pt = get_point(tip)
        pip_pt = get_point(pip_j)
        mcp_pt = get_point(mcp)

        tip_dist = np.linalg.norm(tip_pt)
        pip_dist = np.linalg.norm(pip_pt)

        if i == 0:  # Thumb: use x-distance from palm center
            palm_center = get_point(9)
            tip_lateral = abs(tip_pt[0] - palm_center[0])
            mcp_lateral = abs(mcp_pt[0] - palm_center[0])
            states.append(tip_lateral > mcp_lateral * 1.2)
        else:
            states.append(tip_dist > pip_dist * 1.05)

    return states


def rule_based_classify(landmarks_flat):
    """Classify gesture using rule-based heuristics as fallback."""
    if landmarks_flat is None:
        return None, 0.0

    states = compute_finger_states(landmarks_flat)
    thumb, index, middle, ring, pinky = states

    if all(states):
        return "Open Palm", 0.85

    if not any(states):
        return "Fist", 0.85

    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up", 0.82

    if not thumb and index and middle and not ring and not pinky:
        return "Peace", 0.82

    def get_point(idx):
        return np.array([landmarks_flat[idx*3], landmarks_flat[idx*3+1], landmarks_flat[idx*3+2]])

    thumb_tip = get_point(4)
    index_tip = get_point(8)
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)

    if thumb_index_dist < 0.08 and middle and ring and pinky:
        return "OK", 0.80

    extended_count = sum(states)
    if extended_count >= 4:
        return "Open Palm", 0.60
    elif extended_count == 0:
        return "Fist", 0.60
    elif extended_count == 1 and thumb:
        return "Thumbs Up", 0.55
    elif extended_count == 2 and index and middle:
        return "Peace", 0.55
    else:
        return "Open Palm", 0.40


def generate_synthetic_data():
    """Generate synthetic landmark data for training the ML classifier."""
    np.random.seed(42)
    data = []
    labels = []

    num_samples_per_class = 200

    for _ in range(num_samples_per_class):
        # --- Open Palm: all fingers extended ---
        landmarks = np.zeros(63)
        landmarks[3:6] = [0.08, -0.05, 0]
        landmarks[6:9] = [0.15, -0.10, 0]
        landmarks[9:12] = [0.20, -0.15, 0]
        landmarks[12:15] = [0.25, -0.20, 0]
        landmarks[15:18] = [0.05, -0.08, 0]
        landmarks[18:21] = [0.06, -0.18, 0]
        landmarks[21:24] = [0.06, -0.28, 0]
        landmarks[24:27] = [0.06, -0.36, 0]
        landmarks[27:30] = [0.0, -0.08, 0]
        landmarks[30:33] = [0.0, -0.19, 0]
        landmarks[33:36] = [0.0, -0.30, 0]
        landmarks[36:39] = [0.0, -0.38, 0]
        landmarks[39:42] = [-0.05, -0.08, 0]
        landmarks[42:45] = [-0.05, -0.18, 0]
        landmarks[45:48] = [-0.05, -0.28, 0]
        landmarks[48:51] = [-0.05, -0.35, 0]
        landmarks[51:54] = [-0.10, -0.07, 0]
        landmarks[54:57] = [-0.10, -0.15, 0]
        landmarks[57:60] = [-0.10, -0.23, 0]
        landmarks[60:63] = [-0.10, -0.30, 0]
        noise = np.random.normal(0, 0.015, 63)
        data.append(landmarks + noise)
        labels.append(0)

    for _ in range(num_samples_per_class):
        # --- Fist: all fingers curled ---
        landmarks = np.zeros(63)
        landmarks[3:6] = [0.05, -0.03, 0]
        landmarks[6:9] = [0.08, -0.04, 0]
        landmarks[9:12] = [0.06, -0.03, 0.02]
        landmarks[12:15] = [0.04, -0.02, 0.03]
        landmarks[15:18] = [0.04, -0.06, 0]
        landmarks[18:21] = [0.05, -0.10, 0]
        landmarks[21:24] = [0.04, -0.08, 0.03]
        landmarks[24:27] = [0.03, -0.06, 0.04]
        landmarks[27:30] = [0.0, -0.06, 0]
        landmarks[30:33] = [0.0, -0.10, 0]
        landmarks[33:36] = [0.0, -0.08, 0.03]
        landmarks[36:39] = [0.0, -0.06, 0.04]
        landmarks[39:42] = [-0.04, -0.06, 0]
        landmarks[42:45] = [-0.04, -0.10, 0]
        landmarks[45:48] = [-0.04, -0.08, 0.03]
        landmarks[48:51] = [-0.04, -0.06, 0.04]
        landmarks[51:54] = [-0.08, -0.05, 0]
        landmarks[54:57] = [-0.08, -0.08, 0]
        landmarks[57:60] = [-0.08, -0.06, 0.03]
        landmarks[60:63] = [-0.08, -0.05, 0.04]
        noise = np.random.normal(0, 0.015, 63)
        data.append(landmarks + noise)
        labels.append(1)

    for _ in range(num_samples_per_class):
        # --- Thumbs Up: only thumb extended ---
        landmarks = np.zeros(63)
        landmarks[3:6] = [0.06, -0.06, 0]
        landmarks[6:9] = [0.10, -0.14, 0]
        landmarks[9:12] = [0.12, -0.22, 0]
        landmarks[12:15] = [0.13, -0.30, 0]
        landmarks[15:18] = [0.04, -0.06, 0]
        landmarks[18:21] = [0.05, -0.10, 0]
        landmarks[21:24] = [0.04, -0.08, 0.03]
        landmarks[24:27] = [0.03, -0.06, 0.04]
        landmarks[27:30] = [0.0, -0.06, 0]
        landmarks[30:33] = [0.0, -0.10, 0]
        landmarks[33:36] = [0.0, -0.08, 0.03]
        landmarks[36:39] = [0.0, -0.06, 0.04]
        landmarks[39:42] = [-0.04, -0.06, 0]
        landmarks[42:45] = [-0.04, -0.10, 0]
        landmarks[45:48] = [-0.04, -0.08, 0.03]
        landmarks[48:51] = [-0.04, -0.06, 0.04]
        landmarks[51:54] = [-0.08, -0.05, 0]
        landmarks[54:57] = [-0.08, -0.08, 0]
        landmarks[57:60] = [-0.08, -0.06, 0.03]
        landmarks[60:63] = [-0.08, -0.05, 0.04]
        noise = np.random.normal(0, 0.015, 63)
        data.append(landmarks + noise)
        labels.append(2)

    for _ in range(num_samples_per_class):
        # --- Peace: index and middle extended ---
        landmarks = np.zeros(63)
        landmarks[3:6] = [0.05, -0.03, 0]
        landmarks[6:9] = [0.08, -0.04, 0]
        landmarks[9:12] = [0.06, -0.03, 0.02]
        landmarks[12:15] = [0.04, -0.02, 0.03]
        landmarks[15:18] = [0.04, -0.08, 0]
        landmarks[18:21] = [0.05, -0.18, 0]
        landmarks[21:24] = [0.05, -0.28, 0]
        landmarks[24:27] = [0.05, -0.36, 0]
        landmarks[27:30] = [0.0, -0.08, 0]
        landmarks[30:33] = [0.0, -0.19, 0]
        landmarks[33:36] = [0.0, -0.30, 0]
        landmarks[36:39] = [0.0, -0.38, 0]
        landmarks[39:42] = [-0.04, -0.06, 0]
        landmarks[42:45] = [-0.04, -0.10, 0]
        landmarks[45:48] = [-0.04, -0.08, 0.03]
        landmarks[48:51] = [-0.04, -0.06, 0.04]
        landmarks[51:54] = [-0.08, -0.05, 0]
        landmarks[54:57] = [-0.08, -0.08, 0]
        landmarks[57:60] = [-0.08, -0.06, 0.03]
        landmarks[60:63] = [-0.08, -0.05, 0.04]
        noise = np.random.normal(0, 0.015, 63)
        data.append(landmarks + noise)
        labels.append(3)

    for _ in range(num_samples_per_class):
        # --- OK: thumb and index tips touching, middle/ring/pinky extended ---
        landmarks = np.zeros(63)
        landmarks[3:6] = [0.06, -0.04, 0]
        landmarks[6:9] = [0.09, -0.08, 0]
        landmarks[9:12] = [0.08, -0.12, 0]
        landmarks[12:15] = [0.06, -0.14, 0.01]
        landmarks[15:18] = [0.04, -0.08, 0]
        landmarks[18:21] = [0.05, -0.14, 0]
        landmarks[21:24] = [0.06, -0.16, 0.01]
        landmarks[24:27] = [0.06, -0.14, 0.02]
        landmarks[27:30] = [0.0, -0.08, 0]
        landmarks[30:33] = [0.0, -0.19, 0]
        landmarks[33:36] = [0.0, -0.30, 0]
        landmarks[36:39] = [0.0, -0.38, 0]
        landmarks[39:42] = [-0.05, -0.08, 0]
        landmarks[42:45] = [-0.05, -0.18, 0]
        landmarks[45:48] = [-0.05, -0.28, 0]
        landmarks[48:51] = [-0.05, -0.35, 0]
        landmarks[51:54] = [-0.10, -0.07, 0]
        landmarks[54:57] = [-0.10, -0.15, 0]
        landmarks[57:60] = [-0.10, -0.23, 0]
        landmarks[60:63] = [-0.10, -0.30, 0]
        noise = np.random.normal(0, 0.012, 63)
        data.append(landmarks + noise)
        labels.append(4)

    return np.array(data), np.array(labels)


def train_model():
    """Train the gesture classifier using synthetic data."""
    global gesture_model

    print("[GestureSense] Generating synthetic training data...")
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[GestureSense] Training Random Forest classifier...")
    gesture_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    gesture_model.fit(X_train, y_train)

    y_pred = gesture_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[GestureSense] Model accuracy on test set: {acc*100:.1f}%")

    with open(ML_MODEL_PATH, "wb") as f:
        pickle.dump(gesture_model, f)
    print(f"[GestureSense] Model saved to '{ML_MODEL_PATH}'")

    return acc


# Train model if not loaded
if gesture_model is None:
    train_model()


# --- Request Models ---
class ImageRequest(BaseModel):
    image: str


class CollectRequest(BaseModel):
    image: str
    gesture: str


# --- Routes ---

@app.get("/")
async def serve_dashboard():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_file, media_type="text/html")


@app.get("/api/stats")
async def get_system_stats():
    sample_counts = {}
    for gesture in GESTURE_CLASSES:
        data_file = os.path.join(DATA_DIR, f"{gesture.lower().replace(' ', '_')}.json")
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                samples = json.load(f)
            sample_counts[gesture] = len(samples)
        else:
            sample_counts[gesture] = 0

    return {
        "status": "online",
        "model": "Random Forest (MediaPipe Landmarks)",
        "classes": GESTURE_CLASSES,
        "model_loaded": gesture_model is not None,
        "hand_detector_loaded": hand_landmarker is not None,
        "sample_counts": sample_counts
    }


@app.post("/api/recognize")
async def recognize_gesture(req: ImageRequest):
    """Recognize hand gesture from a webcam frame."""
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)

        landmarks, raw_landmarks = extract_landmarks(img_rgb)

        if landmarks is None:
            return {
                "success": True,
                "hand_detected": False,
                "gesture": None,
                "confidence": 0.0,
                "landmarks": None
            }

        # Classify using ML model
        if gesture_model is not None:
            features = np.array(landmarks).reshape(1, -1)
            prediction = gesture_model.predict(features)[0]
            probabilities = gesture_model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
            gesture_name = GESTURE_CLASSES[prediction]

            if confidence < 0.45:
                rule_gesture, rule_conf = rule_based_classify(landmarks)
                if rule_gesture and rule_conf > confidence:
                    gesture_name = rule_gesture
                    confidence = rule_conf
        else:
            gesture_name, confidence = rule_based_classify(landmarks)

        # Format landmarks for frontend visualization
        landmark_points = []
        if raw_landmarks:
            for lm in raw_landmarks:
                landmark_points.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

        return {
            "success": True,
            "hand_detected": True,
            "gesture": gesture_name,
            "confidence": confidence,
            "landmarks": landmark_points
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/collect")
async def collect_training_data(req: CollectRequest):
    """Collect labeled training samples for gesture recognition."""
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_rgb = np.array(pil_img)

        landmarks, _ = extract_landmarks(img_rgb)

        if landmarks is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No hand detected in the image."}
            )

        gesture = req.gesture
        if gesture not in GESTURE_CLASSES:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Invalid gesture class: {gesture}"}
            )

        data_file = os.path.join(DATA_DIR, f"{gesture.lower().replace(' ', '_')}.json")
        existing_data = []
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                existing_data = json.load(f)

        existing_data.append(landmarks)
        with open(data_file, "w") as f:
            json.dump(existing_data, f)

        return {
            "success": True,
            "gesture": gesture,
            "total_samples": len(existing_data)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/train")
async def retrain_model():
    """Retrain the model with collected + synthetic data."""
    try:
        X_synth, y_synth = generate_synthetic_data()

        X_collected = []
        y_collected = []
        for idx, gesture in enumerate(GESTURE_CLASSES):
            data_file = os.path.join(DATA_DIR, f"{gesture.lower().replace(' ', '_')}.json")
            if os.path.exists(data_file):
                with open(data_file, "r") as f:
                    samples = json.load(f)
                for sample in samples:
                    X_collected.append(sample)
                    y_collected.append(idx)

        if X_collected:
            X_all = np.vstack([X_synth, np.array(X_collected)])
            y_all = np.concatenate([y_synth, np.array(y_collected)])
        else:
            X_all = X_synth
            y_all = y_synth

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        global gesture_model
        gesture_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        gesture_model.fit(X_train, y_train)

        y_pred = gesture_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        with open(ML_MODEL_PATH, "wb") as f:
            pickle.dump(gesture_model, f)

        return {
            "success": True,
            "accuracy": round(acc * 100, 1),
            "total_samples": len(X_all),
            "collected_samples": len(X_collected)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def find_free_port(starting_port=8000):
    for port in range(starting_port, starting_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return starting_port


if __name__ == "__main__":
    import uvicorn
    target_port = find_free_port(8000)
    print(f"Starting GestureSense on http://127.0.0.1:{target_port}")
    uvicorn.run(app, host="127.0.0.1", port=target_port)
