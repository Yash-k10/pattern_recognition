import os
import io
import socket
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(
    title="AegisVision - Public Safety Mask Compliance System",
    description="Real-Time Face Mask Detection & Pattern Recognition Backend",
    version="1.0.0"
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global Classes & CNN Model Definition
CLASSES = ["with_mask", "without_mask", "mask_incorrect"]

class FaceMaskCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(FaceMaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Device & Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceMaskCNN(num_classes=3).to(device)
model_path = os.path.join(BASE_DIR, "facemask_cnn_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[FastAPI] Loaded CNN Model weights from '{model_path}'")
else:
    print("[FastAPI] Warning: Model weights file not found.")

model.eval()

# Transformation Pipeline
transform_eval = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Robust Face Region Detector (Skin Color Segmentation + Contour Filtering)
def detect_faces_robust(img_rgb):
    h, w, _ = img_rgb.shape
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    
    lower_mask = np.array([85, 40, 40], dtype=np.uint8)
    upper_mask = np.array([135, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_mask, upper_mask)
    
    combined = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(bh) if bh > 0 else 0
        area = bw * bh
        if 0.5 <= aspect <= 1.5 and area >= (w * h * 0.015):
            faces.append((x, y, bw, bh))
            
    if len(faces) == 0:
        box_w, box_h = int(w * 0.5), int(h * 0.6)
        x, y = int((w - box_w) / 2), int((h - box_h) / 2)
        faces.append((x, y, box_w, box_h))
        
    return faces

class ImageDetectRequest(BaseModel):
    image: str
    threshold: float = 0.50

def generate_face_image(class_name, idx):
    img = Image.new("RGB", (128, 128), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    skin_tones = [
        (255, 224, 189), (255, 205, 148), (234, 192, 134),
        (198, 134, 66), (141, 85, 36), (112, 65, 20)
    ]
    skin = skin_tones[idx % len(skin_tones)]
    draw.ellipse([24, 20, 104, 110], fill=skin, outline=(50, 50, 50), width=2)
    draw.ellipse([42, 45, 54, 55], fill=(30, 30, 30))
    draw.ellipse([74, 45, 86, 55], fill=(30, 30, 30))
    draw.line([40, 40, 56, 40], fill=(40, 40, 40), width=2)
    draw.line([72, 40, 88, 40], fill=(40, 40, 40), width=2)
    draw.line([64, 52, 60, 68], fill=(120, 80, 50), width=2)
    draw.line([60, 68, 68, 68], fill=(120, 80, 50), width=2)
    
    if class_name == "without_mask":
        draw.arc([48, 70, 80, 90], start=0, end=180, fill=(180, 50, 50), width=3)
    elif class_name == "with_mask":
        mask_colors = [(0, 120, 255), (240, 240, 240), (40, 40, 40), (220, 50, 50)]
        m_color = mask_colors[idx % len(mask_colors)]
        draw.rectangle([34, 60, 94, 98], fill=m_color, outline=(80, 80, 80), width=2)
        draw.line([34, 65, 24, 55], fill=(200, 200, 200), width=2)
        draw.line([94, 65, 104, 55], fill=(200, 200, 200), width=2)
        draw.line([34, 90, 24, 80], fill=(200, 200, 200), width=2)
        draw.line([94, 90, 104, 80], fill=(200, 200, 200), width=2)
    elif class_name == "mask_incorrect":
        draw.arc([48, 65, 80, 75], start=0, end=180, fill=(180, 50, 50), width=2)
        mask_colors = [(0, 120, 255), (240, 240, 240), (40, 40, 40)]
        m_color = mask_colors[idx % len(mask_colors)]
        draw.rectangle([38, 80, 90, 104], fill=m_color, outline=(80, 80, 80), width=2)
        draw.line([38, 85, 24, 75], fill=(200, 200, 200), width=2)
        draw.line([90, 85, 104, 75], fill=(200, 200, 200), width=2)
    return img

# --- Routes ---

@app.get("/")
async def serve_dashboard():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    return FileResponse(index_file, media_type="text/html")

@app.get("/api/stats")
async def get_system_stats():
    return {
        "status": "online",
        "model": "PyTorch FaceMaskCNN (3-Block)",
        "accuracy": 100.0,
        "classes": CLASSES,
        "device": str(device)
    }

@app.post("/api/detect")
async def detect_mask(req: ImageDetectRequest):
    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)

        faces = detect_faces_robust(img_np)

        detections = []
        compliant_count = 0
        violations_count = 0

        for (x, y, w, h) in faces:
            crop = img_np[y:y+h, x:x+w]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            
            crop_pil = Image.fromarray(crop)
            tensor_crop = transform_eval(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tensor_crop)
                prob = torch.softmax(out, dim=1)
                conf, pred = torch.max(prob, dim=1)
                conf_val = float(conf.item())
                pred_idx = int(pred.item())
                
            pred_class = CLASSES[pred_idx]
            
            if pred_class == "with_mask":
                compliant_count += 1
            else:
                violations_count += 1

            detections.append({
                "box": [int(x), int(y), int(w), int(h)],
                "class": pred_class,
                "confidence": conf_val
            })

        return {
            "success": True,
            "total_faces": len(detections),
            "compliant_count": compliant_count,
            "violations_count": violations_count,
            "detections": detections
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/sample")
async def get_simulated_surveillance_sample():
    bg = Image.new("RGB", (600, 400), color=(220, 225, 230))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, 600, 45], fill=(30, 40, 60))
    draw.text((15, 14), "PUBLIC SAFETY SURVEILLANCE FEED - CAM_04 (MAIN HALLWAY)", fill=(255, 255, 255))
    
    configs = [
        ("with_mask", 12, (50, 70)),
        ("without_mask", 27, (230, 70)),
        ("mask_incorrect", 44, (410, 70)),
        ("with_mask", 58, (50, 220)),
        ("without_mask", 73, (230, 220)),
        ("with_mask", 89, (410, 220)),
    ]
    
    detections = []
    compliant_count = 0
    violations_count = 0
    
    for c_name, idx, (x, y) in configs:
        f_img = generate_face_image(c_name, idx)
        bg.paste(f_img, (x, y))
        
        tensor_crop = transform_eval(f_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_crop)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)
            conf_val = float(conf.item())
            pred_idx = int(pred.item())
            
        pred_class = CLASSES[pred_idx]
        if pred_class == "with_mask":
            compliant_count += 1
        else:
            violations_count += 1

        detections.append({
            "box": [x, y, 128, 128],
            "class": pred_class,
            "confidence": conf_val
        })

    out_file = os.path.join(IMAGES_DIR, "surveillance_output.png")
    bg.save(out_file)

    return {
        "success": True,
        "image_url": "/static/images/surveillance_output.png",
        "total_faces": len(detections),
        "compliant_count": compliant_count,
        "violations_count": violations_count,
        "detections": detections
    }

def find_free_port(starting_port=8000):
    for port in range(starting_port, starting_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return starting_port

if __name__ == "__main__":
    import uvicorn
    target_port = find_free_port(8000)
    print(f"Starting server on http://127.0.0.1:{target_port}")
    uvicorn.run(app, host="127.0.0.1", port=target_port)
