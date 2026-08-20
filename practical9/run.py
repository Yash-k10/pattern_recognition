import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def main():
    print("=" * 65)
    print(" PATTERN RECOGNITION - PRACTICAL 09: FACE MASK DETECTION SYSTEM ")
    print("=" * 65)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    classes = ["with_mask", "without_mask", "mask_incorrect"]

    for c in classes:
        os.makedirs(os.path.join(dataset_dir, c), exist_ok=True)

    # 1. Dataset Check / Generation
    np.random.seed(42)
    torch.manual_seed(42)

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

    num_samples = 150
    print("[1/5] Checking / Generating synthetic face dataset...")
    for c in classes:
        c_path = os.path.join(dataset_dir, c)
        if len(os.listdir(c_path)) < num_samples:
            for i in range(num_samples):
                img = generate_face_image(c, i)
                img.save(os.path.join(c_path, f"{c}_{i:03d}.png"))
    print(f"      Dataset ready in '{dataset_dir}'")

    # 2. PyTorch Data Loaders
    class FaceMaskDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_paths = []
            self.labels = []
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            
            for c in classes:
                c_dir = os.path.join(root_dir, c)
                for fname in os.listdir(c_dir):
                    if fname.endswith(".png") or fname.endswith(".jpg"):
                        self.image_paths.append(os.path.join(c_dir, fname))
                        self.labels.append(self.class_to_idx[c])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            image = Image.open(path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FaceMaskDataset(dataset_dir, transform=transform_train)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    val_dataset.dataset.transform = transform_eval
    test_dataset.dataset.transform = transform_eval

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. CNN Model Definition
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceMaskCNN(num_classes=3).to(device)
    model_path = os.path.join(script_dir, "facemask_cnn_model.pth")

    if os.path.exists(model_path):
        print(f"[2/5] Loading pre-trained CNN model from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[2/5] Training CNN Model...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            print(f"      Epoch [{epoch+1:02d}/{num_epochs:02d}] Train Acc: {correct/total*100:.2f}%")
        torch.save(model.state_dict(), model_path)
        print(f"      Model weights saved to '{model_path}'")

    # 4. Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"[3/5] Test Accuracy: {test_acc * 100:.2f}%")

    # 5. Public Safety Surveillance Simulator
    print("[4/5] Running Public Safety Surveillance Frame Inference...")
    bg = Image.new("RGB", (600, 400), color=(220, 225, 230))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, 600, 45], fill=(30, 40, 60))
    draw.text((15, 14), "PUBLIC SAFETY SURVEILLANCE FEED - CAM_04 (MAIN ENTRANCE)", fill=(255, 255, 255))
    
    face_locations = [
        ("with_mask", 10, (50, 70)),
        ("without_mask", 25, (230, 70)),
        ("mask_incorrect", 40, (410, 70)),
        ("with_mask", 55, (50, 220)),
        ("without_mask", 70, (230, 220)),
        ("with_mask", 85, (410, 220)),
    ]
    face_boxes = []
    for c_name, idx, (x, y) in face_locations:
        f_img = generate_face_image(c_name, idx)
        bg.paste(f_img, (x, y))
        face_boxes.append((x, y, 128, 128))

    raw_path = os.path.join(script_dir, "surveillance_raw.png")
    bg.save(raw_path)

    img_bgr = cv2.imread(raw_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    compliant_count, non_compliant_count = 0, 0
    color_map = {
        "with_mask": (0, 220, 0),
        "without_mask": (220, 0, 0),
        "mask_incorrect": (220, 220, 0)
    }

    for (x, y, w, h) in face_boxes:
        face_crop = img_rgb[y:y+h, x:x+w]
        pil_crop = Image.fromarray(face_crop)
        tensor_crop = transform_eval(pil_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_crop)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, dim=1)
            conf_val = conf.item() * 100
            pred_idx = pred.item()
        
        pred_class = classes[pred_idx]
        box_color = color_map[pred_class]
        
        if pred_class == "with_mask":
            label_text = f"Mask ({conf_val:.1f}%)"
            compliant_count += 1
        elif pred_class == "without_mask":
            label_text = f"NO MASK ({conf_val:.1f}%)"
            non_compliant_count += 1
        else:
            label_text = f"INCORRECT ({conf_val:.1f}%)"
            non_compliant_count += 1

        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), box_color, 3)
        cv2.rectangle(img_rgb, (x, y-26), (x+w, y), box_color, -1)
        cv2.putText(img_rgb, label_text, (x+4, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(img_rgb, label_text, (x+4, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    total_faces = len(face_boxes)
    compliance_rate = (compliant_count / total_faces * 100) if total_faces > 0 else 0.0
    status_msg = "SAFE: High Compliance" if compliance_rate >= 70 else "WARNING: Low Compliance"
    status_col = (0, 200, 0) if compliance_rate >= 70 else (220, 0, 0)

    cv2.rectangle(img_rgb, (0, 350), (600, 400), (20, 20, 30), -1)
    dash_text = f"MONITORED: {total_faces} | COMPLIANT: {compliant_count} | VIOLATIONS: {non_compliant_count} | RATE: {compliance_rate:.1f}%"
    cv2.putText(img_rgb, dash_text, (15, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(img_rgb, f"STATUS: {status_msg}", (15, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 2)

    out_img_path = os.path.join(script_dir, "surveillance_detection_output.png")
    cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"[5/5] Surveillance output saved to '{out_img_path}'")
    print("=" * 65)
    print(" EXECUTION COMPLETED SUCCESSFULLY! ")
    print("=" * 65)

if __name__ == "__main__":
    main()
