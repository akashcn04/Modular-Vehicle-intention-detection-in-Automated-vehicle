import json
from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "models/TLD_indicator_best.pt"

def run_vehicle_intent_detection(image_path, output_json):
    model = YOLO(MODEL_PATH)

    img = cv2.imread(image_path)
    results = model(img)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = model.names[cls_id]

        detections.append({
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 3)
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"[Module2] Saved {len(detections)} detections → {output_json}")
