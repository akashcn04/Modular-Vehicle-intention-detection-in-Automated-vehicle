import cv2
import json
import os
import argparse

from module1_traffic_light import run_traffic_light_detection
from module2_vehicle_intent import run_vehicle_intent_detection
from select_traffic_light import select_traffic_light
from select_vehicle import select_nearest_vehicle
from associate_intent import associate_intent
from rule_engine import decide


# =================================================
# Visualization colors
# =================================================
COLOR_TL = (0, 255, 255)         # Yellow
COLOR_VEHICLE = (0, 255, 0)     # Green
COLOR_BRAKE = (0, 0, 255)       # Red
COLOR_LEFT_IND = (255, 0, 0)    # Blue
COLOR_RIGHT_IND = (255, 255, 0) # Cyan


# =================================================
# Helper: draw bounding box with label offset
# =================================================
def draw_box(img, bbox, color, label=None, label_offset=0):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5 + label_offset, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
def draw_box_label_bottom_right(img, bbox, color, label):
    """
    Draw label at bottom-right corner of a bounding box
    """
    x1, y1, x2, y2 = bbox

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label position (slightly inside bottom-right)
    text_x = max(x2 - 70, x1 + 5)
    text_y = min(y2 - 5, img.shape[0] - 5)

    cv2.putText(
        img,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2
    )


# =================================================
# IoU (for indicator/brake association in debug)
# =================================================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


# =================================================
# Argument parser
# =================================================
parser = argparse.ArgumentParser(
    description="Frame-level collision-aware intention inference"
)
parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to input frame image"
)
args = parser.parse_args()


# =================================================
# Image & run folder setup
# =================================================
IMAGE_PATH = args.image
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

image_name = os.path.basename(IMAGE_PATH)
image_stem = os.path.splitext(image_name)[0]

BASE_RUN_DIR = f"runs/inference/{image_stem}"
RUN_DIR = BASE_RUN_DIR
idx = 1
while os.path.exists(RUN_DIR):
    RUN_DIR = f"{BASE_RUN_DIR}_v{idx}"
    idx += 1

RAW_DIR = f"{RUN_DIR}/raw"
FILTERED_DIR = f"{RUN_DIR}/filtered"
DEBUG_DIR = f"{RUN_DIR}/debug_frames"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

print(f"[INFO] Saving results to: {RUN_DIR}")


# =================================================
# Load image
# =================================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

H, W, _ = img.shape


# =================================================
# Module 1: Traffic light detection
# =================================================
run_traffic_light_detection(
    IMAGE_PATH,
    f"{RAW_DIR}/traffic_light_raw.json"
)


# =================================================
# Module 2: Vehicle + indicator detection
# =================================================
run_vehicle_intent_detection(
    IMAGE_PATH,
    f"{RAW_DIR}/vehicle_intent_raw.json"
)


# =================================================
# Select ego-relevant traffic light
# =================================================
select_traffic_light(
    f"{RAW_DIR}/traffic_light_raw.json",
    f"{FILTERED_DIR}/traffic_light.json",
    W,
    H
)


# =================================================
# Select ego-relevant vehicle
# =================================================
select_nearest_vehicle(
    f"{RAW_DIR}/vehicle_intent_raw.json",
    f"{FILTERED_DIR}/vehicle.json",
    W,
    H
)


# =================================================
# Associate indicator & brake
# =================================================
associate_intent(
    f"{FILTERED_DIR}/vehicle.json",
    f"{RAW_DIR}/vehicle_intent_raw.json",
    f"{FILTERED_DIR}/intent.json"
)


# =================================================
# Rule-based decision
# =================================================
with open(f"{FILTERED_DIR}/traffic_light.json") as f:
    traffic_light = json.load(f)

with open(f"{FILTERED_DIR}/intent.json") as f:
    intents = json.load(f)

intent = intents[0] if intents else None
# get bbox
with open(f"{FILTERED_DIR}/vehicle.json") as f:
    vehicle_data = json.load(f)

bbox = None
if "vehicles" in vehicle_data and len(vehicle_data["vehicles"]) > 0:
    bbox = vehicle_data["vehicles"][0]["bbox"]

decision, sici = decide(
    traffic_light["state"],
    intent,
    bbox=bbox,
    frame_height=H
)

print("FRAME DECISION:", decision, "| SICI:", sici)


# =================================================
# Debug visualization
# =================================================

# ---- Traffic light ----
if "bbox" in traffic_light:
    draw_box(
        img,
        traffic_light["bbox"],
        COLOR_TL,
        f"TL: {traffic_light['state']}"
    )

# ---- Target vehicle ----
with open(f"{FILTERED_DIR}/vehicle.json") as f:
    vehicles = json.load(f)["vehicles"]

for v in vehicles:
    draw_box(
        img,
        v["bbox"],
        COLOR_VEHICLE,
        "TARGET VEHICLE",
        label_offset=0
    )

# ---- Indicator & brake lights (only for target vehicle) ----
with open(f"{RAW_DIR}/vehicle_intent_raw.json") as f:
    all_detections = json.load(f)

for v in vehicles:
    vbox = v["bbox"]

    for d in all_detections:
        if d["class"] not in ["brake_light", "left_indicator", "right_indicator"]:
            continue

        if iou(vbox, d["bbox"]) < 0.3:
            continue

        if d["class"] == "brake_light":
            draw_box_label_bottom_right(
            img,
            d["bbox"],
            COLOR_BRAKE,
            "BRAKE"
            )

        elif d["class"] == "left_indicator":
            draw_box_label_bottom_right(
            img,
            d["bbox"],
            COLOR_LEFT_IND,
            "LEFT"
            )

        elif d["class"] == "right_indicator":
            draw_box_label_bottom_right(
            img,
            d["bbox"],
            COLOR_RIGHT_IND,
            "RIGHT"
            )


# ---- Decision text ----
decision_colors = {
    "SAFE": (0, 255, 0),
    "CAUTION": (0, 255, 255),
    "RISK": (0, 0, 255),
    "UNKNOWN": (255, 255, 255)
}

cv2.putText(
    img,
    f"DECISION: {decision} | UCARI: {sici}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    decision_colors.get(decision, (255, 255, 255)),
    3
)

# ---- Save debug frame ----
debug_path = f"{DEBUG_DIR}/debug.jpg"
cv2.imwrite(debug_path, img)

print(f"[INFO] Debug image saved to: {debug_path}")
