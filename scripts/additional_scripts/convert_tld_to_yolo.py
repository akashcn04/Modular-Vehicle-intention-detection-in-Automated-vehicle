import json
import shutil
import random
from pathlib import Path

# ================= PATH SETUP =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

TLD_ROOT = DATASET_ROOT / "TLD-LOKI"          # raw dataset
OUT_ROOT = DATASET_ROOT / "TLD_indicator"     # YOLO output

TRAIN_RATIO = 0.8
SEED = 42
random.seed(SEED)

# ================= CLASS MAPPING =================
# Module-2 (LOCKED)
# 0 vehicle | 1 left | 2 right | 3 brake
CATEGORY_NAME_MAP = {
    "vehicle": 0,
    "cars": 0,
    "go": 0,          # rear vehicle body in some exports
    "left": 1,
    "right": 2,
    "brake": 3
}

# ================= CREATE OUTPUT FOLDERS =================
for split in ["train", "val"]:
    (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ================= COLLECT SCENARIOS =================
scenarios = []
for group in TLD_ROOT.glob("group_*"):
    for scenario in group.glob("scenario_*"):
        ann = scenario / "_annotations.coco.json"
        if ann.exists():
            scenarios.append(scenario)

print(f"Found total scenarios: {len(scenarios)}")

random.shuffle(scenarios)
split_idx = int(len(scenarios) * TRAIN_RATIO)

splits = {
    "train": scenarios[:split_idx],
    "val": scenarios[split_idx:]
}

# ================= CONVERSION =================
for split, scenario_list in splits.items():
    print(f"\nProcessing {split}: {len(scenario_list)} scenarios")

    for scenario in scenario_list:
        ann_file = scenario / "_annotations.coco.json"

        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        labels_per_image = {}

        for ann in coco["annotations"]:
            cat_name = categories.get(ann["category_id"])
            if cat_name not in CATEGORY_NAME_MAP:
                continue

            img = images[ann["image_id"]]
            yolo_cls = CATEGORY_NAME_MAP[cat_name]

            x, y, w, h = ann["bbox"]
            iw, ih = img["width"], img["height"]

            xc = (x + w / 2) / iw
            yc = (y + h / 2) / ih
            wn = w / iw
            hn = h / ih

            key = img["file_name"]
            labels_per_image.setdefault(key, []).append(
                f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
            )

        # ================= WRITE FILES =================
        for img_name, yolo_lines in labels_per_image.items():
            if len(yolo_lines) == 0:
                continue

            new_name = f"{scenario.parent.name}_{scenario.name}_{img_name}"

            src_img = scenario / img_name
            if not src_img.exists():
                continue

            dst_img = OUT_ROOT / "images" / split / new_name
            dst_lbl = OUT_ROOT / "labels" / split / (Path(new_name).stem + ".txt")

            shutil.copy(src_img, dst_img)

            with open(dst_lbl, "w") as f:
                f.write("\n".join(yolo_lines))

# ================= CREATE YAML =================
yaml_path = OUT_ROOT / "module2.yaml"
with open(yaml_path, "w") as f:
    f.write(
        f"""path: {OUT_ROOT}
train: images/train
val: images/val

nc: 4
names:
  0: vehicle
  1: left_indicator
  2: right_indicator
  3: brake_light
"""
    )

print("\n✅ Conversion complete.")
