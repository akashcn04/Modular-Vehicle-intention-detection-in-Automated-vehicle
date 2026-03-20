import os
import yaml
import cv2

CLASS_MAP = {
    "Red": 0,
    "RedLeft": 0,
    "RedRight": 0,
    "Yellow": 1,
    "Green": 2
}

ANNOTATION_FILE = "dataset/dataset_train_rgb/train.yaml"
IMAGE_ROOT = "dataset/dataset_train_rgb"
LABEL_DIR = "dataset/labels/train"

os.makedirs(LABEL_DIR, exist_ok=True)

with open(ANNOTATION_FILE) as f:
    data = yaml.safe_load(f)

for item in data:

    img_path = os.path.join(IMAGE_ROOT, item["path"].replace("./", ""))

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    label_name = os.path.basename(img_path).replace(".png", ".txt")
    label_path = os.path.join(LABEL_DIR, label_name)

    with open(label_path, "w") as f:

        for box in item["boxes"]:

            label = box["label"]

            if label not in CLASS_MAP:
                continue

            cls = CLASS_MAP[label]

            xmin = box["x_min"]
            xmax = box["x_max"]
            ymin = box["y_min"]
            ymax = box["y_max"]

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

print("✅ Conversion finished!")
