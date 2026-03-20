import csv
import os
import cv2
from collections import defaultdict

LISA_ROOT = "dataset/LISA_Signals"
OUTPUT_ROOT = "dataset/LISA_yolo"

LABEL_MAP = {
    "stop": 4,      # red
    "warning": 5,   # yellow
    "go": 6         # green
}

os.makedirs(f"{OUTPUT_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/labels/train", exist_ok=True)

# --------------------------------------------------
# STEP 1: INDEX ALL IMAGES (THIS FIXES EVERYTHING)
# --------------------------------------------------
print("[INFO] Indexing all LISA images...")

image_index = {}
for root, _, files in os.walk(LISA_ROOT):
    for f in files:
        if f.lower().endswith(".jpg"):
            image_index[f] = os.path.join(root, f)

print(f"[INFO] Total images indexed: {len(image_index)}")

# --------------------------------------------------
# STEP 2: PROCESS ALL ANNOTATION FILES
# --------------------------------------------------
annotations_root = os.path.join(LISA_ROOT, "Annotations")

for seq in os.listdir(annotations_root):
    ann_file = os.path.join(annotations_root, seq, "frameAnnotationsBOX.csv")
    if not os.path.exists(ann_file):
        continue

    print(f"[INFO] Processing annotations: {seq}")

    annotations = defaultdict(list)

    with open(ann_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            label = row["Annotation tag"]
            if label not in LABEL_MAP:
                continue

            filename = os.path.basename(row["Filename"])

            if filename not in image_index:
                continue  # image not found on disk

            xmin = int(row["Upper left corner X"])
            ymin = int(row["Upper left corner Y"])
            xmax = int(row["Lower right corner X"])
            ymax = int(row["Lower right corner Y"])

            annotations[filename].append((label, xmin, ymin, xmax, ymax))

    # --------------------------------------------------
    # STEP 3: WRITE YOLO FILES
    # --------------------------------------------------
    for img_name, boxes in annotations.items():
        img_path = image_index[img_name]
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        # copy image
        out_img_path = os.path.join(OUTPUT_ROOT, "images/train", img_name)
        cv2.imwrite(out_img_path, img)

        label_path = os.path.join(
            OUTPUT_ROOT, "labels/train", img_name.replace(".jpg", ".txt")
        )

        with open(label_path, "w") as f:
            for label, xmin, ymin, xmax, ymax in boxes:
                xc = ((xmin + xmax) / 2) / w
                yc = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                f.write(f"{LABEL_MAP[label]} {xc} {yc} {bw} {bh}\n")

print("[DONE] LISA → YOLO conversion complete.")
