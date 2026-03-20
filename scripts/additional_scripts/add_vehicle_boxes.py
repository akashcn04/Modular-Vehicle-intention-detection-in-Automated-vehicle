from pathlib import Path

# ========= CONFIG =========
LABEL_ROOT = Path("dataset/TLD_indicator/labels")
SPLITS = ["train", "val"]

PADDING = 0.05   # normalized padding
VEHICLE_CLASS = "0"
LIGHT_CLASSES = {"1", "2", "3"}

def clamp(v):
    return max(0.0, min(1.0, v))

print("===== AUTO VEHICLE BOX GENERATION =====")

for split in SPLITS:
    label_dir = LABEL_ROOT / split
    print(f"\nProcessing split: {split}")

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        if not raw_lines:
            continue

        parsed = []
        for l in raw_lines:
            parts = l.split()
            if len(parts) != 5:
                continue  # skip malformed lines
            parsed.append(parts)

        if not parsed:
            continue

        classes = [p[0] for p in parsed]

        # If vehicle already exists → skip
        if VEHICLE_CLASS in classes:
            continue

        # Collect valid light boxes
        light_boxes = []
        for p in parsed:
            if p[0] in LIGHT_CLASSES:
                try:
                    xc, yc, w, h = map(float, p[1:])
                    light_boxes.append((xc, yc, w, h))
                except:
                    continue

        if not light_boxes:
            continue

        # Convert to corner coordinates
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for xc, yc, w, h in light_boxes:
            xmins.append(xc - w / 2)
            ymins.append(yc - h / 2)
            xmaxs.append(xc + w / 2)
            ymaxs.append(yc + h / 2)

        xmin = clamp(min(xmins) - PADDING)
        ymin = clamp(min(ymins) - PADDING)
        xmax = clamp(max(xmaxs) + PADDING)
        ymax = clamp(max(ymaxs) + PADDING)

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        vehicle_line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

        # Write back: vehicle first, then original lines
        with open(label_file, "w") as f:
            f.write(vehicle_line + "\n")
            for p in parsed:
                f.write(" ".join(p) + "\n")

    print(f"✔ Completed split: {split}")

print("\n✅ Vehicle bounding boxes added successfully.")
