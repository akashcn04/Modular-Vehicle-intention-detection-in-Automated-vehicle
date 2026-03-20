import xml.etree.ElementTree as ET
import os
import cv2

classes = [
    "vehicle",
    "left_indicator",
    "right_indicator",
    "brake_light",
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green"
]

splits = ["train", "val"]

for split in splits:
    xml_dir = f"labels/{split}"
    img_dir = f"images/{split}"

    if not os.path.exists(xml_dir):
        continue

    print(f"🔄 Processing {split} set")

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_name = root.find("filename").text
        img_path = os.path.join(img_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print("❌ Image not found:", img_path)
            continue

        h, w = img.shape[:2]
        yolo_lines = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in classes:
                continue

            cls_id = classes.index(label)
            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            xmax = float(box.find("xmax").text)
            ymin = float(box.find("ymin").text)
            ymax = float(box.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            yolo_lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
            )

        txt_name = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(xml_dir, txt_name)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("✅ VOC → YOLO conversion completed for train & val.")
