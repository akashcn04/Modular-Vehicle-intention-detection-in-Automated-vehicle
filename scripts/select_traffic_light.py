import json
import os


def select_traffic_light(input_json, output_json,
                         image_width, image_height,
                         selected_vehicle_json=None):
    """
    Indian-robust traffic light selection.

    Improvements:
    - Select traffic light aligned with selected vehicle
    - Must be above vehicle
    - Prefer highest confidence among aligned candidates
    """

    with open(input_json) as f:
        detections = json.load(f)

    vehicle_center_x = image_width / 2
    vehicle_center_y = image_height

    # Load selected vehicle if available
    if selected_vehicle_json and os.path.exists(selected_vehicle_json):
        with open(selected_vehicle_json) as vf:
            vehicle_data = json.load(vf)

        if "vehicles" in vehicle_data and len(vehicle_data["vehicles"]) > 0:
            vbox = vehicle_data["vehicles"][0]["bbox"]
            vx1, vy1, vx2, vy2 = vbox
            vehicle_center_x = (vx1 + vx2) / 2
            vehicle_center_y = vy2

    candidates = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        confidence = d.get("confidence", 0)

        # Must be in upper half (signals are above road)
        if cy > 0.6 * image_height:
            continue

        # Must be above selected vehicle
        if cy > vehicle_center_y:
            continue

        horizontal_alignment = abs(cx - vehicle_center_x)

        # Alignment threshold (wider for Indian setups)
        alignment_threshold = 0.3 * image_width

        if horizontal_alignment < alignment_threshold:
            # Prefer better alignment and higher confidence
            alignment_score = horizontal_alignment
            candidates.append((alignment_score, -confidence, d))

    if not candidates:
        result = {"state": "UNKNOWN"}
    else:
        # Best alignment, then highest confidence
        selected = sorted(candidates)[0][2]
        result = {
            "state": selected["class"].split("_")[-1].upper(),
            "confidence": selected["confidence"],
            "bbox": selected["bbox"]
        }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
