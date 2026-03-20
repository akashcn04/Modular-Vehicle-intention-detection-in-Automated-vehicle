import json
import os


def select_nearest_vehicle(input_json, output_json, image_width, image_height):
    """
    Soft ego-relevance scoring:
    EgoScore = proximity - lambda * horizontal_offset
    """

    with open(input_json) as f:
        detections = json.load(f)

    candidates = []

    image_center_x = image_width / 2
    lambda_penalty = 0.6  # horizontal penalty weight

    for d in detections:
        if d["class"] != "vehicle":
            continue

        x1, y1, x2, y2 = d["bbox"]

        bx = (x1 + x2) / 2
        by = y2

        # Ignore very far vehicles (top half)
        if by < 0.4 * image_height:
            continue

        # ------------------------------
        # Proximity Score (0 to 1)
        # ------------------------------
        proximity = by / image_height

        # ------------------------------
        # Horizontal Offset (0 to 0.5)
        # ------------------------------
        horizontal_offset = abs(bx - image_center_x) / image_width

        # ------------------------------
        # Ego Relevance Score
        # ------------------------------
        ego_score = proximity - lambda_penalty * horizontal_offset

        candidates.append((ego_score, d))

    selected_vehicles = []

    if candidates:
        # Select highest ego score
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_vehicles.append(candidates[0][1])

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"vehicles": selected_vehicles}, f, indent=2)
