import json
import os

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter + 1e-6)

def associate_intent(vehicle_json, detections_json, output_json):
    with open(vehicle_json) as f:
        vehicles = json.load(f)["vehicles"]

    with open(detections_json) as f:
        detections = json.load(f)

    result = []

    for v in vehicles:
        intent = {"left_indicator": False, "right_indicator": False, "brake": False}
        vbox = v["bbox"]

        for d in detections:
            if iou(vbox, d["bbox"]) > 0.3:
                if d["class"] == "left_indicator":
                    intent["left_indicator"] = True
                elif d["class"] == "right_indicator":
                    intent["right_indicator"] = True
                elif d["class"] == "brake_light":
                    intent["brake"] = True

        result.append(intent)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
