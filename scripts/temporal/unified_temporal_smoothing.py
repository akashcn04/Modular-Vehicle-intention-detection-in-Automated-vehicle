import os
import json
import argparse


class StaticRiskEstimator:

    def __init__(self):
        self.w_distance = 0.35
        self.w_brake = 0.25
        self.w_indicator = 0.15
        self.w_signal = 0.25

    def compute(self, bbox, brake, left, right,
                traffic_light, frame_height):

        if bbox is None:
            return {"CRI": 0.0, "Decision": "SAFE"}

        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1

        # Distance Risk
        D_risk = min((bbox_height / frame_height) * 2.0, 1.0)

        B_risk = 1.0 if brake else 0.0
        I_risk = 1.0 if (left or right) else 0.0

        S_risk = 0.0
        if traffic_light == "red" and not brake:
            S_risk = 1.0

        CRI = (
            self.w_distance * D_risk +
            self.w_brake * B_risk +
            self.w_indicator * I_risk +
            self.w_signal * S_risk
        )

        CRI = min(CRI, 1.0)

        if CRI < 0.3:
            decision = "SAFE"
        elif CRI < 0.6:
            decision = "CAUTION"
        else:
            decision = "HIGH_RISK"

        return {
            "CRI": round(CRI, 3),
            "Decision": decision
        }


def process_all_frames(inference_root, output_file):

    estimator = StaticRiskEstimator()
    results = []

    frame_folders = sorted([
        f for f in os.listdir(inference_root)
        if f.startswith("frame_")
    ])

    for frame_folder in frame_folders:

        filtered_path = os.path.join(
            inference_root, frame_folder, "filtered"
        )

        vehicle_path = os.path.join(filtered_path, "vehicle.json")
        tl_path = os.path.join(filtered_path, "traffic_light.json")
        intent_path = os.path.join(filtered_path, "intent.json")

        if not os.path.exists(vehicle_path):
            continue

        with open(vehicle_path) as f:
            vehicle_data = json.load(f)

        with open(tl_path) as f:
            tl_data = json.load(f)

        with open(intent_path) as f:
            intent_data = json.load(f)

       # ---- Vehicle JSON ----
        if isinstance(vehicle_data, list):
            vehicle_data = vehicle_data[0] if len(vehicle_data) > 0 else {}

        bbox = vehicle_data.get("bbox")
        frame_height = vehicle_data.get("frame_height", 720)

# ---- Intent JSON ----
        if isinstance(intent_data, list):
            intent_data = intent_data[0] if len(intent_data) > 0 else {}

        brake = intent_data.get("brake", False)
        left = intent_data.get("left_indicator", False)
        right = intent_data.get("right_indicator", False)

# ---- Traffic Light JSON ----
        if isinstance(tl_data, list):
            tl_data = tl_data[0] if len(tl_data) > 0 else {}

        traffic_light = tl_data.get("state", "UNKNOWN")


        risk_output = estimator.compute(
            bbox=bbox,
            brake=brake,
            left=left,
            right=right,
            traffic_light=traffic_light,
            frame_height=frame_height
        )

        results.append({
            "frame": frame_folder,
            "CRI": risk_output["CRI"],
            "risk_decision": risk_output["Decision"]
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Static risk output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_all_frames(args.input, args.output)
