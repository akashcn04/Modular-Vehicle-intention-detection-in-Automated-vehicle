import os
import json
import argparse


# =================================================
# Continuous Risk Estimator
# =================================================
class ContinuousRiskEstimator:
    def __init__(self, fps=30):
        self.prev_center_y = None
        self.prev_time = None
        self.prev_cri = 0
        self.fps = fps

        self.w_distance = 0.25
        self.w_velocity = 0.25
        self.w_brake = 0.15
        self.w_indicator = 0.10
        self.w_signal = 0.25

        self.beta = 0.7
        self.V_MAX = 200
        self.TTC_MAX = 5

    def compute(self, bbox, brake, left, right,
                traffic_light, frame_height, frame_index):

        if bbox is None:
            return {"CRI": 0.0, "Decision": "SAFE", "TTC": None}

        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        center_y = (y1 + y2) / 2
        current_time = frame_index / self.fps

        # Distance Risk
        D_risk = min((bbox_height / frame_height) * 2.0, 1.0)

        V_risk = 0.0
        TTC = None
        V_rel = 0.0

        if self.prev_center_y is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                V_rel = (center_y - self.prev_center_y) / dt

                if V_rel > 0:
                    V_risk = min(V_rel / self.V_MAX, 1.0)

                    approx_distance = frame_height - center_y
                    if V_rel > 1e-3:
                        TTC = min(approx_distance / V_rel, self.TTC_MAX)

        B_risk = 1.0 if brake else 0.0
        I_risk = 1.0 if (left or right) else 0.0

        S_risk = 0.0
        if traffic_light == "red" and V_rel > 0 and not brake:
            S_risk = 1.0

        CRI_raw = (
            self.w_distance * D_risk +
            self.w_velocity * V_risk +
            self.w_brake * B_risk +
            self.w_indicator * I_risk +
            self.w_signal * S_risk
        )

        CRI_raw = min(CRI_raw, 1.0)

        CRI_smooth = (
            self.beta * self.prev_cri +
            (1 - self.beta) * CRI_raw
        )

        self.prev_center_y = center_y
        self.prev_time = current_time
        self.prev_cri = CRI_smooth

        if CRI_smooth < 0.3:
            decision = "SAFE"
        elif CRI_smooth < 0.6:
            decision = "CAUTION"
        else:
            decision = "HIGH_RISK"

        return {
            "CRI": round(CRI_smooth, 3),
            "Decision": decision,
            "TTC": None if TTC is None else round(TTC, 2)
        }


# =================================================
# Main smoothing across frame folders
# =================================================
def unified_smoothing(inference_root, output_file):

    frames_data = []

    risk_estimator = ContinuousRiskEstimator(fps=30)

    frame_folders = sorted([
        f for f in os.listdir(inference_root)
        if f.startswith("frame_")
    ])

    for idx, frame_folder in enumerate(frame_folders):

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

        # Extract values (adjust keys if needed)
        bbox = vehicle_data.get("bbox")
        frame_height = vehicle_data.get("frame_height", 720)

        brake = intent_data.get("brake", False)
        left = intent_data.get("left_indicator", False)
        right = intent_data.get("right_indicator", False)

        traffic_light = tl_data.get("state", "UNKNOWN")

        risk_output = risk_estimator.compute(
            bbox=bbox,
            brake=brake,
            left=left,
            right=right,
            traffic_light=traffic_light,
            frame_height=frame_height,
            frame_index=idx
        )

        frames_data.append({
            "frame": frame_folder,
            "traffic_light": traffic_light,
            "brake": brake,
            "left_indicator": left,
            "right_indicator": right,
            "CRI": risk_output["CRI"],
            "TTC": risk_output["TTC"],
            "risk_decision": risk_output["Decision"]
        })

    with open(output_file, "w") as f:
        json.dump(frames_data, f, indent=2)

    print(f"[INFO] Risk output saved to {output_file}")


# =================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to runs/inference folder")
    parser.add_argument("--output", required=True,
                        help="Path to save final risk json")
    args = parser.parse_args()

    unified_smoothing(args.input, args.output)
