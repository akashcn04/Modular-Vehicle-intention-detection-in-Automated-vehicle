import os
import json
import argparse
import subprocess


def run_frame_inference(frames_dir, video_name):
    frames = sorted([
        f for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    ])

    if not frames:
        raise RuntimeError("No frames found")

    output_dir = f"runs/video/{video_name}"
    frames_out_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    decisions = []

    total = len(frames)
    print(f"[INFO] Total frames to process: {total}")

    for idx, frame in enumerate(frames):
        print(f"[INFO] Processing frame {idx+1}/{total}: {frame}")

        frame_path = os.path.join(frames_dir, frame)

        # -------------------------------------------------
        # Run frame-level inference (black box)
        # -------------------------------------------------
        subprocess.run(
            [
                "python3",
                "scripts/frame_inference.py",
                "--image",
                frame_path
            ],
            check=True
        )

        frame_id = os.path.splitext(frame)[0]
        frame_run_dir = f"runs/inference/{frame_id}"

        # -------------------------------------------------
        # Move frame results into video directory
        # -------------------------------------------------
        final_frame_dir = os.path.join(frames_out_dir, frame_id)
        os.rename(frame_run_dir, final_frame_dir)

        filtered_dir = os.path.join(final_frame_dir, "filtered")

        traffic_light_path = os.path.join(filtered_dir, "traffic_light.json")
        vehicle_path = os.path.join(filtered_dir, "vehicle.json")

        if os.path.exists(vehicle_path):
            with open(vehicle_path) as f:
                vehicles = json.load(f).get("vehicles", [])
                vehicle_present = len(vehicles) > 0
        else:
            vehicle_present = False


        # -------------------------------------------------
        # Read traffic light safely
        # -------------------------------------------------
        if os.path.exists(traffic_light_path):
            with open(traffic_light_path) as f:
                tl_data = json.load(f)
            tl_state = tl_data.get("state", "UNKNOWN")
        else:
            tl_state = "UNKNOWN"

        # -------------------------------------------------
        # Read intent safely
        # -------------------------------------------------
        # -------------------------------------------------
# Read intent safely (from vehicle.json)
# -------------------------------------------------
        if os.path.exists(vehicle_path):
            with open(vehicle_path) as f:
                    vehicle_data = json.load(f)
            vehicles = vehicle_data.get("vehicles", [])

            if vehicles:
                intent = vehicles[0].get("intent", None)
            else:
                intent = None
        else:
            intent = None

        # -------------------------------------------------
        # Rule-based decision
        # -------------------------------------------------
        from rule_engine import decide
        decision = decide(tl_state, intent)

        # -------------------------------------------------
        # Save per-frame summary
        # -------------------------------------------------
        decisions.append({
            "frame": frame,
            "traffic_light": tl_state,
            "vehicle_present": vehicle_present,
            "decision": decision
        })

    # -------------------------------------------------
    # Save all frame decisions
    # -------------------------------------------------
    with open(os.path.join(output_dir, "frame_decisions.json"), "w") as f:
        json.dump(decisions, f, indent=2)

    print("[INFO] Video inference complete")
    print(f"[INFO] Results saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run frame-level inference on video frames"
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Path to extracted frames directory"
    )

    args = parser.parse_args()

    video_name = os.path.basename(args.frames.rstrip("/"))
    run_frame_inference(args.frames, video_name)
