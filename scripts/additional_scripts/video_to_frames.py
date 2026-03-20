import cv2
import os
import argparse


def extract_frames(video_path, target_fps):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("frames", video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise RuntimeError("Could not read video FPS")

    frame_interval = int(round(video_fps / target_fps))
    frame_interval = max(frame_interval, 1)

    print(f"[INFO] Video FPS: {video_fps:.2f}")
    print(f"[INFO] Target FPS: {target_fps}")
    print(f"[INFO] Saving 1 frame every {frame_interval} frames")

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_path = os.path.join(
                output_dir,
                f"frame_{saved_id:04d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()

    print(f"[INFO] Frames saved: {saved_id}")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from video with FPS control"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Target FPS for frame extraction (default: 5)"
    )

    args = parser.parse_args()
    extract_frames(args.video, args.fps)
