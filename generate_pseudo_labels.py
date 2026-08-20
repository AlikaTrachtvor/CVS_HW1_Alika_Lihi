import os
import argparse
import cv2
from ultralytics import YOLO
from predict import predict_image


def generate_pseudo(model_path, video_path, out_dir, frame_stride=15, conf_threshold=0.6):
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_idx, kept_frames, total_frames, kept_boxes = 0, 0, 0, 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            total_frames += 1
            _, detections = predict_image(model, frame, conf=conf_threshold)

            if len(detections) > 0:
                lines = [f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls, xc, yc, bw, bh, _ in detections]
                fname = f"{video_name}_frame{frame_idx:06d}"

                cv2.imwrite(os.path.join(img_out, f"{fname}.jpg"), frame)
                with open(os.path.join(lbl_out, f"{fname}.txt"), "w") as f:
                    f.write("\n".join(lines))

                kept_frames += 1
                kept_boxes += len(lines)
        frame_idx += 1

    cap.release()
    print(f"[{video_name}] Kept {kept_frames}/{total_frames} frames ({kept_boxes} boxes) -> Saved to {out_dir}")


def generate_pseudo_for_multiple_videos(model_path, video_paths, out_dir, frame_stride=15, conf_threshold=0.6):
    for video_path in video_paths:
        print(f"\nProcessing video: {video_path}")
        generate_pseudo(model_path, video_path, out_dir, frame_stride, conf_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--videos", nargs="+", required=True, help="List of video file paths")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.6)

    args = parser.parse_args()
    generate_pseudo_for_multiple_videos(args.model, args.videos, args.out_dir, args.stride, args.conf)