import os
import cv2
from ultralytics import YOLO
from pathlib import Path

STUDENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = STUDENT_DIR/"HW1_data"
# ID_VIDEO_DATA = DATA_ROOT/"id_video_data"
# ID_VIDEOS = {
#     "ID_1": os.path.join(ID_VIDEO_DATA, "4_2_24_B_2.mp4"),
#     "ID_2": os.path.join(ID_VIDEO_DATA, "20_2_24_1.mp4")
# }
OOD_VIDEO_DATA = DATA_ROOT/"ood_video_data"
OOD_VIDEO = {"OOD_1": os.path.join(OOD_VIDEO_DATA, "4_2_24_A_1.mp4")}
MODEL_PATH = "runs/detect/runs_round1/finetune_id_pseudo/weights/best.pt"
OUT_DIR = "./pseudo_labels_ood"
FRAME_STRIDE = 15          # ~2 frames/sec at 30fps -- avoids near-duplicate
                           # frames while still covering the video well
CONF_THRESHOLD = 0.6

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

model = YOLO(MODEL_PATH)
kept_frames, total_frames, kept_boxes = 0, 0, 0

for video_name, path in OOD_VIDEO.items():
    if not os.path.exists(path):
        print(f"[WARN] {video_name} not found at {path}, skipping")
        continue

    cap = cv2.VideoCapture(path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_STRIDE == 0:
            total_frames += 1
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

            if len(results.boxes) > 0:
                h, w = frame.shape[:2]
                lines = []
                for box in results.boxes:
                    cls = int(box.cls.item())
                    xc, yc, bw, bh = box.xywhn[0].tolist()  # already normalized
                    lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                fname = f"{video_name}_frame{frame_idx:06d}"
                cv2.imwrite(os.path.join(OUT_DIR, "images", fname + ".jpg"), frame)
                with open(os.path.join(OUT_DIR, "labels", fname + ".txt"), "w") as f:
                    f.write("\n".join(lines))

                kept_frames += 1
                kept_boxes += len(lines)
        frame_idx += 1
    cap.release()
    print(f"[{video_name}] processed {frame_idx} frames")

print(f"\nSampled {total_frames} frames total, kept {kept_frames} with >= "
      f"{CONF_THRESHOLD} confidence detections ({kept_boxes} pseudo-labeled boxes).")
