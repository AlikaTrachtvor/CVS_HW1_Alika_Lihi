# video.py
import sys
import cv2
from ultralytics import YOLO
from predict import predict_frame

BEST_MODEL = "runs/detect/experiments/finetune_ood/weights/best.pt"

def process_video(video_path, output_path="ood_annotated.mp4", model_path=BEST_MODEL):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Call the core inference function from predict.py
        results, _ = predict_frame(model, frame, conf=0.5)

        # Plot annotations onto the frame
        annotated_frame = results.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Annotated video written to {output_path}")


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "HW1_data/ood_video_data/4_2_24_A_1.mp4"
    process_video(vid)