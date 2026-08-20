# video.py
import argparse
import cv2
from predict import predict_image
from ultralytics import YOLO


def process_video(video_path, model_path, output_path="ood_annotated_check.mp4"):
  model = YOLO(model_path)
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {video_path}")

  fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    results, _ = predict_image(model, frame, conf=0.5)
    annotated_frame = results.plot()
    out.write(annotated_frame)

  cap.release()
  out.release()
  print(f"Annotated video written to {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser( description="Run YOLO prediction on a video using OpenCV.")
  parser.add_argument("--video", required=True, type=str, help="Path to input video file")
  parser.add_argument("--weights",required=True,type=str,help="Path to YOLO model weights (.pt)",)
  parser.add_argument("--output",default="ood_annotated.mp4",type=str,help="Destination path for annotated video",)

  args = parser.parse_args()
  process_video(video_path=args.video, model_path=args.weights, output_path=args.output)