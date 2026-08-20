import sys
import argparse
from ultralytics import YOLO


def predict_image(model, image, conf=0.6):
    """
    Runs prediction on a single frame using a loaded YOLO model instance.
    Returns the raw results object and extracted bounding box data.
    """
    results = model.predict(image, conf=conf, verbose=False)[0]
    detections = []

    if len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls.item())
            xc, yc, bw, bh = box.xywhn[0].tolist()
            conf_val = float(box.conf.item())
            detections.append((cls, xc, yc, bw, bh, conf_val))

    return results, detections


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run YOLO prediction on an image.")
  parser.add_argument("--image", required=True, type=str, help="Path to input image file")
  parser.add_argument("--weights",required=True,type=str,help="Path to YOLO model weights (.pt)",)
  parser.add_argument("--conf",default=0.5,type=float,help="Confidence threshold (default: 0.5)",)
  parser.add_argument("--output", default="prediction_output.jpg", type=str,help="Destination path for annotated image",)

  args = parser.parse_args()
  model = YOLO(args.weights)
  results, detections = predict_image(model, args.image, conf=args.conf)

  print(f"Detections found: {len(detections)}")
  for det in detections:
    print(det)
  results.save(filename=args.output)
  print(f"Annotated image saved to {args.output}")