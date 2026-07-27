import sys
from ultralytics import YOLO


def predict_frame(model, frame, conf=0.6):
    """
    Runs prediction on a single frame using a loaded YOLO model instance.
    Returns the raw results object and extracted bounding box data.
    """
    results = model.predict(frame, conf=conf, verbose=False)[0]
    detections = []

    if len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls.item())
            xc, yc, bw, bh = box.xywhn[0].tolist()
            conf_val = float(box.conf.item())
            detections.append((cls, xc, yc, bw, bh, conf_val))

    return results, detections


def main(image_path, model_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, conf=0.5)
    print(f"Predictions saved to {results[0].save_dir}")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    main(img)