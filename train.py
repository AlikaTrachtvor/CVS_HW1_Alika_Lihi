from ultralytics import YOLO

DATA_YAML = "./data.yaml"
MODEL = "yolov8n.pt"   # COCO-pretrained; small enough to fine-tune on ~60 images.
                       # 'n' (nano) chosen over 's/m' because the dataset is
                       # tiny -- a bigger model overfits even faster here.

EPOCHS = 150
IMG_SIZE = 640
PATIENCE = 30          # early stopping given how easily this will overfit

results = YOLO(MODEL).train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    patience=PATIENCE,
    project="runs_round0",
    name="initial_supervised",

    # ---- Augmentation policy, tied directly to EDA findings ----

    # Finding: bbox center-heatmap showed Tweezers/Needle_driver boxes
    # clustered in different, fixed regions of the frame (positional bias).
    # A model trained on only 61 images can easily learn "location -> class"
    # as a shortcut instead of learning tool shape. These three params push
    # boxes around the frame during training to break that shortcut.
    translate=0.25,     # up from default 0.10
    scale=0.5,           # default; size/area already had wide overlap between
                         # classes (finding #6/#7), so we don't need to push
                         # scale jitter further than default
    fliplr=0.5,          # horizontal flip further scrambles left/right bias

    # Finding: saturation histograms showed the OOD video's saturation
    # distribution cleanly separated from both ID videos, while brightness/
    # color shift was actually more present *within* the ID videos
    # themselves (finding #11/#12). So: push saturation jitter harder than
    # default to bridge the OOD gap, keep brightness/hue jitter closer to
    # default since that variation is already naturally present in-domain.
    hsv_s=0.9,           # up from default 0.7 (fraction, 0-1)
    hsv_v=0.4,           # default (brightness) -- already varies within ID data
    hsv_h=0.015,         # default (hue)

    # Finding: >100 boxes but only ~65 images with an average of 2.2
    # boxes/image and frequent hand+tool co-occurrence (finding #3: 65
    # images have >1 class). Mosaic helps synthesize more varied
    # co-occurrence patterns and object density from limited images.
    mosaic=1.0,
    mixup=0.0,           # mixup (blending 2 full images) tends to hurt on
                         # very small datasets with fine-grained classes
                         # (Tweezers vs Needle_driver differ in shape, not
                         # color/texture -- mixup blurs that signal)

    # Orientation: camera is fixed/consistent across surgeries (assignment
    # notes leg suturing from a fixed setup), so we do NOT enable vertical
    # flip or large rotation -- that would create unrealistic hand/tool
    # orientations not seen at inference time.
    flipud=0.0,
    degrees=0.0,

    val=True,
    plots=True,          # saves PR curves, confusion matrix, per-class mAP,
                          # and loss curves needed for the report
)

print("\nTraining complete.")
print("Best weights saved at: runs_round0/initial_supervised/weights/best.pt")
print("Loss/mAP curves (results.png) and confusion matrix are in the same folder --"
      " use these directly for the report's 'Train + valid loss/mAP graphs' section.")