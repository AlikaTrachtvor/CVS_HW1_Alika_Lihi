import os
import shutil
from ultralytics import YOLO

ORIG_TRAIN_IMAGES = "./data_resplit/images/train"
ORIG_TRAIN_LABELS = "./data_resplit/labels/train"
PSEUDO_IMAGES = "./pseudo_labels_id/images"
PSEUDO_LABELS = "./pseudo_labels_id/labels"
VAL_IMAGES = "./data_resplit/images/val"    # keep val = original human-labeled
VAL_LABELS = "./data_resplit/labels/val"    # val, never pseudo-labeled -- this
                                             # is the only trustworthy signal
                                             # for whether fine-tuning helped

MERGED_ROOT = "./data_round1"
ROUND0_WEIGHTS = "runs/detect/runs_round0/initial_supervised/weights/best.pt"


def merge_dataset():
    for split, img_src, lbl_src in [
        ("train_orig", ORIG_TRAIN_IMAGES, ORIG_TRAIN_LABELS),
        ("train_pseudo", PSEUDO_IMAGES, PSEUDO_LABELS),
    ]:
        img_dst = os.path.join(MERGED_ROOT, "images", "train")
        lbl_dst = os.path.join(MERGED_ROOT, "labels", "train")
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)
        for fname in os.listdir(img_src):
            shutil.copy(os.path.join(img_src, fname), os.path.join(img_dst, fname))
        for fname in os.listdir(lbl_src):
            shutil.copy(os.path.join(lbl_src, fname), os.path.join(lbl_dst, fname))

    # val stays as the original, human-labeled val set (copy as-is)
    img_dst = os.path.join(MERGED_ROOT, "images", "val")
    lbl_dst = os.path.join(MERGED_ROOT, "labels", "val")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    for fname in os.listdir(VAL_IMAGES):
        shutil.copy(os.path.join(VAL_IMAGES, fname), os.path.join(img_dst, fname))
    for fname in os.listdir(VAL_LABELS):
        shutil.copy(os.path.join(VAL_LABELS, fname), os.path.join(lbl_dst, fname))

    n_train = len(os.listdir(os.path.join(MERGED_ROOT, "images", "train")))
    n_val = len(os.listdir(os.path.join(MERGED_ROOT, "images", "val")))
    print(f"Merged dataset: {n_train} train images (orig + pseudo), {n_val} val images")


def write_data_yaml():
    path = os.path.join(MERGED_ROOT, "data.yaml")
    with open(path, "w") as f:
        f.write(f"""path: {os.path.abspath(MERGED_ROOT)}
train: images/train
val: images/val

names:
  0: Empty
  1: Tweezers
  2: Needle_driver
""")
    return path


if __name__ == "__main__":
    merge_dataset()
    data_yaml = write_data_yaml()

    model = YOLO(ROUND0_WEIGHTS)  # resume from Round-0 weights, not COCO
    model.train(
        data=data_yaml,
        epochs=80,               # fewer epochs than Round 0 -- fine-tuning,
                                  # not training from scratch
        imgsz=640,
        patience=20,
        project="runs_round1",
        name="finetune_id_pseudo",
        # same augmentation policy as train.py, for consistency
        translate=0.25,
        scale=0.5,
        fliplr=0.5,
        hsv_s=0.9,
        hsv_v=0.4,
        hsv_h=0.015,
        mosaic=1.0,
        mixup=0.0,
        flipud=0.0,
        degrees=0.0,
        val=True,
        plots=True,
    )

    print("\nRound 1 fine-tuning complete.")
    print("Compare runs_round1/finetune_id_pseudo/results.png against "
          "runs_round0/initial_supervised/results.png for the report --"
          " val mAP should hold steady or improve, since val is untouched "
          "human-labeled data. If val mAP drops, the pseudo-label confidence "
          "threshold in generate_pseudo_labels.py was probably too low.")
    print("\nNext: repeat this pattern for the OOD video (generate pseudo-labels "
          "from THIS model on the OOD video, merge again, fine-tune again) --"
          " that final checkpoint is the one you submit as your model weights.")