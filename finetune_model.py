import os
import shutil
import argparse
from ultralytics import YOLO

ORIG_TRAIN_IMAGES = "./data_resplit/images/train"
ORIG_TRAIN_LABELS = "./data_resplit/labels/train"
VAL_IMAGES = "./data_resplit/images/val"
VAL_LABELS = "./data_resplit/labels/val"

def merge_dataset(pseudo_dir, merged_root):
    pseudo_imgs = os.path.join(pseudo_dir, "images")
    pseudo_lbls = os.path.join(pseudo_dir, "labels")

    train_img_dst = os.path.join(merged_root, "images", "train")
    train_lbl_dst = os.path.join(merged_root, "labels", "train")
    val_img_dst = os.path.join(merged_root, "images", "val")
    val_lbl_dst = os.path.join(merged_root, "labels", "val")

    for d in [train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst]:
        os.makedirs(d, exist_ok=True)

    # 1. Base human-labeled data
    for fname in os.listdir(ORIG_TRAIN_IMAGES):
        shutil.copy(os.path.join(ORIG_TRAIN_IMAGES, fname), os.path.join(train_img_dst, fname))
    for fname in os.listdir(ORIG_TRAIN_LABELS):
        shutil.copy(os.path.join(ORIG_TRAIN_LABELS, fname), os.path.join(train_lbl_dst, fname))

    # 2. Add pseudo-labeled data
    if os.path.exists(pseudo_imgs):
        for fname in os.listdir(pseudo_imgs):
            shutil.copy(os.path.join(pseudo_imgs, fname), os.path.join(train_img_dst, fname))
        for fname in os.listdir(pseudo_lbls):
            shutil.copy(os.path.join(pseudo_lbls, fname), os.path.join(train_lbl_dst, fname))

    # 3. Clean validation set (human-labeled only)
    for fname in os.listdir(VAL_IMAGES):
        shutil.copy(os.path.join(VAL_IMAGES, fname), os.path.join(val_img_dst, fname))
    for fname in os.listdir(VAL_LABELS):
        shutil.copy(os.path.join(VAL_LABELS, fname), os.path.join(val_lbl_dst, fname))

def write_data_yaml(merged_root):
    path = os.path.join(merged_root, "data.yaml")
    with open(path, "w") as f:
        f.write(f"""path: {os.path.abspath(merged_root)}
train: images/train
val: images/val

names:
  0: Empty
  1: Tweezers
  2: Needle_driver
""")
    return path

def finetune(weights_path, pseudo_dir, merged_root, project_dir, run_name, epochs=80):
    merge_dataset(pseudo_dir, merged_root)
    data_yaml = write_data_yaml(merged_root)

    model = YOLO(weights_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        patience=20,
        project=project_dir,
        name=run_name,
        exist_ok=False,  # <--- Auto-increments (e.g. finetune_id, finetune_id1, finetune_id2) to prevent accidental loss
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

    print(f"\nFine-tuning complete. Run output saved at: {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model with pseudo-labels")
    parser.add_argument("--weights", required=True, help="Base weights path")
    parser.add_argument("--pseudo_dir", required=True, help="Directory of generated pseudo labels")
    parser.add_argument("--data_out", required=True, help="Destination directory for merged dataset")
    parser.add_argument("--project", default="runs_round1", help="Project runs folder")
    parser.add_argument("--name", default="finetune_run", help="Run folder prefix")
    parser.add_argument("--epochs", type=int, default=80)

    args = parser.parse_args()
    finetune(args.weights, args.pseudo_dir, args.data_out, args.project, args.name, args.epochs)