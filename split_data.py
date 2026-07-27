import os
import glob
import shutil

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path

STUDENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = STUDENT_DIR/"HW1_data"
LABELED_DATA_ROOT = DATA_ROOT/"labeled_image_data"
ID_VIDEO_DATA = DATA_ROOT/"id_video_data"
OOD_VIDEO_DATA = DATA_ROOT/"ood_video_data"

IMG_DIRS = [os.path.join(LABELED_DATA_ROOT, "images", "train"), os.path.join(LABELED_DATA_ROOT, "images", "val")]
LBL_DIRS = [os.path.join(LABELED_DATA_ROOT, "labels", "train"), os.path.join(LABELED_DATA_ROOT, "labels", "val")]

OUT_ROOT = "./data_resplit"   # new images/{train,val} + labels/{train,val}
N_CLASSES = 3                 # Empty, Tweezers, Needle_driver
N_FOLDS = 5                   # ~14 images/fold; fold 0 used as val below
VAL_FOLD = 0
SEED = 42


def find_image(stem, img_dir):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def multihot_labels(label_path):
    vec = np.zeros(N_CLASSES, dtype=int)
    if not os.path.exists(label_path):
        return vec
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cls = int(float(line.split()[0]))
                vec[cls] = 1
    return vec


def main():
    # gather every (image, label) pair across both existing splits
    pairs = []
    for img_dir, lbl_dir in zip(IMG_DIRS, LBL_DIRS):
        for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lbl_path))[0]
            img_path = find_image(stem, img_dir)
            if img_path:
                pairs.append((img_path, lbl_path))

    print(f"Total labeled images pooled: {len(pairs)}")

    Y = np.array([multihot_labels(lbl) for _, lbl in pairs])
    print("Total instances per class across full pool:", Y.sum(axis=0))

    mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(mskf.split(np.zeros(len(pairs)), Y))

    val_idx = folds[VAL_FOLD][1]
    train_idx = [i for i in range(len(pairs)) if i not in set(val_idx)]

    print(f"\nNew split -> train: {len(train_idx)} images, val: {len(val_idx)} images")
    print("Train class instance counts:", Y[train_idx].sum(axis=0))
    print("Val   class instance counts:", Y[val_idx].sum(axis=0))

    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    # write out new folder structure
    for split, idx_list in [("train", train_idx), ("val", val_idx)]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)
        for i in idx_list:
            img_path, lbl_path = pairs[i]
            shutil.copy(img_path, os.path.join(OUT_ROOT, "images", split, os.path.basename(img_path)))
            shutil.copy(lbl_path, os.path.join(OUT_ROOT, "labels", split, os.path.basename(lbl_path)))

    print(f"\nNew stratified split written to: {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()