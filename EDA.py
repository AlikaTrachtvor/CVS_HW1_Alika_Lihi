import os
import glob
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STUDENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = STUDENT_DIR/"HW1_data"
LABELED_DATA_ROOT = DATA_ROOT/"labeled_image_data"
ID_VIDEO_DATA = DATA_ROOT/"id_video_data"
OOD_VIDEO_DATA = DATA_ROOT/"ood_video_data"

IMG_DIRS = {"train": os.path.join(LABELED_DATA_ROOT, "images", "train"),
            "val":   os.path.join(LABELED_DATA_ROOT, "images", "val")}
LBL_DIRS = {"train": os.path.join(LABELED_DATA_ROOT, "labels", "train"),
            "val":   os.path.join(LABELED_DATA_ROOT, "labels", "val")}

VIDEO_PATHS = {
    "ID_1": os.path.join(ID_VIDEO_DATA, "4_2_24_B_2.mp4"),
    "ID_2": os.path.join(ID_VIDEO_DATA, "20_2_24_1.mp4"),
    "OOD_1": os.path.join(OOD_VIDEO_DATA, "4_2_24_A_1.mp4")
}

# From notes.json
CLASS_NAMES = {0: "Empty", 1: "Tweezers", 2: "Needle_driver"}

OUT_DIR = "./eda_outputs"
SEED = 42
N_SAMPLES_GRID = 9  # 3x3 grid
N_SAMPLE_FRAMES = 9          # frames per video for the sample grid
N_FRAMES_FOR_STATS = 60      # frames per video sampled for brightness/color stats


random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def find_image_for_label(label_path, img_dir):
    stem = os.path.splitext(os.path.basename(label_path))[0]
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        cand = os.path.join(img_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def parse_label_file(path):
    """Returns list of (class_id, xc, yc, w, h) floats."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def load_split(split):
    img_dir, lbl_dir = IMG_DIRS[split], LBL_DIRS[split]
    label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
    records = []
    for lf in label_files:
        img_path = find_image_for_label(lf, img_dir)
        boxes = parse_label_file(lf)
        if img_path is not None:
            h_img, w_img = cv2.imread(img_path).shape[:2]
        else:
            h_img, w_img = None, None
        records.append({
            "split": split,
            "label_file": lf,
            "image_file": img_path,
            "img_w": w_img,
            "img_h": h_img,
            "boxes": boxes,
            "num_boxes": len(boxes),
            "classes_present": sorted(set(b[0] for b in boxes)),
        })
    return records


def main_labeled():
    all_records = []
    for split in ["train", "val"]:
        recs = load_split(split)
        print(f"[{split}] {len(recs)} labeled images found")
        all_records.extend(recs)

    if not all_records:
        print("No records found - check DATA_ROOT / folder paths.")
        return

    # ---------------- per-image stats table ----------------
    df_rows = []
    for r in all_records:
        df_rows.append({
            "split": r["split"],
            "label_file": os.path.basename(r["label_file"]),
            "image_file": os.path.basename(r["image_file"]) if r["image_file"] else None,
            "img_w": r["img_w"],
            "img_h": r["img_h"],
            "num_boxes": r["num_boxes"],
            "classes_present": ",".join(str(c) for c in r["classes_present"]),
            "missing_image": r["image_file"] is None,
        })
    df = pd.DataFrame(df_rows)
    df.to_csv(os.path.join(OUT_DIR, "label_stats.csv"), index=False)

    # ---------------- class distribution ----------------
    class_counts = {split: Counter() for split in ["train", "val"]}
    box_dims = defaultdict(list)  # class_id -> list of (w_px, h_px, aspect, xc, yc) normalized+abs
    for r in all_records:
        for cls, xc, yc, w, h in r["boxes"]:
            class_counts[r["split"]][cls] += 1
            w_px = w * (r["img_w"] or 1)
            h_px = h * (r["img_h"] or 1)
            box_dims[cls].append({
                "w_norm": w, "h_norm": h, "area_norm": w * h,
                "aspect": w / h if h > 0 else np.nan,
                "w_px": w_px, "h_px": h_px,
                "xc": xc, "yc": yc, "split": r["split"],
            })

    all_classes = sorted(set(list(class_counts["train"].keys()) + list(class_counts["val"].keys())))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(all_classes))
    width = 0.35
    train_vals = [class_counts["train"].get(c, 0) for c in all_classes]
    val_vals = [class_counts["val"].get(c, 0) for c in all_classes]
    ax.bar(x - width/2, train_vals, width, label="train")
    ax.bar(x + width/2, val_vals, width, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_NAMES.get(c, str(c)) for c in all_classes])
    ax.set_ylabel("Instance count")
    ax.set_title("Class distribution (instances) by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # ---------------- bbox size distributions ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for cls in all_classes:
        dims = box_dims[cls]
        if not dims:
            continue
        widths = [d["w_norm"] for d in dims]
        heights = [d["h_norm"] for d in dims]
        areas = [d["area_norm"] for d in dims]
        label = CLASS_NAMES.get(cls, str(cls))
        axes[0].hist(widths, bins=20, alpha=0.5, label=label)
        axes[1].hist(heights, bins=20, alpha=0.5, label=label)
        axes[2].hist(areas, bins=20, alpha=0.5, label=label)
    axes[0].set_title("BBox width (normalized)")
    axes[1].set_title("BBox height (normalized)")
    axes[2].set_title("BBox area (normalized)")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bbox_size_distribution.png"), dpi=150)
    plt.close(fig)

    # ---------------- aspect ratio ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls in all_classes:
        dims = box_dims[cls]
        aspects = [d["aspect"] for d in dims if not np.isnan(d["aspect"])]
        if aspects:
            ax.hist(aspects, bins=20, alpha=0.5, label=CLASS_NAMES.get(cls, str(cls)))
    ax.set_title("BBox aspect ratio (w/h) by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bbox_aspect_ratio.png"), dpi=150)
    plt.close(fig)

    # ---------------- center heatmap (spatial bias) ----------------
    fig, axes = plt.subplots(1, len(all_classes), figsize=(5 * len(all_classes), 4))
    if len(all_classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, all_classes):
        dims = box_dims[cls]
        xs = [d["xc"] for d in dims]
        ys = [d["yc"] for d in dims]
        ax.hist2d(xs, ys, bins=20, range=[[0, 1], [0, 1]])
        ax.invert_yaxis()
        ax.set_title(f"Center heatmap: {CLASS_NAMES.get(cls, str(cls))}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bbox_center_heatmap.png"), dpi=150)
    plt.close(fig)

    # ---------------- sample grids with boxes drawn ----------------
    # BGR colors, one per real class: Empty=yellow, Tweezers=green, Needle_driver=red
    colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (0, 0, 255)}
    for split in ["train", "val"]:
        split_records = [r for r in all_records if r["split"] == split and r["image_file"]]
        if not split_records:
            continue
        sample = random.sample(split_records, min(N_SAMPLES_GRID, len(split_records)))
        n = len(sample)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, r in zip(axes, sample):
            img = cv2.imread(r["image_file"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img.shape[:2]
            for cls, xc, yc, w, h in r["boxes"]:
                x1 = int((xc - w / 2) * w_img)
                y1 = int((yc - h / 2) * h_img)
                x2 = int((xc + w / 2) * w_img)
                y2 = int((yc + h / 2) * h_img)
                color = colors.get(cls, (255, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, CLASS_NAMES.get(cls, str(cls)), (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            ax.imshow(img)
            ax.set_title(os.path.basename(r["image_file"]), fontsize=8)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"sample_grid_{split}.png"), dpi=150)
        plt.close(fig)

    # ---------------- text summary ----------------
    lines = []
    lines.append("Labeled Data EDA Summary")
    for split in ["train", "val"]:
        n_imgs = sum(1 for r in all_records if r["split"] == split)
        n_boxes = sum(r["num_boxes"] for r in all_records if r["split"] == split)
        lines.append(f"{split}: {n_imgs} images, {n_boxes} total boxes, "
                     f"{n_boxes / n_imgs:.2f} boxes/image (avg)")
    lines.append("")
    lines.append("Class instance counts (train / val):")
    for c in all_classes:
        lines.append(f"  {CLASS_NAMES.get(c, str(c))}: "
                     f"{class_counts['train'].get(c, 0)} / {class_counts['val'].get(c, 0)}")
    lines.append("")
    no_box = sum(1 for r in all_records if r["num_boxes"] == 0)
    multi_class = sum(1 for r in all_records if len(r["classes_present"]) > 1)
    lines.append(f"Images with 0 boxes: {no_box}")
    lines.append(f"Images with >1 class present: {multi_class}")
    missing = df["missing_image"].sum()
    lines.append(f"Label files with no matching image: {missing}")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)

    print(f"\nAll outputs saved to: {os.path.abspath(OUT_DIR)}")

def get_metadata(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps else None
    cap.release()
    return {"fps": fps, "n_frames": n_frames, "width": w, "height": h, "duration_sec": duration}

def sample_frame_indices(n_frames, n_samples):
    if n_frames <= 0:
        return []
    n_samples = min(n_samples, n_frames)
    return np.linspace(0, n_frames - 1, n_samples, dtype=int)


def read_frames(path, indices):
    cap = cv2.VideoCapture(path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def video_main():
    meta_rows = []
    all_frames_for_stats = {}

    for name, path in VIDEO_PATHS.items():
        if not os.path.exists(path):
            print(f"[WARN] {name}: file not found at {path}, skipping")
            continue
        meta = get_metadata(path)
        if meta is None:
            print(f"[WARN] {name}: could not open video")
            continue
        meta["video"] = name
        meta_rows.append(meta)
        print(f"[{name}] {meta}")

        # sample grid
        sample_idx = sample_frame_indices(meta["n_frames"], N_SAMPLE_FRAMES)
        frames = read_frames(path, sample_idx)
        if frames:
            cols = 3
            rows = int(np.ceil(len(frames) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)
            for ax, fr in zip(axes, frames):
                ax.imshow(fr)
                ax.axis("off")
            for ax in axes[len(frames):]:
                ax.axis("off")
            fig.suptitle(f"Sample frames: {name}")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"sample_frames_{name}.png"), dpi=150)
            plt.close(fig)

        # frames for brightness/color stats (separate, denser sample)
        stat_idx = sample_frame_indices(meta["n_frames"], N_FRAMES_FOR_STATS)
        all_frames_for_stats[name] = read_frames(path, stat_idx)

    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(os.path.join(OUT_DIR, "video_metadata.csv"), index=False)

    if not all_frames_for_stats:
        print("No videos loaded - check VIDEO_PATHS.")
        return

    # ---------------- brightness distribution (domain shift indicator) ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frames in all_frames_for_stats.items():
        brightness = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).mean() for f in frames]
        ax.hist(brightness, bins=20, alpha=0.5, label=name, density=True)
    ax.set_title("Frame brightness distribution per video")
    ax.set_xlabel("Mean grayscale intensity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "brightness_hist.png"), dpi=150)
    plt.close(fig)

    # ---------------- RGB channel histograms ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    channel_names = ["R", "G", "B"]
    for ch in range(3):
        for name, frames in all_frames_for_stats.items():
            vals = np.concatenate([f[:, :, ch].ravel() for f in frames])
            axes[ch].hist(vals, bins=30, alpha=0.4, label=name, density=True)
        axes[ch].set_title(f"{channel_names[ch]} channel")
        axes[ch].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "color_hist.png"), dpi=150)
    plt.close(fig)

    # ---------------- saturation histograms ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, frames in all_frames_for_stats.items():
        sats = []
        for f in frames:
            hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
            sats.append(hsv[:, :, 1].mean())
        ax.hist(sats, bins=20, alpha=0.5, label=name, density=True)
    ax.set_title("Mean saturation per frame, by video")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "saturation_hist.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll outputs saved to: {os.path.abspath(OUT_DIR)}")

def main():
    main_labeled()
    video_main()

if __name__ == "__main__":
    main()