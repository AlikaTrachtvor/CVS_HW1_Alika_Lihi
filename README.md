# Surgical Tool Detection with Semi-Supervised Learning (HW1)

##  Model Weights Download

Download the final fine-tuned model weights:
* **https://drive.google.com/file/d/1NEqzDgbGdmGPOkz8BWhVQLkHw4WxywNP/view?usp=sharing**

---

## Environment Setup & Installation

Clone the repository and install all required dependencies:

```bash
# Install dependencies
pip install -r requirements.txt
```
## Repository Structure
```bash
├── EDA.py                     # Exploratory Data Analysis & visual distribution checks
├── finetune_model.py          # Merges datasets and fine-tunes the model over the merged data
├── generate_pseudo_labels.py  # Pseudo-label generator for unlabeled videos
├── predict.py                 # Inference script for single images
├── split_data.py              # Splits data into train/val sets
├── train.py                   # Train initial supervised model
├── video.py                   # OpenCV-based inference and video writer
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation and instructions
```
### Single Image Prediction (`predict.py`)
Runs detection on a single image and prints bounding boxes in YOLO format. 
```bash
python predict.py --image path/to/image.jpg --weights path/to/best.pt --output prediction_output.jpg
```
Options:
- --image: Path to the input image file (Required).  
- --weights: Path to YOLO model weights (.pt) (Required).  
- --conf: Confidence threshold (default: 0.5).
- --output: Path to save the annotated prediction image (default: prediction_output.jpg).

### Video Prediction (`video.py`)
Processes a surgery video frame-by-frame using OpenCV and generates an annotated output video with bounding boxes and labels
```bash
python video.py --video video.mp4 --weights path/to/best.pt --output video_annotated.mp4
```
Options:
- --video: Path to input video file (Required).  
- --weights: Path to YOLO model weights (.pt) (Required).  
- --output: Path to save the annotated video (default: ood_annotated.mp4).

### Additional Scripts
The rest of the files in this directory helped us create the final submitted model and video prediction. 
They are not needed if you do not wish to retrain the current model/train a completely new one. However, if you do wish to do so, here is a short explanation of each file:

#### `EDA.py`
Runs a basic EDA on the initially provided data. **Note**: This file uses absolute paths, so change the paths according to your configuration.

#### `finetune_model.py`
Merges the labels data with pseudo-labeled data, and finetune the current model.
```bash
python finetune_model.py --weights path/to/best.pt --pseudo_dir path/to/pseudo_dir --data_out path/to/merged_dir
```
Options:
- --weights: Path to YOLO model weights (.pt) (Required).
- --pseudo_dir: Path to the pseudo-labeled images (Required).  
- --data_out: Path to the merged data (Required).
- --project: YOLO attribute
- --name: YOLO attribute
- --epochs: (default: 80)

#### `generate_pseudo_labels.py`
Receives a model and generates pseudo labels on a given data.
```bash
python generate_pseudo_labels.py --mode path/to/best.pt --videos path/to/videos --out_dir path/to/pseudo_dir
```
Options:
- --model: Path to YOLO model weights (.pt) (Required).
- --videos: Path to the videos to be pseudo-labeled (Required, can be multiple).  
- --out_dir: Path to the output dir (Required).
- --stride: To avoid similar frames (default: 15) 
- --conf: Confidence threshold (default: 0.6)

#### `split_data.py`
Reshuffles the labeled data to train/val set using a multilabel stratified k-folds.

#### `train.py`
Trains the initial supervised YOLO model
